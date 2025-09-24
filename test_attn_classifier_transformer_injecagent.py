import os
import json
from json import JSONDecoder
import gc
import copy
import string
import math

from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
import numpy as np
import matplotlib.pyplot as plt

from utils import get_simulated_attacker_tool_response, get_score
from output_parsing import evaluate_output_json, evaluate_output_fineutned_qwen, evaluate_output_fineutned_llama, evaluate_output_react
from prompts.agent_prompts import PROMPT_DICT
from pastalib.pasta_IPI import PASTA_IPI_adv_inst
from config import model_paths, special_token_ids
from sklearn.metrics import roc_curve
import torch.nn.functional as F
import re
import jinja2
from collections import Counter
from evaluate import load
bertscore = load("metrics/bertscore")
import spacy
nlp = spacy.load("en_core_web_sm")

from utils_for_cls_and_loc_injecagent import load_data, load_model, get_tool_dict, get_sentence_attn_tool, attn_map_segment_each_sentence, update_item




class BiLSTM(nn.Module):
    def __init__(self, input_size, max_seq_len, hidden_size=256, d_seg=256, num_layers=2, num_classes=2):
        super(BiLSTM, self).__init__()
        self.segment_embedding = nn.Embedding(4, d_seg)  # 3个段落编号

        self.lstm = nn.LSTM(
            input_size=input_size + d_seg,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向 → 隐藏状态 × 2

    def forward(self, x):
        mask = (x.abs().sum(dim=-1) >= 1e-10)    # [batch_size, seq_len], 填充位置为 0，非填充为 1
        lengths = torch.tensor([sum(i) for i in mask])

        segment_ids = x[:,:,-1].int()
        x = x[:,:,:-1]

        sorted_lengths, indices = torch.sort(lengths, descending=True)
        x = x[indices]
        segment_ids = segment_ids[indices]  # 段落ID同步排序

        p_emb = self.segment_embedding(segment_ids)  # (batch_size, seq_len, d_seg)
        x_with_p = torch.cat([x, p_emb], dim=2)  # (batch_size, seq_len, input_size + d_seg)
        x_packed = pack_padded_sequence(x_with_p, sorted_lengths.cpu(), batch_first=True)
        out_packed, (hn, _) = self.lstm(x_packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)
        _, reverse_indices = torch.sort(indices)
        out = out[reverse_indices]
        hn = hn[:, reverse_indices]

        final_hidden_state = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)  # [batch_size, hidden_dim * 2]
        logits = self.fc(final_hidden_state)
        return logits

    def predict(self, x, device=None):
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            logits = self.forward(x)  # (batch_size, num_labels)
            probabilities = torch.softmax(logits, dim=1)  # (batch_size, num_labels)
            predictions = torch.argmax(probabilities, dim=1)  # (batch_size)

        return predictions


class TransformerWithGAP(nn.Module):
    def __init__(self, input_size, max_seq_len, d_model=256, d_seg=64, nhead=8, num_layers=6, num_labels=2):
        super(TransformerWithGAP, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model-d_seg)    # 输入投影层
        self.segment_embedding = nn.Embedding(4, d_seg)   # 段位置嵌入
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))  # 可学习位置编码
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.pool = nn.AdaptiveMaxPool1d(1)  # 自适应池化层
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_labels)
        )

    def forward(self, x):
        mask = (x.abs().sum(dim=-1) < 1e-10)    # [batch_size, seq_len], 填充位置为 1，非填充为 0

        segment_ids = x[:,:,-1].int()
        x = x[:,:,:-1]
        
        x = self.input_proj(x)  # [batch_size, seq_len, d_model-d_seg]
        segment_emb = self.segment_embedding(segment_ids)  # [batch_size, seq_len, d_seg]
        x = torch.cat([x, segment_emb], dim=2)  # [batch_size, seq_len, d_model]

        x = x + self.positional_encoding[:, :x.size(1), :]  # 添加位置编码
        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # [batch_size, seq_len, d_model]
        
        x = x.permute(0, 2, 1)  # [batch_size, d_model, seq_len] 适配池化层
        x = self.pool(x).squeeze(-1)  # [batch_size, d_model]
        x = self.classifier(x)  # [batch_size, num_classes]
        return x

    def predict(self, x, device=None):
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            logits = self.forward(x)  # (batch_size, num_labels)
            probabilities = torch.softmax(logits, dim=1)  # (batch_size, num_labels)
            predictions = torch.argmax(probabilities, dim=1)  # (batch_size)

        return predictions




class TransformerWithCLS(nn.Module):
    def __init__(self, input_size, max_seq_len, d_model=1024, d_seg=32, nhead=16, num_layers=3, num_labels=2):
        super().__init__()
        self.feature_embedding = nn.Linear(input_size, d_model-d_seg)
        self.segment_embedding = nn.Embedding(4, d_seg)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len + 1, d_model))  # +1 for [CLS]
        # self._generate_positional_encoding(max_seq_len + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_labels)
        )

    # def _generate_positional_encoding(self, max_seq_len, d_model):
    #     pe = torch.zeros(max_seq_len, d_model)
    #     position = torch.arange(0, max_seq_len, dtype=torch.float32)
    #     div_term = torch.exp(
    #         torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    #     )
    #     pe[:, 0::2] = torch.sin(position.unsqueeze(1) * div_term)
    #     pe[:, 1::2] = torch.cos(position.unsqueeze(1) * div_term)
    #     pe = pe.unsqueeze(0)  # 形状变为 (1, max_seq_len, d_model)
    #     self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        mask = (x.abs().sum(dim=-1) < 1e-10)    # [batch_size, seq_len], 填充位置为 1，非填充为 0

        segment_ids = x[:,:,-1].int()
        x = x[:,:,:-1]

        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, d_model]
        x = self.feature_embedding(x)  # [batch_size, seq_len, d_model-d_seg]
        segment_emb = self.segment_embedding(segment_ids)  # [batch_size, seq_len, d_seg]
        x = torch.cat([x, segment_emb], dim=2)  # [batch_size, seq_len, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, seq_len+1, d_model]
        
        x = x + self.positional_encoding[:, :x.size(1), :]
        mask_exp = torch.cat([torch.zeros(batch_size, 1, device=x.device), mask], dim=1)  # [batch_size, seq_len+1]
        x = self.transformer_encoder(x, src_key_padding_mask=mask_exp)  # [batch_size, seq_len+1, d_model]
        
        cls_output = x[:, 0, :]  # [batch_size, d_model]
        logits = self.classifier(cls_output)  # [batch_size, num_labels]
        return logits

    def predict(self, x, device=None):
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            logits = self.forward(x)  # (batch_size, num_labels)
            probabilities = torch.softmax(logits, dim=1)  # (batch_size, num_labels)
            predictions = torch.argmax(probabilities, dim=1)  # (batch_size)

        return predictions


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data      # 列表，每个元素是一个未填充的序列（如 [1, 2, 3]）
        self.labels = labels  # 列表，每个元素是一个未填充的标签序列（如 [0, 1, 2]）

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# def collate_fn(batch):
#     # PackedSequence类型变量，仅适用于RNN
#     # data, labels = batch[:,0], batch[:,1]
#     data, labels = [], []
#     for d,l in batch:
#         data.append(d)
#         labels.append(l)
#     data.sort(key=lambda x: len(x), reverse=True)
#     data = pack_sequence(data)
#     labels.sort(key=lambda x: len(x), reverse=True)
#     labels = pad_sequence(labels, batch_first=True, padding_value=-100)
#     # labels = pack_sequence(labels)
#     return data, labels


def collate_fn(batch):
    # 仅填充，适用于所有线性层
    # data, labels = batch[:,0], batch[:,1]
    data, labels = [], []
    for d,l in batch:
        data.append(d)
        labels.append(l)
    data = pad_sequence(data, batch_first=True, padding_value=0)
    return data, torch.tensor(labels)


def detection(model_name, jailbreak, classifier_type, prompt_format, without_template):
    model, tokenizer = load_model(model_name, model_paths)
    tool_dict = get_tool_dict('data_tool/tools.json', gpt_format=True)
    all_jailbreak_list = ["base", "ignore", "fake_completion"]
    train_jailbreak_list = all_jailbreak_list if jailbreak=="alljail" else [jailbreak]
    train_num, val_num, test_num = 200,200,1600
    num1 = train_num
    validation = False
    num2 = train_num if validation else train_num + val_num
    num3 = train_num + val_num if validation else train_num + val_num + test_num
    

    train_harmless_data, train_harmful_data = [], []
    test_harmless_data, test_harmful_data = [], []

    test_harmless_data += load_data("data_tool/data_useful/dh_harmless_with_injection.json")[num2:num3]
    for jail in all_jailbreak_list:
        test_harmful_data += load_data(f"data_tool/data_useful/dh_{jail}_with_injection.json")[num2:num3]
    
    for jail in train_jailbreak_list:
        train_harmless_data += load_data("data_tool/data_useful/dh_harmless_with_injection.json")[:num1]
        train_harmful_data += load_data(f"data_tool/data_useful/dh_{jail}_with_injection.json")[:num1]


    len_test_harmful_set = len(test_harmful_data)
    len_test_harmless_set = len(test_harmless_data)


    if_fit = True
    classifier_dir = f"./injecagent_classifier/{classifier_type}/{model_name}"
    if not os.path.exists(classifier_dir):
        os.makedirs(classifier_dir)

    classifier_name = f"IPIAttack_{classifier_type}_{'WithoutTemplate_' if without_template else ''}classifier_{model_name}_{jailbreak}.joblib"
    classifier_file = os.path.join(classifier_dir, classifier_name)
    if os.path.exists(classifier_file):
        if_fit = False

    device1 = torch.device("cuda:0")
    device2 = torch.device("cpu")

    train_harmful_attn_maps, train_harmful_sentence_token_ranges, train_harmful_seg_label_list, train_harmful_inj_label_list = \
        get_sentence_attn_tool(model, tokenizer, model_name, train_harmful_data[:], tool_dict, prompt_format=prompt_format, without_template=without_template)
    train_harmless_attn_maps, train_harmless_sentence_token_ranges, train_harmless_seg_label_list, train_harmless_inj_label_list = \
        get_sentence_attn_tool(model, tokenizer, model_name, train_harmless_data[:], tool_dict, prompt_format=prompt_format, without_template=without_template)

    del model
    del train_harmful_data
    del train_harmless_data

    max_seq_len = 150
    input_size = train_harmful_attn_maps.shape[0]*train_harmful_attn_maps.shape[1]*4
    num_epoches = 50
    batch_size = 128

    best_para_dir = f"./injecagent_results/results_detection/best_para/{model_name}"
    if not os.path.exists(best_para_dir):
        os.makedirs(best_para_dir)
    best_para_file = f"IPIAttack_{classifier_type}_{'WithoutTemplate_' if without_template else ''}classifier_{model_name}_{jailbreak}.json"
    best_para_file = os.path.join(best_para_dir, best_para_file)
    best_para_dict = load_data(best_para_file)
    d_model, d_seg, nhead, num_layers = best_para_dict["d_model"], best_para_dict["d_seg"], best_para_dict["nhead"], best_para_dict["num_layers"]

    print(f"Set Max Seq Len : {max_seq_len}")
    print(f"Input Size : {input_size}")
    print(f"Best Para : {best_para_dict}")

    if if_fit:
        train_harmful_feature = attn_map_segment_each_sentence(train_harmful_attn_maps, train_harmful_sentence_token_ranges, max_seq_len, device1)     # 32 * 32 * 4 * max_len * 2000 = 1.2*10^9
        train_harmless_feature = attn_map_segment_each_sentence(train_harmless_attn_maps, train_harmless_sentence_token_ranges, max_seq_len, device1)

        _train_harmful_seg_label_list, _train_harmless_seg_label_list = [], []
        for i in range(len(train_harmful_seg_label_list)):
            _train_harmful_seg_label_list.append(torch.tensor(train_harmful_seg_label_list[i]).unsqueeze(-1).to(device1))
        for i in range(len(train_harmless_seg_label_list)):
            _train_harmless_seg_label_list.append(torch.tensor(train_harmless_seg_label_list[i]).unsqueeze(-1).to(device1))
        
        for i in range(len(train_harmful_feature)):
            train_harmful_feature[i] = torch.cat((train_harmful_feature[i], _train_harmful_seg_label_list[i]),-1)
        for i in range(len(train_harmless_feature)):
            train_harmless_feature[i] = torch.cat((train_harmless_feature[i], _train_harmless_seg_label_list[i]),-1)

        labels = [1 for i in range(len(train_harmful_feature))] + [0 for i in range(len(train_harmless_feature))]

        print(len(train_harmful_feature))
        print(train_harmful_feature[0].shape)
        print(len(labels))

        dataset = MyDataset(train_harmful_feature + train_harmless_feature, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

        if classifier_type=="BiLSTM":
            classifier = BiLSTM(input_size=input_size, max_seq_len=max_seq_len).to(device1)
        elif classifier_type=="TransformerWithGAP":
            classifier = TransformerWithGAP(input_size=input_size, max_seq_len=max_seq_len, d_model=d_model, d_seg=d_seg, nhead=nhead, num_layers=num_layers).to(device1)
        elif classifier_type=="TransformerWithCLS":
            classifier = TransformerWithCLS(input_size=input_size, max_seq_len=max_seq_len, d_model=d_model, d_seg=d_seg, nhead=nhead, num_layers=num_layers).to(device1)
        else:
            print(f"No Model {classifier_type} Support!")
            exit()
        
        
        criterion = nn.CrossEntropyLoss()   # ignore_index=-100
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0001)    # lr=0.001
        train_loss = []
        for epoch in range(num_epoches):
            classifier.train()
            for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = classifier(batch_x)
                loss = criterion(outputs.view(-1, 2), batch_y.to(device1).view(-1).long())
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                if (batch_idx + 1) % 1 == 0:
                    print(f"Epoch [{epoch+1}/{epoch}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        torch.save(classifier.state_dict(), classifier_file)

        del train_harmful_feature
        del train_harmless_feature

        plt.figure()
        ax = plt.axes()
        plt.xlabel(f'steps ({num_epoches} epoches)')
        plt.ylabel('loss')
        plt.plot(range(len(train_loss)), train_loss, linewidth=1, linestyle="solid", label="train loss")
        plt.legend()
        plt.title('Loss curve')
        plt.tight_layout()
        plt.savefig(f"./injecagent_classifier/{classifier_type}/{model_name}/training_loss_{classifier_type}_{'WithoutTemplate_' if without_template else ''}classifier_{model_name}_{jailbreak}.png")

    else:
        if classifier_type=="BiLSTM":
            classifier = BiLSTM(input_size=input_size, max_seq_len=max_seq_len).to(device1)
        elif classifier_type=="TransformerWithGAP":
            classifier = TransformerWithGAP(input_size=input_size, max_seq_len=max_seq_len, d_model=d_model, d_seg=d_seg, nhead=nhead, num_layers=num_layers).to(device1)
        elif classifier_type=="TransformerWithCLS":
            classifier = TransformerWithCLS(input_size=input_size, max_seq_len=max_seq_len, d_model=d_model, d_seg=d_seg, nhead=nhead, num_layers=num_layers).to(device1)
        else:
            print(f"No Model {classifier_type} Support!")
            exit()

        classifier.load_state_dict(torch.load(classifier_file))
        classifier.eval()



    # Test Set
    result_dict = {}
    len_test_subset = len_test_harmful_set // len(all_jailbreak_list)

    model, tokenizer = load_model(model_name, model_paths)
    test_attn_maps, test_sentence_token_ranges, test_seg_label_list, test_inj_label_list = \
        get_sentence_attn_tool(model, tokenizer, model_name, test_harmless_data[:], tool_dict, prompt_format=prompt_format, without_template=without_template)
    del model

    test_feature = attn_map_segment_each_sentence(test_attn_maps, test_sentence_token_ranges, max_seq_len, device1)
    _test_seg_label_list = []
    for i in range(len(test_seg_label_list)):
        _test_seg_label_list.append(torch.tensor(test_seg_label_list[i]).unsqueeze(-1).to(device1))

    for i in range(len(test_feature)):
        test_feature[i] = torch.cat((test_feature[i], _test_seg_label_list[i]),-1)

    true_labels = torch.tensor([0 for i in range(len_test_subset)])
    ds = MyDataset(test_feature, true_labels)
    test_dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    acc = 0
    with torch.no_grad():
        for feature, true_label in test_dataloader:
            feature = feature.to(device1)
            true_label = true_label.to(device1)
            pred_label = classifier.predict(feature)
            for pred, true in zip(pred_label, true_label):
                if pred==true:
                    acc += 1
    acc /= len_test_subset
    print(f"harmless Acc: {acc}")
    result_dict["harmless Acc"] = acc


    for i, jail in enumerate(all_jailbreak_list):
        model, tokenizer = load_model(model_name, model_paths)
        test_attn_maps, test_sentence_token_ranges, test_seg_label_list, test_inj_label_list = \
            get_sentence_attn_tool(model, tokenizer, model_name, test_harmful_data[i*len_test_subset:(i+1)*len_test_subset], tool_dict, prompt_format=prompt_format, without_template=without_template)
        del model
        
        test_feature = attn_map_segment_each_sentence(test_attn_maps, test_sentence_token_ranges, max_seq_len, device1)
        _test_seg_label_list = []
        for i in range(len(test_seg_label_list)):
            _test_seg_label_list.append(torch.tensor(test_seg_label_list[i]).unsqueeze(-1).to(device1))

        for i in range(len(test_feature)):
            test_feature[i] = torch.cat((test_feature[i], _test_seg_label_list[i]),-1)

        true_labels = torch.tensor([1 for i in range(len_test_subset)])
        ds = MyDataset(test_feature, true_labels)
        test_dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        acc = 0
        with torch.no_grad():
            for feature, true_label in test_dataloader:
                feature = feature.to(device1)
                true_label = true_label.to(device1)
                pred_label = classifier.predict(feature)
                for pred, true in zip(pred_label, true_label):
                    if pred==true:
                        acc += 1
        acc /= len_test_subset
        print(f"{jail} Acc: {acc}")
        result_dict[f"{jail} Acc"] = acc

    
    result_dir = f"./injecagent_results/results_detection/{classifier_type}/{model_name}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file = f"{classifier_name[:-7]}.json"
    result_file = os.path.join(result_dir, result_file)
    with open(result_file, 'w') as f:
        f.write(json.dumps(result_dict, indent=True))
    print(json.dumps(result_dict, indent=True))



if __name__ == '__main__':
    # Get parameters
    parser = argparse.ArgumentParser(description='Attention Detection')
    parser.add_argument('--model_name', type=str, help='Target model')
    parser.add_argument('--jailbreak', type=str, choices=["alljail", "base", "ignore", "fake_completion"], default="alljail", help='Train Jailbreak type')
    parser.add_argument('--classifier_type', type=str, choices=["BiLSTM", "TransformerWithGAP", "TransformerWithCLS"], default="TransformerWithCLS",help='Train base dataset')
    parser.add_argument('--prompt_format', type=str, choices=["json", "react"], default="json", help='Prompt type')

    parser.add_argument('--without_template', default=False, action='store_true', help="if without template")


    args = parser.parse_args()
    model_name = args.model_name
    jailbreak = args.jailbreak
    classifier_type = args.classifier_type
    prompt_format = args.prompt_format
    without_template = args.without_template

    detection(model_name, jailbreak, classifier_type, prompt_format, without_template)