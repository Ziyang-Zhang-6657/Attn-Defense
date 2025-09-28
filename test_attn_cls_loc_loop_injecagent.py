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

from prompts.agent_prompts import PROMPT_DICT
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

# from utils_for_cls_and_loc import *
# from utils_for_resp import *
from utils_for_cls_and_loc_injecagent import load_data, load_model, get_tool_dict, get_sentence_attn_tool, attn_map_segment_each_sentence, update_item, create_messages_tool, find_sentence_range
from utils_for_resp_injecagent import predict_one_case, evaluate_output_fineutned_llama, evaluate_output_react, get_score


class BiLSTM_classifier(nn.Module):
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
        mask = (x.abs().sum(dim=-1) >= 1e-10)  # [batch_size, seq_len], 填充位置为 0，非填充为 1
        lengths = torch.tensor([sum(i) for i in mask])

        segment_ids = x[:, :, -1].int()
        x = x[:, :, :-1]

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


class TransformerWithGAP_classifier(nn.Module):
    def __init__(self, input_size, max_seq_len, d_model=256, d_seg=64, nhead=8, num_layers=6, num_labels=2):
        super(TransformerWithGAP, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model - d_seg)  # 输入投影层
        self.segment_embedding = nn.Embedding(4, d_seg)  # 段位置嵌入
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
        mask = (x.abs().sum(dim=-1) < 1e-10)  # [batch_size, seq_len], 填充位置为 1，非填充为 0

        segment_ids = x[:, :, -1].int()
        x = x[:, :, :-1]

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


class TransformerWithCLS_classifier(nn.Module):
    def __init__(self, input_size, max_seq_len, d_model=1024, d_seg=32, nhead=16, num_layers=3, num_labels=2):
        super().__init__()
        self.feature_embedding = nn.Linear(input_size, d_model - d_seg)
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
        mask = (x.abs().sum(dim=-1) < 1e-10)  # [batch_size, seq_len], 填充位置为 1，非填充为 0

        segment_ids = x[:, :, -1].int()
        x = x[:, :, :-1]

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


class BiLSTM_locator(nn.Module):
    def __init__(self, input_size, hidden_size=1024, num_layers=2, num_classes=2):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        packed_out, _ = self.bilstm(x)  # (batch_size, seq_len, 2*hidden_size)
        out, out_len = pad_packed_sequence(packed_out, padding_value=-100, batch_first=True)  # , batch_first=True
        logits = self.fc(out)  # (batch_size, seq_len, num_classes)
        return logits

    def predict(self, x, device=None):
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        x = x.to(device)
        with torch.no_grad():
            logits = self.forward(x)  # (batch_size, seq_len, num_classes)
            probabilities = torch.softmax(logits, dim=2)  # (batch_size, seq_len, num_classes)
            predictions = torch.argmax(probabilities, dim=2)  # (batch_size, seq_len)

        return predictions


class Transformer_locator(nn.Module):
    def __init__(self, input_size, max_seq_len, d_model=1024, d_seg=16, nhead=8, num_layers=3, num_labels=2):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model - d_seg)
        self.segment_embedding = nn.Embedding(4, d_seg)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        # self._generate_positional_encoding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_labels)

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
        out, out_len = pad_packed_sequence(x, padding_value=0, batch_first=True)
        mask = torch.where(out == 0, torch.ones_like(out), torch.zeros_like(out))
        mask = mask.all(dim=-1)

        segment_ids = out[:, :, -1].int()
        x = out[:, :, :-1]

        x = self.embedding(x)  # [batch_size, seq_len, d_model-d_seg]
        seg_emb = self.segment_embedding(segment_ids)  # [batch_size, seq_len, d_seg]
        x = torch.cat([x, seg_emb], dim=2)

        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # [batch_size, seq_len, d_model]
        logits = self.classifier(x)  # [batch_size, seq_len, num_labels]
        return logits

    def predict(self, x, device=None):
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            logits = self.forward(x)  # (batch_size, seq_len, num_classes)
            probabilities = torch.softmax(logits, dim=2)  # (batch_size, seq_len, num_classes)
            predictions = torch.argmax(probabilities, dim=2)  # (batch_size, seq_len)

        return predictions


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def collate_fn_cls(batch):
    data, labels = [], []
    for d,l in batch:
        data.append(d)
        labels.append(l)
    data = pad_sequence(data, batch_first=True, padding_value=0)
    return data, torch.tensor(labels)


def collate_fn_loc(batch):
    data, labels = [], []
    for d,l in batch:
        data.append(d)
        labels.append(l)
    data.sort(key=lambda x: len(x), reverse=True)
    data = pack_sequence(data)
    labels.sort(key=lambda x: len(x), reverse=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return data, labels



def run(model_name, jailbreak, classifier_type, locator_type, prompt_format, without_template):
    model, tokenizer = load_model(model_name, model_paths)
    tool_dict = get_tool_dict('data_tool/tools.json', gpt_format=True)
    all_jailbreak_list = ["base", "ignore", "fake_completion"]
    train_jailbreak_list = all_jailbreak_list if jailbreak=="alljail" else [jailbreak]
    train_num, val_num, test_num = 200,200,1600
    num1 = train_num
    num2 = train_num + val_num
    num3 = train_num + val_num + test_num
    
    test_harmless_data, test_harmful_data = [], []

    test_harmless_data += load_data("data_tool/data_useful/dh_harmless_with_injection.json")[num2:num3]
    for jail in all_jailbreak_list:
        test_harmful_data += load_data(f"data_tool/data_useful/dh_{jail}_with_injection.json")[num2:num3]

    len_test_harmful_set = len(test_harmful_data)
    len_test_harmless_set = len(test_harmless_data)
    len_test_subset = len_test_harmful_set // len(all_jailbreak_list)


    classifier_dir = f"./injecagent_classifier/{classifier_type}/{model_name}"
    if not os.path.exists(classifier_dir):
        os.makedirs(classifier_dir)

    classifier_name = f"IPIAttack_{classifier_type}_{'WithoutTemplate_' if without_template else ''}classifier_{model_name}_alljail.joblib"
    print(f"Classifier Name : {classifier_name}")
    classifier_file = os.path.join(classifier_dir, classifier_name)
    

    locator_dir = f"./injecagent_locator/{locator_type}/{model_name}"
    if not os.path.exists(locator_dir):
        os.makedirs(locator_dir)

    locator_name = f"IPIAttack_{locator_type}_{'WithoutTemplate_' if without_template else ''}locator_{model_name}_alljail.joblib"
    print(f"Locator Name : {locator_name}")
    locator_file = os.path.join(locator_dir, locator_name)


    device1 = torch.device("cuda:0")
    device2 = torch.device("cpu")

    max_seq_len = 150
    input_size = model.config.num_attention_heads * model.config.num_hidden_layers * 4
    num_epoches = 50
    batch_size = 1

    best_para_dir = f"./injecagent_results/results_detection/best_para/{model_name}"
    if not os.path.exists(best_para_dir):
        os.makedirs(best_para_dir)
    best_para_file = f"IPIAttack_{classifier_type}_{'WithoutTemplate_' if without_template else ''}classifier_{model_name}_{jailbreak}.json"
    best_para_file = os.path.join(best_para_dir, best_para_file)
    best_para_dict = load_data(best_para_file)
    cls_d_model, cls_d_seg, cls_nhead, cls_num_layers = best_para_dict["d_model"], best_para_dict["d_seg"], best_para_dict["nhead"], best_para_dict["num_layers"]
    print(f"CLS Best Para : {best_para_dict}")

    best_para_dir = f"./injecagent_results/results_location/best_para/{model_name}"
    if not os.path.exists(best_para_dir):
        os.makedirs(best_para_dir)
    best_para_file = f"IPIAttack_{locator_type}_{'WithoutTemplate_' if without_template else ''}locator_{model_name}_{jailbreak}.json"
    best_para_file = os.path.join(best_para_dir, best_para_file)
    best_para_dict = load_data(best_para_file)
    loc_d_model, loc_d_seg, loc_nhead, loc_num_layers = best_para_dict["d_model"], best_para_dict["d_seg"], best_para_dict["nhead"], best_para_dict["num_layers"]
    print(f"LOC Best Para : {best_para_dict}")

    print(f"Max Seq Len : {max_seq_len}")
    print(f"Input Size : {input_size}")

    
    if classifier_type=="BiLSTM":
        classifier = BiLSTM_classifier(input_size=input_size, max_seq_len=max_seq_len).to(device1)
    elif classifier_type=="TransformerWithGAP":
        classifier = TransformerWithGAP_classifier(input_size=input_size, max_seq_len=max_seq_len, d_model=cls_d_model, d_seg=cls_d_seg, nhead=cls_nhead, num_layers=cls_num_layers).to(device1)
    elif classifier_type=="TransformerWithCLS":
        classifier = TransformerWithCLS_classifier(input_size=input_size, max_seq_len=max_seq_len, d_model=cls_d_model, d_seg=cls_d_seg, nhead=cls_nhead, num_layers=cls_num_layers).to(device1)
    else:
        print(f"No Model {classifier_type} Support!")
        exit()

    classifier.load_state_dict(torch.load(classifier_file))
    classifier.eval()
    
    if locator_type == "BiLSTM":
        locator = BiLSTM_locator(input_size=input_size).to(device1)
    elif locator_type == "Transformer":
        locator = Transformer_locator(input_size=input_size, max_seq_len=max_seq_len, d_model=loc_d_model, d_seg=loc_d_seg, nhead=loc_nhead, num_layers=loc_num_layers).to(device1)

    locator.load_state_dict(torch.load(locator_file))
    locator.eval()

    output_dir = f"./injecagent_results/results_cls_with_loc_defense/{classifier_type}_CLS__{locator_type}_LOC/{model_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result_dir = f"./injecagent_results/results_data_cls_with_loc_defense/{classifier_type}_CLS__{locator_type}_LOC/{model_name}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)



    for i,jail in enumerate(["harmless", "base", "ignore", "fake_completion"]):
        data = test_harmless_data if jail=="harmless" else test_harmful_data[(i-1)*len_test_subset:i*len_test_subset]
        print(f"\nJail: {jail}\n")

        # 准备数据
        model, tokenizer = load_model(model_name, model_paths)
        print("+" * 100 + f"Get model {jail} attention feature ..." + "+" * 100)
        cls_labels, loc_labels = [], []
        test_attn_maps, test_sentence_token_ranges, test_seg_label_list, test_inj_label_list = get_sentence_attn_tool(model, tokenizer, model_name, data[:], tool_dict, prompt_format=prompt_format, without_template=without_template)
        del model
        test_attn_maps = test_attn_maps.to(device1)

        test_feature = attn_map_segment_each_sentence(test_attn_maps, test_sentence_token_ranges, max_seq_len, device1)
        del test_attn_maps

        _test_seg_label_list = []
        for i in range(len(test_seg_label_list)):
            _test_seg_label_list.append(torch.tensor(test_seg_label_list[i]).unsqueeze(-1).to(device1))

        for i in range(len(test_feature)):
            test_feature[i] = torch.cat((test_feature[i], _test_seg_label_list[i]),-1)

        ## 获取cls & loc label
        print("+" * 100 + f"Get cls&loc {jail} label ..." + "+" * 100)
        
        acc,tn,tp,fn,fp = 0,0,0,0,0

        cls_true_labels = torch.tensor([0 for i in range(len_test_subset)])
        cls_ds = MyDataset(test_feature, cls_true_labels)
        cls_dataloader = DataLoader(cls_ds, batch_size=batch_size, collate_fn=collate_fn_cls, shuffle=False)
        with torch.no_grad():
            for feature, true_label in cls_dataloader:
                feature = feature.to(device1)
                pred_cls_label = classifier.predict(feature)
                cls_labels += pred_cls_label.cpu().numpy().tolist()

                for pred, true in zip(pred_cls_label, true_label):
                    if pred==true:
                        acc += 1
        print(f"{jail} Acc: {acc/len(data)}")
        del cls_ds, cls_dataloader

        loc_true_labels = [torch.tensor(seq) for seq in test_inj_label_list]
        loc_ds = MyDataset(test_feature, loc_true_labels)
        loc_dataloader = DataLoader(loc_ds, batch_size=batch_size, collate_fn=collate_fn_loc, shuffle=False)
        with torch.no_grad():
            for feature, true_label in loc_dataloader:
                feature = feature.to(device1)
                pred_loc_label = locator.predict(feature)
                loc_labels += pred_loc_label.cpu().numpy().tolist()

                for pred, true in zip(pred_loc_label, true_label):
                    for pr, tr in zip(pred, true):
                        if tr == -100:
                            break
                        elif tr==0 and pr==0:
                            tn+=1
                        elif tr==1 and pr==1:
                            tp+=1
                        elif tr==1 and pr==0:
                            fn+=1
                        elif tr==0 and pr==1:
                            fp+=1 

        precision = tp/(tp+fp) if (tp+fp) else 0
        recall = tp/(tp+fn) if (tp+fn) else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0
        print(f"{jail} Precision: {precision}")
        print(f"{jail} Recall: {recall}")
        print(f"{jail} F1 Score: {f1}")
        del loc_ds, loc_dataloader
        del test_feature

        # 删除IPI定位句子
        print("+" * 100 + f"Removing {jail} data IPI instruction ..." + "+" * 100)
        for i in range(len(data)):
            if cls_labels[i]:
                data[i] = update_item(model_name, data[i], tokenizer, tool_dict, prompt_format=prompt_format, loc_label=loc_labels[i], without_template=without_template)

        # 获取cls & loc防御下model输出
        file_name = f"{jail}.json"
        output_file = os.path.join(output_dir, file_name)

        model, tokenizer = load_model(model_name, model_paths)
        print("+"*100+f"Get model {jail} output ..."+"+"*100)
        with open(output_file, 'w') as f:
            for item in data:
                # try:
                output = predict_one_case(model_name, item, model, tokenizer, tool_dict, prompt_format=prompt_format)
                item["output"] = output
                

                if prompt_format=="json":
                    parsed_output = evaluate_output_fineutned_llama(item['output'], item['Attacker Tools'][0], item['User Tool'])
                elif prompt_format=="react":
                    parsed_output = evaluate_output_react(item['output'], item['Attacker Tools'][0], item['Attacker Tools'], item['User Tool'], item["Attacker Instruction"])
                item['eval'] = parsed_output['eval']
                if item['eval'] == 'invalid':
                    item['invalid_reason'] = parsed_output['invalid_reason']

                f.write(json.dumps(item) + '\n')
    
        scores = get_score(output_file)
        
        file_name = f"{jail}.json"
        result_file = os.path.join(result_dir, file_name)

        with open(result_file, 'w') as f:
            f.write(json.dumps(scores, indent=True))
        print(json.dumps(scores, indent=True))



if __name__ == '__main__':
    # Get parameters
    parser = argparse.ArgumentParser(description='Attention Detection')
    parser.add_argument('--model_name', type=str, help='Target model')
    parser.add_argument('--jailbreak', type=str, choices=["alljail", "base", "ignore", "fake_completion"], default="alljail", help='Train Jailbreak type')
    parser.add_argument('--classifier_type', type=str, choices=["BiLSTM", "TransformerWithGAP", "TransformerWithCLS"], default="TransformerWithCLS",help='Train base dataset')
    parser.add_argument('--locator_type', type=str, choices=["BiLSTM", "BiLSTM_Attention", "Transformer"], help='Locator model type')
    parser.add_argument('--prompt_format', type=str, choices=["json", "react"], default="json", help='Prompt type')

    parser.add_argument('--without_template', default=False, action='store_true', help="if without template")

    args = parser.parse_args()
    model_name = args.model_name
    jailbreak = args.jailbreak
    classifier_type = args.classifier_type
    locator_type = args.locator_type
    prompt_format = args.prompt_format
    without_template = args.without_template

    run(model_name, jailbreak, classifier_type, locator_type, prompt_format, without_template)
