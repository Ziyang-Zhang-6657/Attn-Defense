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



class Transformer(nn.Module):
    def __init__(self, input_size, max_seq_len, d_model=1024, d_seg=16, nhead=8, num_layers=3, num_labels=2):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model-d_seg)
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
        mask = torch.where(out==0, torch.ones_like(out), torch.zeros_like(out))
        mask = mask.all(dim=-1)

        segment_ids = out[:,:,-1].int()
        x = out[:,:,:-1]

        x = self.embedding(x)  # [batch_size, seq_len, d_model-d_seg]
        seg_emb = self.segment_embedding(segment_ids)   # [batch_size, seq_len, d_seg]
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
        self.data = data      # 列表，每个元素是一个未填充的序列（如 [1, 2, 3]）
        self.labels = labels  # 列表，每个元素是一个未填充的标签序列（如 [0, 1, 2]）

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def collate_fn(batch):
    data, labels = [], []
    for d,l in batch:
        data.append(d)
        labels.append(l)
    data.sort(key=lambda x: len(x), reverse=True)
    data = pack_sequence(data)
    labels.sort(key=lambda x: len(x), reverse=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return data, labels



def location(model_name, jailbreak, locator_type, prompt_format, without_template):
    model, tokenizer = load_model(model_name, model_paths)
    tool_dict = get_tool_dict('data_tool/tools.json', gpt_format=True)
    all_jailbreak_list = ["base", "ignore", "fake_completion"]
    train_jailbreak_list = all_jailbreak_list if jailbreak=="alljail" else [jailbreak]
    train_num, val_num, test_num = 200,200,1600
    num1 = train_num
    num2 = train_num + val_num
    num3 = train_num + val_num + test_num
    
    train_harmless_data, train_harmful_data = [], []
    test_harmless_data, test_harmful_data = [], []

    test_harmless_data += load_data("data_tool/data_useful/dh_harmless_with_injection.json")[num2:num3]
    for jail in all_jailbreak_list:
        test_harmful_data += load_data(f"data_tool/data_useful/dh_{jail}_with_injection.json")[num2:num3]
    
    for jail in train_jailbreak_list:
        train_harmless_data += load_data("data_tool/data_useful/dh_harmless_with_injection.json")[:num1]
        train_harmful_data += load_data(f"data_tool/data_useful/dh_{jail}_with_injection.json")[:num1]

    
    len_test_set = len(test_harmful_data)


    if_train = True

    locator_dir = f"./injecagent_locator/{locator_type}/{model_name}"
    if not os.path.exists(locator_dir):
        os.makedirs(locator_dir)

    locator_name = f"IPIAttack_{locator_type}_{'WithoutTemplate_' if without_template else ''}locator_{model_name}_{jailbreak}.joblib"
    print(f"Locator Name : {locator_name}")
    locator_file = os.path.join(locator_dir, locator_name)
    if os.path.exists(locator_file):
        if_train = False
    
    device1 = torch.device("cuda:0")
    device2 = torch.device("cpu")


    train_attn_maps, train_sentence_token_ranges, train_seg_label_list, train_inj_label_list = \
        get_sentence_attn_tool(model, tokenizer, model_name, train_harmful_data[:], tool_dict, prompt_format=prompt_format, without_template=without_template)


    del model
    del train_harmful_data


    max_seq_len = 150
    input_size = train_attn_maps.shape[0]*train_attn_maps.shape[1]*4
    num_epoches = 75
    batch_size = 128

    best_para_dir = f"./injecagent_results/results_location/best_para/{model_name}"
    if not os.path.exists(best_para_dir):
        os.makedirs(best_para_dir)
    best_para_file = f"IPIAttack_{locator_type}_{'WithoutTemplate_' if without_template else ''}locator_{model_name}_{jailbreak}.json"
    best_para_file = os.path.join(best_para_dir, best_para_file)
    best_para_dict = load_data(best_para_file)
    d_model, d_seg, nhead, num_layers = best_para_dict["d_model"], best_para_dict["d_seg"], best_para_dict["nhead"], best_para_dict["num_layers"]

    print(f"Max Seq Len : {max_seq_len}")
    print(f"Input Size : {input_size}")
    print(f"Best Para : {best_para_dict}")

    if if_train:
        train_feature = attn_map_segment_each_sentence(train_attn_maps, train_sentence_token_ranges, max_seq_len, device1)     # 32 * 32 * 4 * max_len * 2000 = 1.2*10^9

        _train_seg_label_list = []
        for i in range(len(train_seg_label_list)):
            _train_seg_label_list.append(torch.tensor(train_seg_label_list[i]).unsqueeze(-1).to(device1))

        for i in range(len(train_feature)):
            train_feature[i] = torch.cat((train_feature[i], _train_seg_label_list[i]),-1)

        labels = [torch.tensor(seq) for seq in train_inj_label_list]

        ds = MyDataset(train_feature, labels)
        dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)


        if locator_type == "BiLSTM":
            locator = BiLSTM(input_size=input_size).to(device1)
        elif locator_type == "Transformer":
            locator = Transformer(input_size=input_size, max_seq_len=max_seq_len, d_model=d_model, d_seg=d_seg, nhead=nhead, num_layers=num_layers).to(device1)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)   # ignore_index=-100
        optimizer = torch.optim.Adam(locator.parameters(), lr=0.0001)    # lr=0.001

        train_loss = []

        for epoch in range(num_epoches):
            locator.train()
            for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = locator(batch_x)
                loss = criterion(outputs.view(-1, 2), batch_y.to(device1).view(-1).long())
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                if (batch_idx + 1) % 1 == 0:
                    print(f"Epoch [{epoch+1}/{epoch}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        torch.save(locator.state_dict(), locator_file)

        del train_feature

        plt.figure()
        ax = plt.axes()
        plt.xlabel(f'steps ({num_epoches} epoches)')
        plt.ylabel('loss')
        plt.plot(range(len(train_loss)), train_loss, linewidth=1, linestyle="solid", label="train loss")
        plt.legend()
        plt.title('Loss curve')
        plt.tight_layout()
        plt.savefig(locator_dir + f"/training_loss_{locator_type}_{'WithoutTemplate_' if without_template else ''}locator_{model_name}_{jailbreak}.png")


    else:
        if locator_type == "BiLSTM":
            locator = BiLSTM(input_size=input_size).to(device1)
        elif locator_type == "Transformer":
            locator = Transformer(input_size=input_size, max_seq_len=max_seq_len, d_model=d_model, d_seg=d_seg, nhead=nhead, num_layers=num_layers).to(device1)

        locator.load_state_dict(torch.load(locator_file))
        locator.eval()

    result_dict = {}
    len_test_subset = len_test_set // len(all_jailbreak_list)

    for i, jail in enumerate(all_jailbreak_list):
        model, tokenizer = load_model(model_name, model_paths)
        test_attn_maps, test_sentence_token_ranges, test_seg_label_list, test_inj_label_list = \
            get_sentence_attn_tool(model, tokenizer, model_name, test_harmful_data[i*len_test_subset:(i+1)*len_test_subset], tool_dict, prompt_format=prompt_format, without_template=without_template)

    
        del model
        
        test_feature = attn_map_segment_each_sentence(test_attn_maps, test_sentence_token_ranges, max_seq_len, device1)     # 32 * 32 * 4 * max_len * 2000 = 1.2*10^9

        _test_seg_label_list = []
        for i in range(len(test_seg_label_list)):
            _test_seg_label_list.append(torch.tensor(test_seg_label_list[i]).unsqueeze(-1).to(device1))

        for i in range(len(test_feature)):
            test_feature[i] = torch.cat((test_feature[i], _test_seg_label_list[i]),-1)

        true_labels = [torch.tensor(seq) for seq in test_inj_label_list]
        ds = MyDataset(test_feature, true_labels)
        test_dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

        tp,tn,fp,fn = 0,0,0,0
        with torch.no_grad():
            for feature, true_label in test_dataloader:
                feature = feature.to(device1)
                true_label = true_label.to(device1)
                pred_label = locator.predict(feature)
                for pred, true in zip(pred_label, true_label):
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
        result_dict[f"{jail} Precision"] = precision
        result_dict[f"{jail} Recall"] = recall
        result_dict[f"{jail} F1 Score"] = f1
        
        del test_feature
    
    result_dir = f"./injecagent_results/results_location/{locator_type}/{model_name}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file = f"{locator_name[:-7]}.json"
    result_file = os.path.join(result_dir, result_file)
    with open(result_file, 'w') as f:
        f.write(json.dumps(result_dict, indent=True))
    print(json.dumps(result_dict, indent=True))


if __name__ == '__main__':
    # Get parameters
    parser = argparse.ArgumentParser(description='Attention Detection')
    parser.add_argument('--model_name', type=str, help='Target model')
    parser.add_argument('--jailbreak', type=str, choices=["alljail", "base", "ignore", "fake_completion"], default="alljail", help='Train Jailbreak type')
    parser.add_argument('--locator_type', type=str, choices=["BiLSTM", "BiLSTM_Attention", "Transformer"], help='Locator model type')
    parser.add_argument('--prompt_format', type=str, choices=["json", "react"], default="json", help='Prompt type')

    parser.add_argument('--without_template', default=False, action='store_true', help="if without template")


    args = parser.parse_args()
    model_name = args.model_name
    jailbreak = args.jailbreak
    locator_type = args.locator_type
    prompt_format = args.prompt_format
    without_template = args.without_template

    location(model_name, jailbreak, locator_type, prompt_format, without_template)
