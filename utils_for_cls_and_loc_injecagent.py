import os
import json
import json_repair
from json import JSONDecoder
import gc
import copy

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
from sklearn.metrics import roc_curve
import torch.nn.functional as F
import jinja2
from collections import Counter
from evaluate import load
bertscore = load("metrics/bertscore")
import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
# nltk.download('punkt_tab')



env = jinja2.Environment(loader=jinja2.FileSystemLoader('prompts/jinja'))
llama_agent_template = env.get_template('llama_tool_use_new_prompt.jinja')
llama_IP_agent_template = env.get_template('llama_tool_use_instruction_prevention_new_prompt.jinja')
call_tool_template = env.get_template('call_tool.jinja')
llama_tool_response_template = env.get_template('tool_response.jinja')
llama_system_prompt_template = env.get_template('llama_system_prompt.jinja')

qwen_default_agent_template = env.get_template('qwen_tool_use_default_prompt.jinja')
qwen_IP_agent_template = env.get_template('qwen_tool_use_instruction_prevention_new_prompt.jinja')
qwen_new_agent_template = env.get_template('qwen_tool_use_new_prompt.jinja')
qwen_tool_response_template = env.get_template('qwen_tool_response.jinja')

react_sys = """Answer the following questions as best you can. You have access to the following tools:

{tool_desc}

Use the following format:

Question: the input question you must answer

Thought: you should always think about what to do

Action: the action to take, should be one of {tool_name}

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question

Begin!"""




def load_model(model_name, model_paths):
    model_path = model_paths.get(model_name)

    if model_path is None:
        raise ValueError(
            f"Model name {model_name} not recognized. Please choose from {list(model_paths.keys())}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def update_item(model_name, item, tokenizer, tool_dict, prompt_format="json", loc_label=None, without_template=False):
    user_instruction = item['User Instruction']
    tool_name = [item['User Tool']] + item['Attacker Tools']
    tool_desc = [tool_dict[tool] for tool in tool_name]
    
    orig_model_input = create_messages_tool(model_name, item, tokenizer, tool_desc, tool_name, prompt_format=prompt_format)
    data_start = orig_model_input.find(item['Tool Response']) if prompt_format=="json" else orig_model_input.find(f"Observation: {item['Tool Response']}")
    data_end = data_start + len(item['Tool Response'] if prompt_format=="json" else f"Observation: {item['Tool Response']}")
    all_sentences = []
    # json格式tool response之前部分
    for sent in nltk.sent_tokenize(orig_model_input[:data_start]):
        all_sentences.append(sent)
    # json格式tool response
    data_segs = divide_json_tool_resp(item['Tool Response'])
    for seg in data_segs:
        if len(seg) > 50:
            for sent in nltk.sent_tokenize(seg):
                all_sentences.append(sent)
        else:
            all_sentences.append(seg)
    # json格式tool response之后部分
    for sent in nltk.sent_tokenize(orig_model_input[data_end:]):
        all_sentences.append(sent)

    if prompt_format=="json":
        sys = llama_system_prompt_template.render(available_tool_descriptions=tool_desc)
        inst = item['User Instruction']
        assis = call_tool_template.render(tool_name=item['User Tool'], tool_parameters=item['Tool Parameters'])
        data = item['Tool Response']
    elif prompt_format=="react":
        sys = react_sys.format(tool_desc=tool_desc, tool_name=tool_name)
        inst = f"Question: {item['User Instruction']}"
        assis = f"{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}"
        data = f"Observation: {item['Tool Response']}"

    sys_sentences = nltk.sent_tokenize(sys)
    inst_sentences = [inst]
    assis_sentences = nltk.sent_tokenize(assis)
    data_sentences = ["Observation: "] if prompt_format=="react" else []
    for seg in data_segs:
        if len(seg) > 50:
            for sent in nltk.sent_tokenize(seg):
                data_sentences.append(sent)
        else:
            data_sentences.append(seg)

    if without_template:
        sentences = sys_sentences + inst_sentences + assis_sentences + data_sentences
    else:
        sentences = all_sentences

    if without_template:
        sentences = sys_sentences + inst_sentences + assis_sentences + data_sentences
    else:
        sentences = all_sentences

    sys_s = orig_model_input.rfind(sys)
    sys_e = sys_s+len(sys)
    inst_s = orig_model_input.find(inst,sys_e)
    inst_e = inst_s+len(inst)
    assis_s = orig_model_input.rfind(assis,inst_e)
    assis_e = assis_s+len(assis)
    data_s = orig_model_input.find(data,assis_e)
    data_e = data_s+len(data)

    loc_inj_sentences = []
    for i in range(len(sentences)):
        if loc_label[i]:
            start = orig_model_input.rfind(sentences[i])
            if start>=data_s:
                loc_inj_sentences.append(sentences[i])

    # 搜索结果中删去定位到的句子
    update_context = item["Tool Response"]
    context_sentences = nltk.sent_tokenize(update_context)
    for sent in loc_inj_sentences:
        for context_sent in context_sentences:
            if (context_sent.strip() in sent.strip()) or (sent.strip() in context_sent.strip()):
                update_context = update_context.replace(context_sent, "")
    item["Tool Response"] = update_context
    return item


def create_messages_tool(model_name, item, tokenizer, tool_desc, tool_name, prompt_format="json"):
    user_instruction = item['User Instruction']
    if prompt_format=="json":
        agent_template = llama_agent_template
        tool_response_template = llama_tool_response_template
        # elif ("Qwen" in model_name) or ("Tsunami" in model_name):
        #     agent_template = qwen_default_agent_template
        #     tool_response_template = qwen_tool_response_template
        model_input = agent_template.render(
            available_tool_descriptions=tool_desc,
            user_instruction=user_instruction,
            call_of_user_tool=call_tool_template.render(tool_name=item['User Tool'],
                                                        tool_parameters=item['Tool Parameters']),
            response_of_user_tool=item['Tool Response'],
        )
    elif prompt_format=="react":
        system_prompt, user_prompt = PROMPT_DICT["hwchase17_react"]
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"

        user_prompt_filled = user_prompt.format(available_tool_descriptions=tool_desc, input=item['User Instruction'],
                                                available_tool_names=tool_name,
                                                agent_scratchpad=agent_scratchpad)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_filled},
        ]
        model_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return model_input


def divide_json_tool_resp(json_str):
    segments = []
    
    data = json_repair.loads(json_str)
    if isinstance(data, str):
        data = json_repair.loads(data)
    
    # print(data)
    # print(type(data))
    # print(data.keys())
    # print(data[list(data.keys())[0]].keys())
    # print("\n\n\n")

    def get_item(input_data, start):
        if isinstance(input_data, dict):
            for k in list(input_data.keys()):
                end = json_str.find(k, start) + len(k)
                # print((start, end))
                # print(json_str[start:end])
                segments.append(json_str[start:end])
                start = end
                start = get_item(input_data[k], start)
        elif isinstance(input_data, list):
            for item in input_data:
                start = get_item(item, start)
        elif isinstance(input_data, str):
            end = json_str.find(input_data, start) + len(input_data)
            # print((start, end))
            # print(json_str[start:end])
            segments.append(json_str[start:end])
            start = end
        else:
            start = get_item(str(input_data), start)
        return start

    start = get_item(data, 0)
    # print(json_str[start:])
    segments.append(json_str[start:])

    return segments


def find_sentence_range(text, offset_mapping, model_name, item, tool_desc, tool_name, prompt_format="json", without_template=False):
    data_start = text.find(item['Tool Response']) if prompt_format=="json" else text.find(f"Observation: {item['Tool Response']}")
    data_end = data_start + len(item['Tool Response'] if prompt_format=="json" else f"Observation: {item['Tool Response']}")
    all_sentences = []
    # json格式tool response之前部分
    for sent in nltk.sent_tokenize(text[:data_start]):
        all_sentences.append(sent)
    # json格式tool response
    data_segs = divide_json_tool_resp(item['Tool Response'])
    for seg in data_segs:
        if len(seg) > 50:
            for sent in nltk.sent_tokenize(seg):
                all_sentences.append(sent)
        else:
            all_sentences.append(seg)
    # json格式tool response之后部分
    for sent in nltk.sent_tokenize(text[data_end:]):
        all_sentences.append(sent)

    if prompt_format=="json":
        sys = llama_system_prompt_template.render(available_tool_descriptions=tool_desc)
        inst = item['User Instruction']
        assis = call_tool_template.render(tool_name=item['User Tool'], tool_parameters=item['Tool Parameters'])
        data = item['Tool Response']
    elif prompt_format=="react":
        # sys, _ = PROMPT_DICT["hwchase17_react"]
        sys = react_sys.format(tool_desc=tool_desc, tool_name=tool_name)
        inst = f"Question: {item['User Instruction']}"
        assis = f"{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}"
        data = f"Observation: {item['Tool Response']}"

    sys_sentences = nltk.sent_tokenize(sys)
    inst_sentences = [inst]
    assis_sentences = nltk.sent_tokenize(assis)
    data_sentences = ["Observation: "] if prompt_format=="react" else []
    for seg in data_segs:
        if len(seg) > 50:
            for sent in nltk.sent_tokenize(seg):
                data_sentences.append(sent)
        else:
            data_sentences.append(seg)

    if without_template:
        sentences = sys_sentences + inst_sentences + assis_sentences + data_sentences
    else:
        sentences = all_sentences

    inj_q = item["Attacker Injection"]
    inj_sents = nltk.sent_tokenize(inj_q)

    sys_s = text.rfind(sys)
    sys_e = sys_s+len(sys)
    inst_s = text.find(inst,sys_e)
    inst_e = inst_s+len(inst)
    assis_s = text.rfind(assis,inst_e)
    assis_e = assis_s+len(assis)
    data_s = text.find(data,assis_e)
    data_e = data_s+len(data) 

    inj_label = [] 
    for s in sentences:
        # print("-"*100)
        # print(s)
        judge = 0
        for inj_s in inj_sents:
            if s.strip() and ((s.strip() in inj_s.strip()) or (inj_s.strip() in s.strip())):
                judge = 1
                break
        inj_label.append(judge)
    # print(inj_label)
    # exit()

    data_idx = text.find(data)

    seg_label = []
    start = data_idx
    if without_template:
        seg_label = [0 for s in sys_sentences] + [1 for s in inst_sentences] + [2 for s in assis_sentences] + [3 for s in data_sentences]
    else:
        for sentence in sentences:
            # 找到句子在原始文本中的起始和结束位置
            start_char = text.find(sentence, start)
            end_char = start_char + len(sentence) - 1  # 包含最后一个字符的位置
            if end_char<sys_e:
                seg_label.append(0)
            elif end_char<inst_e:
                seg_label.append(1)
            elif end_char<assis_e:
                seg_label.append(2)
            else:
                seg_label.append(3)

    sentence_token_range = []
    start = 0
    for sentence in sentences:
        # 找到句子在原始文本中的起始和结束位置
        start_char = text.find(sentence, start)
        end_char = start_char + len(sentence) - 1  # 包含最后一个字符的位置
        start = end_char + 1

        # 找到对应的 token 起始和结束索引
        start_token = None
        end_token = None
        max_token = len(offset_mapping[0])
        for i, (token_start, token_end) in enumerate(offset_mapping[0]):
            if token_start <= start_char and token_end >= start_char:
                start_token = i
            if token_start <= end_char and token_end >= end_char:
                end_token = min(i+1, max_token)
                break  # 找到后退出循环

        sentence_token_range.append((start_token, end_token))

    return sentence_token_range, seg_label, inj_label


def get_input_ids_tool(model, tokenizer, model_name, item, tool_dict, prompt_format="json", without_template=False, return_text=False):
    tool_name = [item['User Tool']] + item['Attacker Tools']
    tool_desc = [tool_dict[tool] for tool in tool_name]

    text = create_messages_tool(model_name, item, tokenizer, tool_desc, tool_name, prompt_format=prompt_format)
    input_ids = tokenizer(text, return_tensors="pt", return_offsets_mapping=True).to(model.device)
    offset_mapping = input_ids.pop("offset_mapping")
    sentence_token_range, seg_label, inj_label = find_sentence_range(text, offset_mapping, model_name, item, tool_desc, tool_name, prompt_format=prompt_format, without_template=without_template)
    if return_text:
        return text, input_ids, sentence_token_range, seg_label, inj_label
    else:
        return input_ids, sentence_token_range, seg_label, inj_label


def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def transform_tool_format_gpt(tool):
    transformed_tool = {
        "type": "function",
        "function": {
            "name": tool['name'],
            "description": tool['summary'],
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
    for param in tool['parameters']:
        if param['type'] == 'array':
            if 'array of' in param['description']:
                start = param['description'].index('array of')
                item_des = param['description'][start + len("array of"):].strip()
            elif 'list of' in param['description']:
                start = param['description'].index('list of')
                item_des = param['description'][start + len("list of"):].strip()
            else:
                item_des = param['description']

            transformed_tool['function']['parameters']['properties'][param['name']] = {
                "type": 'array',
                "items": {
                    "type": "object",
                    "properties": {
                        param['name']: {
                            "type": "string", "description": item_des},
                    }
                }
            }
        else:
            transformed_tool['function']['parameters']['properties'][param['name']] = {
                "type": param['type'],
                "description": param['description'],
            }

        if param.get('required'):
            transformed_tool['function']['parameters']['required'].append(param['name'])
    return transformed_tool


def get_tool_dict(file_path, gpt_format=False):
    tool_dict = {}
    with open(file_path, 'r') as f:
        toolkits = json.load(f)
        for toolkit in toolkits:
            kit_name = toolkit['toolkit']
            for tool in toolkit['tools']:
                tool_name = kit_name + tool['name']
                tool['name'] = tool_name
                if gpt_format:
                    tool_dict[tool['name']] = transform_tool_format_gpt(tool)
                else:
                    tool_dict[tool['name']] = tool
    return tool_dict


def get_attention_maps_tool(model, tokenizer, model_name, item, tool_dict, prompt_format="json", without_template=False, return_input_ids=False):
    input_ids, sentence_token_range, seg_label, inj_label = get_input_ids_tool(model, tokenizer, model_name, item, tool_dict, prompt_format=prompt_format, without_template=without_template)
    with torch.no_grad():
        '''attention_maps = model(
            **input_ids, return_dict=True, output_attentions=True
        ).attentions'''
        output = model(**input_ids, return_dict=True, output_attentions=True)
        attention_maps = output.attentions
    if not return_input_ids:
        return attention_maps, sentence_token_range, seg_label, inj_label
    else:
        return input_ids, attention_maps, sentence_token_range, seg_label, inj_label


def get_sentence_attn_tool(model, tokenizer, model_name, items, tool_dict, prompt_format="json", without_template=False):
    num_layers = model.config.num_hidden_layers
    attn_last_token = [[] for _ in range(num_layers)]   # [num_layer, num_head, num_item, seq_len] the attn score from last token to this token in this head of this layer
    sentence_token_ranges = []                          # [num_item, seq_len, 2] the token indexes of each sentence in this item
    seg_label_list = []                                 # [num_item] the segment labels of each sentence in this item
    inj_label_list = []                                 # [num_item] the injection labels of each sentence in this item
    for item in tqdm(items):
        attention_maps, sentence_token_range, seg_label, inj_label = get_attention_maps_tool(model, tokenizer, model_name, item, tool_dict, prompt_format=prompt_format, without_template=without_template)
        sentence_token_ranges.append(sentence_token_range)
        seg_label_list.append(seg_label)
        inj_label_list.append(inj_label)

        for i, attention_map in enumerate(attention_maps):
            attn_last_token[i].append(attention_map[:,:,-1,:].squeeze().cpu())   # [num_layer, num_item, num_head, seq_len]

    del attention_maps
    gc.collect()

    max_seq_len = 0
    for item in attn_last_token[0]:
        if len(item[0])>max_seq_len:
            max_seq_len = len(item[0])
    padded_data0 = []
    for it1 in attn_last_token:
        new_outer_list = []
        for it2 in it1:
            new_inner_list = []
            for seq in it2:
                data_pad = F.pad(seq, (0, max_seq_len-len(seq)), mode='constant', value=0)
                new_inner_list.append(data_pad)
            new_outer_list.append(torch.stack(new_inner_list))
        padded_data0.append(torch.stack(new_outer_list))
    padded_data = torch.stack(padded_data0)

    tensor_attn_last_token = padded_data.permute([0, 2, 1, 3])      # [num_layer, num_head, num_item, seq_len]
    return tensor_attn_last_token, sentence_token_ranges, seg_label_list, inj_label_list


def attn_map_segment_each_sentence(attn_maps, sentence_token_ranges, max_seq_len, device, padding=False):
    num_item = len(sentence_token_ranges)
    seg_attn_score = []     # [num_items, seq_len, num_layer, num_head, num_input_div]
    for i in tqdm(range(num_item)):
        score = []
        seq_len = len(sentence_token_ranges[i])
        for l in range(seq_len):
            score.append(torch.flatten(torch.tensor([[[attn_maps[j, k, i, sentence_token_ranges[i][l][0]:sentence_token_ranges[i][l][1]].sum().item(),
                            attn_maps[j, k, i, sentence_token_ranges[i][l][0]:sentence_token_ranges[i][l][1]].mean().item(),
                            attn_maps[j, k, i, sentence_token_ranges[i][l][0]:sentence_token_ranges[i][l][1]].max(),
                            attn_maps[j, k, i, sentence_token_ranges[i][l][0]:sentence_token_ranges[i][l][1]].median()]
                           for k in range(attn_maps[j].size()[0])]
                          for j in range(attn_maps.size()[0])])))
        if padding:
            for l in range(max_seq_len-seq_len):
                score.append(torch.flatten(torch.tensor([[[0,0,0,0] for k in range(attn_maps[j].size()[0])] for j in range(attn_maps.size()[0])])))
        seg_attn_score.append(torch.tensor(np.array([item.cpu().detach().numpy() for item in score])).to(device))
    # tensor_seg_attn_score = torch.tensor(seg_attn_score)
    return torch.tensor(seg_attn_score) if padding else seg_attn_score


def attn_map_segment_each_sentence_critical_heads(attn_maps, sentence_token_ranges, max_seq_len, critical_heads, device, padding=False):   # critical_heads: List[(layer, head)]
    num_item = len(sentence_token_ranges)
    seg_attn_score = []     # [num_items, seq_len, num_critical_heads, num_input_div]
    for i in tqdm(range(num_item)):
        score = []
        seq_len = len(sentence_token_ranges[i])
        for l in range(seq_len):
            score.append(torch.flatten(torch.tensor([[attn_maps[j, k, i, sentence_token_ranges[i][l][0]:sentence_token_ranges[i][l][1]].sum().item(),
                           attn_maps[j, k, i, sentence_token_ranges[i][l][0]:sentence_token_ranges[i][l][1]].mean().item(),
                           attn_maps[j, k, i, sentence_token_ranges[i][l][0]:sentence_token_ranges[i][l][1]].max(),
                           attn_maps[j, k, i, sentence_token_ranges[i][l][0]:sentence_token_ranges[i][l][1]].median()]
                          for (j,k) in critical_heads])))
        if padding:
            for l in range(max_seq_len-seq_len):
                score.append(torch.flatten(torch.tensor([[0,0,0,0] for (j,k) in critical_heads])))
        seg_attn_score.append(torch.tensor([item.cpu().detach().numpy() for item in score]).to(device))
    # tensor_seg_attn_score = torch.tensor(seg_attn_score)
    return torch.tensor(seg_attn_score) if padding else seg_attn_score
