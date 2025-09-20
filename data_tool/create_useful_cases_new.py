import json
import jsonlines
import random

def load_data_from_json(file_path):
    """从指定路径加载JSON文件并返回其内容"""
    data = []
    with open(file_path, "r") as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data


def save_data_to_json(data, file_path):
    """将数据保存到指定路径的JSON文件中"""
    with jsonlines.open(file_path, mode="a") as file_jsonl:
        for line in data:
            file_jsonl.write(line)


def create_useful_set(input_file, output_file):
    """从输入JSON文件创建一个包含subset_size条目的子集JSON文件"""
    # 加载原始数据
    data = load_data_from_json(input_file)
    useful_content_file = "data_tool/useful_content.jsonl"
    useful_contents = []
    with open(useful_content_file, "r") as file:
        for line in file:
            json_obj = json.loads(line)
            useful_contents.append(json_obj)
    useful_content_tool_list = []
    for content in useful_contents:
        useful_content_tool_list.append(content["User Tool"])

    for item in data:
        if item["User Tool"] in useful_content_tool_list:
            for useful_content in useful_contents:
                if useful_content["User Tool"]==item["User Tool"]:
                    index = item["Tool Response Template"].find("<Attacker Instruction>")
                    item["Tool Response Template"] = item["Tool Response Template"][:index] + useful_content["Useful Content"] + " " + item["Tool Response Template"][index:]

    for d in data:
        print(d)
        print("\n\n")
    # 将子集保存到新文件
    save_data_to_json(data, output_file)


if __name__ == "__main__":
    input_json_file = f'data_tool/user_cases_new.jsonl'
    output_json_file = f'data_tool/user_cases_useful_new.jsonl'
    create_useful_set(input_json_file, output_json_file)