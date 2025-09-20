import json
import random



def load_data_from_json(file_path):
    """从指定路径加载JSON文件并返回其内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(len(data))
    return data


def save_data_to_json(data, file_path):
    """将数据保存到指定路径的JSON文件中"""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def create_harmless_set(input_file, output_file):
    """从输入JSON文件创建一个包含subset_size条目的子集JSON文件"""
    # 加载原始数据
    data = load_data_from_json(input_file)

    for item in data:
        attacker_instruction = item["Attacker Instruction"]
        if item["Attacker Instruction"] in item["Tool Response"]:
            item["Tool Response"] = item["Tool Response"].replace(attacker_instruction,"None")
        else:
            print("Attacker Instruction not found!!!")

    # 将子集保存到新文件
    save_data_to_json(data, output_file)


if __name__ == "__main__":
    for attack in ["dh", "ds"]:
        for difficulty in ["base", "enhanced"]:
            for type in ["", "calibration_", "test_"]:
                input_json_file = f'{type}{attack}_{difficulty}.json'
                output_json_file = f'{type}{attack}_{difficulty}_harmless.json'
                create_harmless_set(input_json_file, output_json_file)