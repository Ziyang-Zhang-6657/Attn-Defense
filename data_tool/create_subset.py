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


def create_subset(input_file, output_file1, output_file2, subset_size=0.2):
    """从输入JSON文件创建一个包含subset_size条目的子集JSON文件"""
    # 加载原始数据
    data = load_data_from_json(input_file)
    subset_size = int(len(data)*subset_size)
    print(subset_size)


    # 确保数据是列表类型并且有足够多的条目可供抽取
    if not isinstance(data, list) or len(data) < subset_size:
        raise ValueError("输入数据不是列表或条目数少于所需子集大小")

    # 随机打乱数据
    random.shuffle(data)

    # 抽取前subset_size条目作为子集
    subset1 = data[:subset_size]
    subset2 = data[subset_size:]

    print("All: ", len(data))
    print("Sub1: ", len(subset1))
    print("Sub2: ", len(subset2))
    # 将子集保存到新文件
    save_data_to_json(subset1, output_file1)
    save_data_to_json(subset2, output_file2)


if __name__ == "__main__":
    for attack in ["dh", "ds"]:
        for difficulty in ["base", "enhanced"]:
            input_json_file = f'{attack}_{difficulty}.json'  # 替换为你的输入JSON文件路径
            output_json_file1 = f'calibration_{attack}_{difficulty}.json'  # 替换为你的输出JSON文件路径
            output_json_file2 = f'test_{attack}_{difficulty}.json'
            create_subset(input_json_file, output_json_file1, output_json_file2)
