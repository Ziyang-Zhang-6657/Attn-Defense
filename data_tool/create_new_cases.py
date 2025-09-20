import json
from json_repair import repair_json, loads
import jsonlines
from openai import OpenAI
from prompts.generation_prompts import SYSTEM_MESSAGE, ATTACKER_INSTRUCTION_GEN_HUMAN_MESSAGE, USER_INSTRUCTION_GEN_HUMAN_MESSAGE, EXAMPLE




client = OpenAI(
            base_url='https://xiaoai.plus/v1',
            # sk-xxx替换为自己的key
            api_key='sk-YP2RHFk6AQGUWtgCaMG9hdm685ZtkcFX1Uf5vplQjzI1VHCc'
        )



attack_cases = []
attack_tools = []
user_cases = []
user_tools = []
with open("data_tool/attacker_cases_dh_new.jsonl", 'r', encoding='utf-8') as f:
    for line in f.readlines():
        # attack_cases.append(line)
        attack_cases.append(json.loads(line))
with open("data_tool/user_cases_new.jsonl", 'r', encoding='utf-8') as f:
    for line in f.readlines():
        # user_cases.append(line)
        user_cases.append(json.loads(line))

for case in attack_cases:
    # attack_tools.append(json.loads(case)["Attacker Tools"][0])
    attack_tools.append(case["Attacker Tools"][0])
for case in user_cases:
    # user_tools.append(json.loads(case)["User Tool"])
    user_tools.append(case["User Tool"])
# print(attack_tools)
# print(user_tools)


tool_name_list = []
tool_list = []
with open('data_tool/tools.json', 'r') as f:
    toolkits = json.load(f)
    for toolkit in toolkits:
        toolkit_tool_name_list = []
        toolkit_tool_list = []
        kit_name = toolkit['toolkit']
        for tool in toolkit['tools']:
            tool_name = kit_name + tool['name']
            tool['name'] = tool_name
            if (tool_name in attack_tools) or (tool_name in user_tools) or tool_name=="GmailSendEmail":
                continue
            else:
                toolkit_tool_name_list.append(tool_name)
                toolkit_tool_list.append(tool)
        tool_name_list.append(toolkit_tool_name_list)
        tool_list.append(toolkit_tool_list)




# prompt = f"""I need you to help me expand my dataset by generating new JSON samples based on the provided examples. Here are the requirements:
#
# Reference Existing Examples:
# Study the following JSON examples to understand the format and content structure:
# {attack_cases[0]}
# {attack_cases[2]}
# {attack_cases[9]}
# Avoid Repetition:
# Ensure no duplicate Attacker Tools or Attacker Instructions from the examples I provided. The new entries must introduce entirely new tools and instructions not seen in the examples.
# You should choose the tool from the name list below:
# {tool_name_list}
# Maintain Structure:
# Keep the same JSON structure . Include all required fields present in the examples.
#
# Attention, you should generate detailed attack instructions with specific input, so that it can be executed easily. And the instruction should be no more than 30 words.
# Generate 1 new unique entries (adjust the number as needed). After that, you only need to output json format content without any other words. Let's begin!"""
# # print(prompt)
#
# completion = client.chat.completions.create(
#           model="gpt-4o",
#           messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#           ]
#         )


attacker_cases_examples = [attack_cases[0], attack_cases[2], attack_cases[9]]
# user_cases_examples = [user_cases[0], user_cases[16], user_cases[17]]
# user_cases_examples = str(user_cases[0]) + "\n" + str(user_cases[16]) + "\n" + str(user_cases[17])
user_cases_examples = json.dumps(user_cases[0], ensure_ascii=False, indent=2) + "\n" + json.dumps(user_cases[16], ensure_ascii=False, indent=2) + "\n" + json.dumps(user_cases[17], ensure_ascii=False, indent=2)


print("\n" + "+" * 100 + "\n")
print(len(tool_list))
while ([] in tool_list):
    tool_list.remove([])
print(len(tool_list))
print("\n" + "+" * 100 + "\n")

for i in range(10):
    for toolkit_tool_list in tool_list[:]:
        if len(toolkit_tool_list)>i:
            tool = toolkit_tool_list[i]
        else:
            continue
        # candidate_tools = toolkit_tool_list
        # print("\n" + "+" * 100 + "\n")
        # print(toolkit_tool_list)
        # print("\n" + "+" * 100 + "\n")

        # prompt = ATTACKER_INSTRUCTION_GEN_HUMAN_MESSAGE.format(examples=attacker_cases_examples, candidate_tools=candidate_tools, num_case=1)
        prompt = USER_INSTRUCTION_GEN_HUMAN_MESSAGE.format(examples=user_cases_examples, user_tool=tool, num_case=1)
        completion = client.chat.completions.create(
                  model="gpt-4o",
                  messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                  ]
                )
        try:
            content = completion.choices[0].message.content

            print("parameters:")
            for p in tool["parameters"]:
                print(p)
            print("\n" + "+" * 100 + "\n")

            print("returns:")
            for r in tool["returns"]:
                print(r)
            print("\n" + "+" * 100 + "\n")

            json_content = content[content.rfind('{\n'):content.rfind('}\n') + 1]

            print("案例：")
            print(json_content)
            print("\n" + "+" * 100 + "\n")

            json_content = repair_json(json_content, ensure_ascii=False)

            print("json repair案例：")
            print(json_content)
            print("\n" + "+" * 100 + "\n")

            # print(content[content.rfind('{'):content.rfind('}\n')+1].replace('\'','\"'))
            # print("\n" + "+" * 100 + "\n")

            judge = input("输入y接受案例进行写入，否则拒绝：")
            if judge=="y":
                with jsonlines.open("data_tool/user_cases_new.jsonl", mode="a") as file_jsonl:
                    file_jsonl.write(loads(json_content))
        except Exception as e:
            print(f"An error occurred: {e}")