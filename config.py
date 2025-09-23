"""
Configuration file for JBShield.
"""

# Data paths
path_harmful = "data/harmful.csv"
path_non_harmless = "data/harmless.csv"
path_harmful_test = "data/harmful_test.csv"
path_non_harmless_test = "data/harmless_test.csv"
path_harmful_calibration = "data/harmful_calibration.csv"
path_non_harmless_calibration = "data/harmless_calibration.csv"
# xstest_path = "data/xstest_V2_prompts.csv"

# Model paths
model_paths = {
    "mistral-sorry-bench": "./models/ft-mistral-7b-instruct-v0.2-sorry-bench-202406",
    "Meta-Llama-3.1-8B-Instruct": "/opt/data/private/model/Meta-Llama-3.1-8B-Instruct",
    "Meta-Llama-3-8B-Instruct": "/opt/data/private/model/Meta-Llama-3-8B-Instruct",
    "Qwen2.5-0.5B-Instruct": "/opt/data/private/model/Qwen2.5-0.5B-Instruct",
    "Qwen2.5-7b-instruct": "/opt/data/private/model/Qwen2.5-7b-instruct",
    "Qwen2-7b-instruct": "/opt/data/private/model/Qwen2-7b-instruct",
    "ToolLLaMA-2-7b-v2": "/opt/data/private/model/ToolLLaMA-2-7b-v2",
    "Mistral-7B-Instruct-v0.3": "/opt/data/private/model/Mistral-7B-Instruct-v0.3",
    "internlm2_5-7b-chat": "/opt/data/private/model/internlm2_5-7b-chat",
    "Vicuna-7b-v1.5": "/opt/data/private/model/Vicuna-7b-v1.5",
    "Vicuna-13b-v1.5": "/opt/data/private/model/Vicuna-13b-v1.5",
    "Vicuna-7b": "/opt/data/private/model/Vicuna-7b",
    "internlm2.5-7b-chat": "/opt/data/private/model/internlm2_5-7b-chat",
    "Falcon3-7B-Instruct": "/opt/data/private/model/Falcon3-7B-Instruct",
    "Tsunami-0.5-7B-Instruct": "/opt/data/private/model/Tsunami-0.5-7B-Instruct",
    "phi-4-unsloth-bnb-4bit": "/opt/data/private/model/phi-4-unsloth-bnb-4bit",
    "Phi-4-mini-instruct-unsloth-bnb-4bit": "/opt/data/private/model/Phi-4-mini-instruct-unsloth-bnb-4bit",
    "deberta-v3-base-prompt-injection-v2": "/opt/data/private/model/deberta-v3-base-prompt-injection-v2",
}


special_token_ids = {
    "Meta-Llama-3.1-8B-Instruct": 128006,
    "Meta-Llama-3-8B-Instruct": 128006,
    "Qwen2.5-7b-instruct": 151644,
    "Qwen2-7b-instruct": 151644,
    "Vicuna-7b-v1.5": 1,
    "Vicuna-13b-v1.5": 1,
    "Vicuna-7b": 1,
    "internlm2_5-7b-chat": 1,
    "Falcon3-7B-Instruct": 10,
    "Tsunami-0.5-7B-Instruct": 151644,
    "phi-4-unsloth-bnb-4bit": 100264,
    "Phi-4-mini-instruct-unsloth-bnb-4bit": 200022,
}
