from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "AlekseyKorshuk/llama-7b-japanese"  # 日本語向け
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.float16)
