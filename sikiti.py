from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "rinna/japanese-gpt-1b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

input_text = "今日は天気が良いので、"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(generated[0], skip_special_tokens=True))
