from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "rinna/japanese-gpt-1b"

# slow tokenizer を使う
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(model_name)
