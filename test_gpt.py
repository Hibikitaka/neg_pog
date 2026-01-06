from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "rinna/japanese-gpt-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("=== 日本語GPT チャット ===")
print("終了するには空行を入力してください。\n")

while True:
    user_input = input("あなた：")
    if not user_input:
        break

    # 入力文章の後に AI: を付ける
    prompt = f"{user_input}\nAI:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )

    # デコードして AI の応答だけ取り出す
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = generated_text[len(prompt):].strip()  # 入力部分を削除

    print(f"AI：{reply}\n")

