from transformers import AutoTokenizer, AutoModelForCausalLM

# 正しいモデル名を指定
model_name = "AlekseyKorshuk/llama-7b-japanese-hf"

# トークン認証付きでロード
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, token=True)

# テスト生成
inputs = tokenizer("今日はいい天気ですね。", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
