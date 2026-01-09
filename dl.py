from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Windows でも安全なパス
MODEL_PATH = Path(r"C:\Users\C\projects\pos_neg\elyza_model")

print("モデルロード中…")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),      # Pathオブジェクトを文字列に変換
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()
print("モデルロード完了！GPUに常駐しています。")


# ============================
# 応答生成関数
# ============================
def generate_reply(user_input, history=""):
    state = get_eeg_state()  # EEG取得
    # EEGに応じた文体プロンプト
    if state["SC"] > 70:
        style = "AIは短く落ち着いて返答する。\n"
    elif state["RC"] > 30:
        style = "AIはやさしく穏やかに返答する。\n"
    else:
        style = "AIは自然で簡潔に返答する。\n"

    prompt = (
        history +
        style +
        f"人間: {user_input}\nAI: "
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,    # 長文生成
            min_new_tokens=20,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = text.split("AI:")[-1].split("人間:")[0].strip()

    if len(reply) < 5:
        reply = "なるほど、そうなんですね。"

    return reply, state

# ============================
# 簡易チャット実行（テスト用）
# ============================
if __name__ == "__main__":
    print("=== EEG × 日本語GPT チャット ===")
    print("終了するには空行を入力\n")

    history = ""
    while True:
        try:
            user = input("あなた：").strip()
            if user == "":
                print("終了します")
                break

            reply, state = generate_reply(user, history)
            print(f"AI：{reply}")
            print(f"(CC:{state['CC']:.1f} RC:{state['RC']:.1f} SC:{state['SC']:.1f})\n")
            history += f"人間: {user}\nAI: {reply}\n"

        except KeyboardInterrupt:
            print("\n終了します")
            break
