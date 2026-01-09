import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from pos_neg4 import get_eeg_state  # EEG状態取得関数

# ============================
# モデル設定（ローカル Windows対応）
# ============================
MODEL_PATH = "elyza/Llama-3-ELYZA-JP-8B"

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    dtype=torch.float16,       # GPU利用時は高速
    device_map="auto",
    local_files_only=True
)
model.eval()

# ============================
# プロンプト生成（ルールなしLLM全任せ）
# ============================
def make_prompt(user, state, history):
    """
    EEG状態に応じて文体を指定し、履歴とユーザー発話を含める
    """
    # EEG状態ヒント
    if state["SC"] > 70:
        style_hint = "AIは短く落ち着いて返答してください。\n"
    elif state["RC"] > 30:
        style_hint = "AIはやさしく穏やかに返答してください。\n"
    else:
        style_hint = "AIは自然で簡潔に返答してください。\n"

    prompt = (
        style_hint +
        "これまでの会話履歴:\n" +
        history +
        f"人間: {user}\nAI: "
    )
    return prompt

# ============================
# 応答生成
# ============================
def generate_reply(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            min_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # "AI:"の直後の返答を取得
    reply = text.split("AI:")[-1].split("人間:")[0].strip()
    if len(reply) < 5:
        reply = "なるほど、そうなんですね。"
    return reply

# ============================
# メインループ
# ============================
print("=== EEG × 日本語GPT チャット ===")
print("終了するには空行を入力\n")

history = ""

while True:
    try:
        user = input("あなた：").strip()
        if user == "":
            print("終了します")
            break

        state = get_eeg_state()
        prompt = make_prompt(user, state, history)
        reply = generate_reply(prompt)

        print(f"AI：{reply}")
        print(f"(CC:{state['CC']:.1f} RC:{state['RC']:.1f} SC:{state['SC']:.1f})\n")

        # 履歴に追加
        history += f"人間: {user}\nAI: {reply}\n"

    except KeyboardInterrupt:
        print("\n終了します")
        break


