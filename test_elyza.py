import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from CC_RC_SC.py import get_eeg_state  # EEG状態取得関数

def get_eeg_state_safe():
    """
    eeg_state.json を読む or fallback 50固定
    """
    # ファイルから最新状態を読む
    if not os.path.exists("eeg_state.json"):
        state = {"CC": 50, "RC": 50, "SC": 50}
    else:
        with open("eeg_state.json", "r", encoding="utf-8") as f:
            state = json.load(f)

    # 初期化直後・取得失敗時の保険
    if (
        state is None
        or "CC" not in state
        or "RC" not in state
        or "SC" not in state
        or (state["CC"] == 0 and state["RC"] == 0 and state["SC"] == 0)
    ):
        state = {"CC": 50, "RC": 50, "SC": 50}

    return state


# ============================
# モデル設定（ローカル Windows対応）
# ============================
MODEL_PATH = "elyza/Llama-3-ELYZA-JP-8B"

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    dtype=torch.float16,       
    device_map="auto",
    local_files_only=True
)
model.eval()

# ============================
# プロンプト生成
# ============================
def classify_eeg_state(state):
    CC, RC, SC = state["CC"], state["RC"], state["SC"]

    if SC > 75 and CC < 40:
        return "overloaded"      # ストレス過多
    if SC > 65 and CC > 60:
        return "panic_focus"     # 焦り集中
    if RC > 60 and SC < 40:
        return "relaxed"         # 安定
    if CC > 70 and SC < 50:
        return "deep_focus"      # 深い集中
    if CC < 30 and RC < 30:
        return "disengaged"      # 注意散漫

    return "neutral"

EEG_PROMPTS = {
    "overloaded": """
AIは相手が疲れていることを理解し、
情報量を減らし、安心感を与える返答をしてください。
質問は1つまでにしてください。
""",

    "panic_focus": """
AIは相手が焦って集中している状態であることを理解し、
手順を整理して箇条書きで説明してください。
感情的な表現は控えてください。
""",

    "relaxed": """
AIは相手が落ち着いている状態であることを理解し、
少し余談や例えを交えながら自然に会話してください。
""",

    "deep_focus": """
AIは相手が深く集中している状態であることを理解し、
論理的で無駄のない説明を行ってください。
専門用語の使用を許可します。
""",

    "disengaged": """
AIは相手の注意が散っている可能性を考慮し、
短い文でテンポよく返答してください。
興味を引く一言を最初に入れてください。
""",

    "neutral": """
AIは自然で簡潔な会話を心がけてください。
"""
}

def make_prompt(user, state, history):
    eeg_mode = classify_eeg_state(state)
    eeg_prompt = EEG_PROMPTS[eeg_mode]

    system_prompt = f"""
あなたは人間と自然な日本語会話を行うAIです。

【厳守ルール】
・解説、理由、分析、注釈、メタ発言は禁止
・「AIの出力」「（理由）」などは絶対に書かない
・会話文のみを1発話として返すこと

【現在の相手の状態】
{eeg_prompt}
"""

    return (
        system_prompt +
        "これまでの会話履歴:\n" +
        history +
        f"人間: {user}\nAI: "
    )


# ============================
# 応答生成
# ============================

def get_generation_params(state):
    SC, CC, RC = state["SC"], state["CC"], state["RC"]

    params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 120
    }

    if SC > 70:
        params["temperature"] = 0.5
        params["max_new_tokens"] = 60

    if CC > 70:
        params["temperature"] = 0.6
        params["top_p"] = 0.9

    if RC > 60:
        params["temperature"] = 0.9

    return params

def generate_reply(prompt, state):
    gen_params = get_generation_params(state)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            **gen_params
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    reply = text.split("AI:")[-1].split("人間:")[0].strip()

    if len(reply) < 3:
        reply = "うん、そうなんですね。"

    return reply

# ============================
# メインループ
# ============================
print("=== EEG × 日本語GPT チャット ===")
print("終了するには空行を入力\n")

# 履歴に追加する部分
history = ""


while True:
    try:
        user = input("あなた：").strip()
        if user == "":
            print("終了します")
            break

        state = get_eeg_state_safe()
        prompt = make_prompt(user, state, history)
        reply = generate_reply(prompt, state)

        print(f"AI：{reply}")
        print(f"(CC:{state['CC']:.1f} RC:{state['RC']:.1f} SC:{state['SC']:.1f})\n")

        history += f"人間: {user}\nAI: {reply}\n"

    except KeyboardInterrupt:
        print("\n終了します")
        break



