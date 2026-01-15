import torch
import json
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from ccrcsc import get_eeg_state  # EEG状態取得関数
from pos_neg import classify  # ネガポジ判定関数


LAST_VALID_STATE = {
    "CC": 50.0,
    "RC": 50.0,
    "SC": 50.0
}

def get_eeg_state_safe(path="eeg_state.json", retry=3, wait=0.05):
    global LAST_VALID_STATE

    for _ in range(retry):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if not text:
                    raise ValueError("empty file")

                state = json.loads(text)

                # 値チェック（保険）
                if all(k in state for k in ("CC", "RC", "SC")):
                    LAST_VALID_STATE = state
                    return state

        except (json.JSONDecodeError, ValueError, OSError):
            time.sleep(wait)

    # 最終手段：直前の正常値
    return LAST_VALID_STATE

def get_eeg_state_average(duration=10, interval=1.0):
    """
    指定秒数間 EEG を取得して平均化する
    """
    cc_list, rc_list, sc_list = [], [], []

    start_time = time.time()

    while time.time() - start_time < duration:
        state = get_eeg_state_safe()

        cc_list.append(state["CC"])
        rc_list.append(state["RC"])
        sc_list.append(state["SC"])

        time.sleep(interval)

    avg_state = {
        "CC": sum(cc_list) / len(cc_list),
        "RC": sum(rc_list) / len(rc_list),
        "SC": sum(sc_list) / len(sc_list),
    }

    return avg_state

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
def classify_mode_by_sentiment_and_eeg(sentiment_score, state):
    CC, RC, SC = state["CC"], state["RC"], state["SC"]

    # ネガポジ・EEG統合判定
    if SC > 75 and CC < 40 and sentiment_score < -0.3:
        return "overloaded"

    if SC > 65 and CC > 60 and -0.3 <= sentiment_score < -0.1:
        return "panic_focus"

    if RC > 60 and SC < 40 and -0.1 <= sentiment_score <= 0.2:
        return "relaxed"

    if CC > 70 and SC < 50 and sentiment_score > 0.3:
        return "deep_focus"

    if CC < 30 and RC < 30 and -0.1 <= sentiment_score <= 0.1:
        return "disengaged"

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

def make_prompt(user, state, history, sentiment_score=None, sentiment_label=None):
    # ① モード判定
    eeg_mode = classify_mode_by_sentiment_and_eeg(
        sentiment_score=sentiment_score,
        state=state
    )

    # ② プロンプト取得（← これが抜けていた）
    eeg_prompt = EEG_PROMPTS[eeg_mode]

    # ③ 表示用ではなく LLM 用の最小情報
    sentiment_info = ""
    if sentiment_score is not None and sentiment_label is not None:
        sentiment_info = f"文章感情: {sentiment_label}（スコア: {sentiment_score:.3f}）"

    system_prompt = f"""
あなたは人間と自然な日本語会話を行うAIです。

【厳守ルール】
・解説、理由、分析、注釈、メタ発言は禁止
・「AIの出力」「（理由）」などは絶対に書かない
・会話文のみを1発話として返すこと

【現在の相手の状態】
{eeg_prompt}
{sentiment_info}
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
print("=== EEG × ネガポジ判定 ELYZA 日本語チャット ===")
print("終了するには空行を入力\n")

# 履歴に追加する部分
history = ""


while True:
    try:
        user = input("あなた：").strip()
        if user == "":
            print("終了します")
            break

        print("EEG計測中（10秒）...")
        state = get_eeg_state_average(duration=10)
        print("EEG計測完了")

        sentiment_score, sentiment_label = classify(user)

        selected_mode = classify_mode_by_sentiment_and_eeg(sentiment_score, state)

        prompt = make_prompt(
            user,
            state,
            history,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label
)
        reply = generate_reply(prompt, state)

        print(f"AI：{reply}")
        print(f"(CC:{state['CC']:.1f} RC:{state['RC']:.1f} SC:{state['SC']:.1f})")
        print(f"文章感情: {sentiment_label}（スコア: {sentiment_score:.3f}）")
        print(f"選択プロンプト: {selected_mode}")
        
        history += f"人間: {user}\nAI: {reply}\n"

    except KeyboardInterrupt:
        print("\n終了します")
        break



