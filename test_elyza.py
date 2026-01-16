import torch
import json
import os
import time
import logging
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from ccrcsc import get_eeg_state  # EEG状態取得関数
from pos_neg import classify  # ネガポジ判定関数

# ============================
# logging 設定
# ============================
log_name = datetime.now().strftime("test_elyza_%Y%m%d_%H%M.log")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ファイル出力
fh = logging.FileHandler(log_name, encoding="utf-8")
# ターミナル出力
sh = logging.StreamHandler()

formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
)

fh.setFormatter(formatter)
sh.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(sh)

logging.info("=== test_elyza.py 起動 ===")


LAST_VALID_STATE = {
    "CC": 50.0,
    "RC": 50.0,
    "SC": 50.0
}

MODE_CENTERS = {
    "overloaded": {
        "CC": 30, "RC": 40, "SC": 80, "sent": -1.0
    },
    "panic_focus": {
        "CC": 70, "RC": 35, "SC": 70, "sent": -0.1
    },
    "relaxed": {
        "CC": 50, "RC": 70, "SC": 50, "sent": -0.06
    },
    "deep_focus": {
        "CC": 80, "RC": 50, "SC": 40, "sent": -0.06 
    },
    "disengaged": {
        "CC": 30, "RC": 30, "SC": 60, "sent": -1.0
    }
}

def calc_distance(state, sentiment_score, center):
    return (
        (state["CC"] - center["CC"]) ** 2 +
        (state["RC"] - center["RC"]) ** 2 +
        (state["SC"] - center["SC"]) ** 2 +
        ((sentiment_score * 50) - (center["sent"] * 50)) ** 2
    ) ** 0.5


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

logging.info("ELYZAモデルロード開始")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)
model.eval()

logging.info("ELYZAモデルロード完了")

# ============================
# プロンプト生成
# ============================
def classify_mode_by_sentiment_and_eeg(sentiment_score, state):
    distances = {}

    for mode, center in MODE_CENTERS.items():
        d = calc_distance(state, sentiment_score, center)
        distances[mode] = d

    # 距離が最小のモードを採用
    selected_mode = min(distances, key=distances.get)

    return selected_mode



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
print("=== EEG × ネガポジ判定 ELYZA 日本語チャット ===\n")
history = ""

while True:
    try:
        user = input("あなた：").strip()
        if user == "":
            logging.info("ユーザー終了")
            break

        logging.info(f"USER: {user}")

        print("EEG計測中（10秒）...")
        state = get_eeg_state_average(duration=10)
        logging.info(
            f"EEG_AVG CC={state['CC']:.2f} RC={state['RC']:.2f} SC={state['SC']:.2f}"
        )

        sentiment_score, sentiment_label = classify(user)
        logging.info(
            f"SENTIMENT {sentiment_label} ({sentiment_score:.3f})"
        )

        selected_mode = classify_mode_by_sentiment_and_eeg(sentiment_score, state)
        logging.info(f"MODE {selected_mode}")

        prompt = make_prompt(
            user, state, history,
            sentiment_score, sentiment_label
        )

        reply = generate_reply(prompt, state)

        print(f"AI：{reply}")
        logging.info(f"AI: {reply}")

        history += f"人間: {user}\nAI: {reply}\n"

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt 終了")
        break

logging.info("=== test_elyza.py 終了 ===")



