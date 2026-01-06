import time
import pickle
import numpy as np
import MeCab
from pylsl import StreamInlet, resolve_byprop

# ===============================
# 設定
# ===============================
FS = 256
EEG_WINDOW_SEC = 3
EEG_BASELINE_SEC = 3   # 最初に baseline を取得する時間

CH_AF7 = 1
CH_AF8 = 2

# ===============================
# MeCab
# ===============================
tagger = MeCab.Tagger("-Ochasen")

def tokenize(text):
    words = []
    for line in tagger.parse(text).split("\n"):
        cols = line.split("\t")
        if len(cols) >= 3:
            base = cols[2]
            if base:
                words.append(base)
    return words

# ===============================
# 感情辞書ロード
# ===============================
with open("sentiment_dict.pkl", "rb") as f:
    sentiment_dict = pickle.load(f)

# ===============================
# 言語感情
# ===============================
def language_sentiment(text):
    words = tokenize(text)
    if not words:
        return 0.0, "中立"

    scores = [sentiment_dict.get(w, (0.0, ""))[0] for w in words]
    score = np.mean(scores)

    if score > 0.1:
        label = "ポジ"
    elif score < -0.1:
        label = "ネガ"
    else:
        label = "中立"

    return score, label

# ===============================
# EEG 接続
# ===============================
print("EEG stream 探索中...")
streams = resolve_byprop("type", "EEG", timeout=5)

if not streams:
    raise RuntimeError("EEG stream が見つかりません")

inlet = StreamInlet(streams[0])
print("EEG 接続完了")

# ===============================
# EEG 収集
# ===============================
def collect_eeg(seconds):
    buf = []
    start = time.time()
    while time.time() - start < seconds:
        sample, _ = inlet.pull_sample(timeout=1.0)
        if sample:
            eeg = (sample[CH_AF7] + sample[CH_AF8]) / 2
            buf.append(eeg)
    return np.array(buf)

# ===============================
# EEG感情（オフセット補正版）
# ===============================
def eeg_sentiment(eeg, baseline):
    if len(eeg) < 10:
        return 0.0
    eeg_corrected = eeg - baseline
    score = np.mean(eeg_corrected) / 50
    return np.clip(score, -1, 1)

# ===============================
# 乖離評価
# ===============================
def divergence_label(D):
    if D < 0.2:
        return "一致"
    elif D < 0.5:
        return "軽度乖離"
    else:
        return "強乖離"

# ===============================
# 最初に baseline を取得
# ===============================
print(f"EEG baseline を {EEG_BASELINE_SEC} 秒間取得中...")
baseline_eeg = collect_eeg(EEG_BASELINE_SEC)
baseline = np.mean(baseline_eeg)
print(f"EEG baseline = {baseline:.3f}")

# ===============================
# メイン
# ===============================
print("\n--- EEG × 言語 乖離計算（Enterで発話）---")

while True:
    text = input("文章：")
    if not text:
        break

    eeg = collect_eeg(EEG_WINDOW_SEC)

    L, lang_label = language_sentiment(text)
    E = eeg_sentiment(eeg, baseline)

    D = abs(L - E)
    S = L - E

    print("\n文章:", text)
    print(f"言語感情 L = {L:.3f} ({lang_label})")
    print(f"EEG感情  E = {E:.3f}")
    print(f"乖離スコア D = {D:.3f}")
    print(f"符号付き乖離 L-E = {S:.3f}")
    print(f"乖離タイプ = {divergence_label(D)}")

