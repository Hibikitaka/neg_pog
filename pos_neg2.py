import time
import pickle
import numpy as np
import MeCab
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pylsl import StreamInlet, resolve_byprop

# ===============================
# 設定
# ===============================
FS = 256
EEG_WINDOW_SEC = 3
EEG_BASELINE_SEC = 3

CH_AF7 = 1
CH_AF8 = 2

MAX_POINTS = 50  # グラフ表示の最大点数

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

def collect_eeg(seconds):
    buf = []
    start = time.time()
    while time.time() - start < seconds:
        sample, _ = inlet.pull_sample(timeout=1.0)
        if sample:
            eeg = (sample[CH_AF7] + sample[CH_AF8]) / 2
            buf.append(eeg)
    return np.array(buf)

def eeg_sentiment(eeg, baseline):
    if len(eeg) < 10:
        return 0.0
    eeg_corrected = eeg - baseline
    score = np.mean(eeg_corrected) / 50
    return np.clip(score, -1, 1)

def divergence_label(D):
    if D < 0.2:
        return "一致"
    elif D < 0.5:
        return "軽度乖離"
    else:
        return "強乖離"

# ===============================
# baseline 取得
# ===============================
print(f"EEG baseline を {EEG_BASELINE_SEC} 秒間取得中...")
baseline_eeg = collect_eeg(EEG_BASELINE_SEC)
baseline = np.mean(baseline_eeg)
print(f"EEG baseline = {baseline:.3f}")

# ===============================
# データ保存用リスト
# ===============================
timestamps = []
L_list = []
E_list = []
D_list = []
labels = []

# ===============================
# グラフ初期化
# ===============================
plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), sharex=True)
line_L, = ax1.plot([], [], label='言語感情 L', color='blue')
line_E, = ax1.plot([], [], label='EEG感情 E', color='red')
ax1.set_ylabel("感情スコア")
ax1.legend()
ax1.set_ylim(-1, 1)

bar_colors = {'一致':'green', '軽度乖離':'yellow', '強乖離':'red'}
bars = ax2.bar([], [])
ax2.set_ylabel("乖離スコア D")
ax2.set_ylim(0, 1.5)

# ===============================
# アニメーション更新関数
# ===============================
def update(frame):
    line_L.set_data(range(len(L_list)), L_list)
    line_E.set_data(range(len(E_list)), E_list)

    ax2.clear()
    ax2.set_ylim(0,1.5)
    colors = [bar_colors.get(lbl, 'grey') for lbl in labels]
    ax2.bar(range(len(D_list)), D_list, color=colors)
    ax2.set_ylabel("乖離スコア D")

    ax1.set_xlim(0, max(MAX_POINTS, len(L_list)))
    return line_L, line_E

ani = FuncAnimation(fig, update, interval=500)

# ===============================
# メイン入力ループ
# ===============================
print("\n--- EEG × 言語 乖離計算（Enterで発話）---")
plt.ion()
plt.show()

while True:
    text = input("文章：")
    if not text:
        break

    eeg = collect_eeg(EEG_WINDOW_SEC)

    L, lang_label = language_sentiment(text)
    E = eeg_sentiment(eeg, baseline)
    D = abs(L - E)
    S = L - E
    div_label = divergence_label(D)

    # データ追加
    timestamps.append(time.time())
    L_list.append(L)
    E_list.append(E)
    D_list.append(D)
    labels.append(div_label)

    # 最新50点まで保持
    L_list = L_list[-MAX_POINTS:]
    E_list = E_list[-MAX_POINTS:]
    D_list = D_list[-MAX_POINTS:]
    labels = labels[-MAX_POINTS:]

    print("\n文章:", text)
    print(f"言語感情 L = {L:.3f} ({lang_label})")
    print(f"EEG感情  E = {E:.3f}")
    print(f"乖離スコア D = {D:.3f}")
    print(f"符号付き乖離 L-E = {S:.3f}")
    print(f"乖離タイプ = {div_label}")

    plt.pause(0.1)
