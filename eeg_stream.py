import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, filtfilt, hilbert
from pylsl import StreamInlet, resolve_streams
import time
import threading
import json

# =====================
# 設定
# =====================
FS = 256
WINDOW_SEC = 4
N = FS * WINDOW_SEC

CH_AF7 = 1
CH_AF8 = 2

# =====================
# 信号処理
# =====================
def bandpass(data, low, high, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, data)

def smooth_envelope(env, fs, cutoff=1.5):
    nyq = fs / 2
    b, a = butter(2, cutoff/nyq, btype="low")
    return filtfilt(b, a, env)

def band_envelope(data, band, fs):
    filtered = bandpass(data, band[0], band[1], fs)
    env = np.abs(hilbert(filtered))
    return smooth_envelope(env, fs)

def alpha_beta_ratio(alpha_env, beta_env, eps=1e-6):
    return np.mean(alpha_env) / (np.mean(beta_env) + eps)

def ratio_to_state(ratio):
    if ratio < 0.6:
        return "集中", "red"
    elif ratio < 1.0:
        return "安定", "orange"
    elif ratio > 1.3:
        return "リラックス", "green"
    else:
        return "中間", "gray"

# =====================
# ★ EEG / 言語ログ
# =====================
eeg_log = []        # 時間付き EEG 感情グラデーション
language_log = []   # 言語イベント

# =====================
# ★ 言語入力スレッド
# =====================
def language_input_loop():
    while True:
        text = input("発話 > ")
        t = time.time()
        language_log.append({
            "time": t,
            "text": text
        })
        print(f"[LOG] 言語入力 @ {t:.2f}")

threading.Thread(target=language_input_loop, daemon=True).start()

# =====================
# LSL 接続
# =====================
print("EEG stream を探索中...")
streams = resolve_streams()
eeg_streams = [s for s in streams if s.type() == "EEG"]

if not eeg_streams:
    raise RuntimeError("EEG stream が見つかりません")

inlet = StreamInlet(eeg_streams[0])
print("EEG stream 接続完了")

# =====================
# バッファ
# =====================
eeg_buf = np.zeros(N)
alpha_buf = np.zeros(N)
beta_buf = np.zeros(N)
ratio_hist = np.zeros(N)

# =====================
# 描画
# =====================
fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1])

ax_wave = fig.add_subplot(gs[:, 0])
line_alpha, = ax_wave.plot(alpha_buf, label="Alpha")
line_beta,  = ax_wave.plot(beta_buf, label="Beta")
ax_wave.set_ylim(0, 50)
ax_wave.legend()

ax_gauge = fig.add_subplot(gs[0, 1])
bar = ax_gauge.barh([0], [1.0])
ax_gauge.set_xlim(0, 3)
ax_gauge.set_yticks([])
ax_gauge.set_title("α / β Ratio")

ratio_state_text = ax_gauge.text(
    1.5, 0.0, "", ha="center", va="center", fontsize=14
)

ax_ratio = fig.add_subplot(gs[1, 1])
line_ratio, = ax_ratio.plot(ratio_hist)
ax_ratio.set_ylim(0, 3)

# =====================
# 更新関数
# =====================
def update(frame):
    global eeg_buf, ratio_hist

    sample, _ = inlet.pull_sample(timeout=0.0)
    if sample is None:
        return

    eeg = (sample[CH_AF7] + sample[CH_AF8]) / 2
    eeg_buf = np.roll(eeg_buf, -1)
    eeg_buf[-1] = eeg

    alpha_env = band_envelope(eeg_buf, (8, 13), FS)
    beta_env  = band_envelope(eeg_buf, (13, 30), FS)

    ratio = alpha_beta_ratio(alpha_env[-FS*3:], beta_env[-FS*3:])
    ratio_hist = np.roll(ratio_hist, -1)
    ratio_hist[-1] = ratio

    state, color = ratio_to_state(ratio)

    # ★ EEG ログ保存
    eeg_log.append({
        "time": time.time(),
        "alpha_beta_ratio": float(ratio),
        "state": state
    })

    line_alpha.set_ydata(alpha_env)
    line_beta.set_ydata(beta_env)

    bar[0].set_width(ratio)
    bar[0].set_color(color)
    ratio_state_text.set_text(f"{ratio:.2f}\n{state}")
    ratio_state_text.set_color(color)

    line_ratio.set_ydata(ratio_hist)

    return line_alpha, line_beta, bar, line_ratio

# =====================
# 実行
# =====================
ani = FuncAnimation(fig, update, interval=30)
plt.tight_layout()
plt.show()

# =====================
# ★ 終了時にログ保存
# =====================
with open("eeg_log.json", "w", encoding="utf-8") as f:
    json.dump(eeg_log, f, ensure_ascii=False, indent=2)

with open("language_log.json", "w", encoding="utf-8") as f:
    json.dump(language_log, f, ensure_ascii=False, indent=2)

