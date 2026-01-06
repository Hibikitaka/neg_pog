import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, filtfilt, hilbert
from pylsl import StreamInlet, resolve_streams

# =====================
# 設定
# =====================
FS = 256                 # Muse サンプリング周波数
WINDOW_SEC = 4           # 表示窓（秒）
N = FS * WINDOW_SEC

# Muse チャンネル順: TP9, AF7, AF8, TP10
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
# 描画準備
# =====================
fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1])

# ---- 左：α / β 包絡線 ----
ax_wave = fig.add_subplot(gs[:, 0])
line_alpha, = ax_wave.plot(alpha_buf, label="Alpha (8–13Hz)")
line_beta,  = ax_wave.plot(beta_buf, label="Beta (13–30Hz)")
ax_wave.set_ylim(0, 50)
ax_wave.set_title("Band Envelope (AF7 + AF8)")
ax_wave.legend()

# ---- 右上：α/β Ratio（文字＋ゲージ）----
ax_gauge = fig.add_subplot(gs[0, 1])
bar = ax_gauge.barh([0], [1.0])
ax_gauge.set_xlim(0, 3)
ax_gauge.set_yticks([])
ax_gauge.set_title("α / β Ratio")

# ゲージ内テキスト（数値＋状態）
ratio_state_text = ax_gauge.text(
    1.5, 0.0, "",
    ha="center", va="center",
    fontsize=14, fontweight="bold"
)

# ---- 右下：α/β 履歴 ----
ax_ratio = fig.add_subplot(gs[1, 1])
line_ratio, = ax_ratio.plot(ratio_hist)
ax_ratio.set_ylim(0, 3)
ax_ratio.set_title("α / β (history)")

# =====================
# 更新関数
# =====================
def update(frame):
    global eeg_buf, alpha_buf, beta_buf, ratio_hist

    sample, _ = inlet.pull_sample(timeout=0.0)
    if sample is None:
        return

    # AF7 + AF8 平均（安定）
    eeg = (sample[CH_AF7] + sample[CH_AF8]) / 2

    eeg_buf = np.roll(eeg_buf, -1)
    eeg_buf[-1] = eeg

    # 包絡線
    alpha_env = band_envelope(eeg_buf, (8, 13), FS)
    beta_env  = band_envelope(eeg_buf, (13, 30), FS)

    alpha_buf = alpha_env
    beta_buf = beta_env

    # α/β 比（3秒移動平均）
    ratio = alpha_beta_ratio(alpha_env[-FS*3:], beta_env[-FS*3:])
    ratio_hist = np.roll(ratio_hist, -1)
    ratio_hist[-1] = ratio

    # 状態判定
    state, color = ratio_to_state(ratio)

    # 描画更新
    line_alpha.set_ydata(alpha_buf)
    line_beta.set_ydata(beta_buf)

    bar[0].set_width(ratio)
    bar[0].set_color(color)

    ratio_state_text.set_text(f"α/β = {ratio:.2f}\n{state}")
    ratio_state_text.set_color(color)

    line_ratio.set_ydata(ratio_hist)

    return line_alpha, line_beta, bar, line_ratio

# =====================
# 実行
# =====================
ani = FuncAnimation(fig, update, interval=30)
plt.tight_layout()
plt.show()
