# front_alphabeta_asymmetry.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import font_manager, rcParams
from scipy.signal import butter, filtfilt, hilbert
from pylsl import StreamInlet, resolve_streams
import time
import csv
import json
import math

# =====================
# 最新EEG状態
# =====================
LATEST_STATE = {"CC": 0.0, "RC": 0.0, "SC": 0.0}

def save_state():
    with open("eeg_state_faa.json", "w", encoding="utf-8") as f:
        json.dump(LATEST_STATE, f, ensure_ascii=False)

# =====================
# 日本語フォント
# =====================
font_path = "C:/Windows/Fonts/meiryo.ttc"
font_prop = font_manager.FontProperties(fname=font_path)
rcParams["font.family"] = font_prop.get_name()

# =====================
# 設定
# =====================
FS = 256
WINDOW_SEC = 4
N = FS * WINDOW_SEC

CH_AF7 = 1  # 左前頭
CH_AF8 = 2  # 右前頭

# =====================
# 信号処理
# =====================
def bandpass(data, low, high, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)

def envelope(data):
    return np.abs(hilbert(data))

def band_env(data, band):
    return envelope(bandpass(data, band[0], band[1], FS))

# =====================
# 正規化
# =====================
EPS = 1e-6

def normalize(x, xmin, xmax):
    x = max(xmin, min(x, xmax))
    return int((x - xmin) / (xmax - xmin) * 100)

# =====================
# FAA派生 CC / RC / SC
# =====================
def CC(beta_L, beta_R):
    # 集中：β左優位
    val = math.log(beta_L + EPS) - math.log(beta_R + EPS)
    return normalize(val, -1.0, 1.0)

def RC(alpha_L, alpha_R):
    # リラックス：α量
    val = (alpha_L + alpha_R) / 2
    return normalize(val, 0.2, 2.0)

def SC(alpha_L, alpha_R):
    # ストレス：α右優位
    val = math.log(alpha_R + EPS) - math.log(alpha_L + EPS)
    return normalize(val, -1.0, 1.0)

# =====================
# メイン
# =====================
def main():
    print("EEG stream 探索中...")
    streams = resolve_streams()
    eeg_streams = [s for s in streams if s.type() == "EEG"]

    if not eeg_streams:
        raise RuntimeError("EEG stream が見つかりません")

    inlet = StreamInlet(eeg_streams[0])
    print("EEG stream 接続完了")
    start_time = time.time()

    buf_L = np.zeros(N)
    buf_R = np.zeros(N)

    # ---------- CSV ----------
    csv_file = open("eeg_faa.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow([
        "timestamp",
        "alpha_L", "alpha_R",
        "beta_L", "beta_R",
        "CC", "RC", "SC"
    ])

    # ---------- 描画 ----------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_ylim(0, 100)
    line_cc, = ax.plot([], [], label="集中 CC")
    line_rc, = ax.plot([], [], label="リラックス RC")
    line_sc, = ax.plot([], [], label="ストレス SC")
    ax.legend()

    history = {"CC": [], "RC": [], "SC": []}

    def update(frame):
        sample, _ = inlet.pull_sample(timeout=0.0)
        if sample is None:
            return

        buf_L[:-1] = buf_L[1:]
        buf_R[:-1] = buf_R[1:]
        buf_L[-1] = sample[CH_AF7]
        buf_R[-1] = sample[CH_AF8]

        alpha_L = np.mean(band_env(buf_L, (8, 13))[-FS*2:])
        alpha_R = np.mean(band_env(buf_R, (8, 13))[-FS*2:])
        beta_L  = np.mean(band_env(buf_L, (13, 30))[-FS*2:])
        beta_R  = np.mean(band_env(buf_R, (13, 30))[-FS*2:])

        cc = CC(beta_L, beta_R)
        rc = RC(alpha_L, alpha_R)
        sc = SC(alpha_L, alpha_R)

        LATEST_STATE.update({"CC": cc, "RC": rc, "SC": sc})
        save_state()

        for k, v in zip(["CC", "RC", "SC"], [cc, rc, sc]):
            history[k].append(v)
            if len(history[k]) > 300:
                history[k].pop(0)

        line_cc.set_data(range(len(history["CC"])), history["CC"])
        line_rc.set_data(range(len(history["RC"])), history["RC"])
        line_sc.set_data(range(len(history["SC"])), history["SC"])
        ax.set_xlim(0, len(history["CC"]))

        writer.writerow([
            time.time(),
            alpha_L, alpha_R,
            beta_L, beta_R,
            cc, rc, sc
        ])
        csv_file.flush()

        return line_cc, line_rc, line_sc

    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.show()
    csv_file.close()

if __name__ == "__main__":
    main()
