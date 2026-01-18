import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import font_manager, rcParams
from scipy.signal import butter, filtfilt
from pylsl import StreamInlet, resolve_streams
import time
import csv
import json

# =====================
# Sample Entropy
# =====================
def sample_entropy(signal, m=2, r=0.2):
    signal = np.array(signal)
    N = len(signal)
    if N < m + 2:
        return 0.0

    r *= np.std(signal)
    def _phi(m):
        x = np.array([signal[i:i+m] for i in range(N - m + 1)])
        C = np.sum(
            np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r,
            axis=0
        ) - 1
        return np.sum(C) / ((N - m + 1) * (N - m))

    return -np.log(_phi(m+1) / _phi(m) + 1e-10)

# =====================
# 設定
# =====================
FS = 256
WIN = FS * 3
CH_AF7 = 1
CH_AF8 = 2
EPS = 1e-6

# =====================
# フィルタ
# =====================
def bandpass(data, low, high):
    nyq = FS / 2
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)

def normalize(x, xmin, xmax):
    x = max(xmin, min(x, xmax))
    return int((x - xmin) / (xmax - xmin) * 100)

# =====================
# メイン
# =====================
def main():
    print("EEG stream 探索中...")
    streams = resolve_streams()
    eeg_streams = [s for s in streams if s.type() == "EEG"]
    inlet = StreamInlet(eeg_streams[0])

    buf = np.zeros(WIN)
    print("ベースライン取得完了")
    start_time = time.time()

    csv_file = open("eeg_sample_entropy.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow([
        "timestamp",
        "SampEn_theta",
        "SampEn_alpha",
        "SampEn_beta",
        "CC",
        "RC",
        "SC"
    ])

    fig, ax = plt.subplots()
    ax.set_ylim(0, 100)
    history = {"CC": [], "RC": [], "SC": []}

    def update(frame):
        sample, _ = inlet.pull_sample(timeout=0.0)
        if sample is None:
            return

        eeg = (sample[CH_AF7] + sample[CH_AF8]) / 2
        buf[:-1] = buf[1:]
        buf[-1] = eeg

        theta = bandpass(buf, 4, 7)
        alpha = bandpass(buf, 8, 13)
        beta  = bandpass(buf, 13, 30)

        se_theta = sample_entropy(theta)
        se_alpha = sample_entropy(alpha)
        se_beta  = sample_entropy(beta)

        cc = normalize(se_beta, 0.2, 1.5)
        rc = normalize(1 - se_alpha, 0.0, 1.0)
        sc = normalize(se_theta, 0.2, 1.5)

        for k, v in zip(["CC", "RC", "SC"], [cc, rc, sc]):
            history[k].append(v)
            if len(history[k]) > 200:
                history[k].pop(0)

        ax.clear()
        ax.plot(history["CC"], label="CC")
        ax.plot(history["RC"], label="RC")
        ax.plot(history["SC"], label="SC")
        ax.legend()
        ax.set_ylim(0, 100)

        writer.writerow([
            time.time(),
            se_theta,
            se_alpha,
            se_beta,
            cc,
            rc,
            sc
        ])
        csv_file.flush()

    ani = FuncAnimation(fig, update, interval=200)
    plt.show()
    csv_file.close()

if __name__ == "__main__":
    main()
