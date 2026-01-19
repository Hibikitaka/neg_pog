# sample_entropy.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, filtfilt
from pylsl import StreamInlet, resolve_streams
import time
import csv
import math

# =====================
# Sample Entropy
# =====================
def sample_entropy(signal, m=2, r=0.2):
    signal = np.asarray(signal)
    N = len(signal)
    if N < m + 2:
        return 0, 0, 0, 0.0

    r *= np.std(signal)

    def _count(mm):
        x = np.array([signal[i:i+mm] for i in range(N - mm + 1)])
        cnt = 0
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if np.max(np.abs(x[i] - x[j])) <= r:
                    cnt += 1
        return cnt

    CC_raw = _count(m)
    RC_raw = _count(m + 1)
    SC_raw = RC_raw / CC_raw if CC_raw > 0 else 0.0
    SampEn = -np.log(SC_raw) if SC_raw > 0 else 0.0

    return CC_raw, RC_raw, SC_raw, SampEn

# =====================
# 設定
# =====================
FS = 256
WIN = FS * 3
CH_AF7 = 1
CH_AF8 = 2

SE_MIN = 0.005
SE_MAX = 0.05
LOGSC_MIN = -0.05
LOGSC_MAX = 0.0

# =====================
# フィルタ
# =====================
def bandpass(data, low, high):
    nyq = FS / 2
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)

def normalize(x, xmin, xmax):
    x = max(xmin, min(x, xmax))
    return 100 * (x - xmin) / (xmax - xmin)

# =====================
# Δ算出（10秒移動平均）
# =====================
AVG_WINDOW_SEC = 10
UPDATE_HZ = 5          # interval=200ms → 約5Hz
AVG_SAMPLES = AVG_WINDOW_SEC * UPDATE_HZ

baseline_buf = {"CC": [], "RC": [], "SC": []}
avg_buf = {"CC": [], "RC": [], "SC": []}
baseline_val = {"CC": None, "RC": None, "SC": None}

def update_delta(cc, rc, sc, elapsed):
    avg_out = {"CC_avg": None, "RC_avg": None, "SC_avg": None}
    delta_out = {"ΔCC": None, "ΔRC": None, "ΔSC": None}

    if elapsed < AVG_WINDOW_SEC:
        baseline_buf["CC"].append(cc)
        baseline_buf["RC"].append(rc)
        baseline_buf["SC"].append(sc)
        return avg_out, delta_out

    if baseline_val["CC"] is None:
        for k in baseline_val:
            baseline_val[k] = np.mean(baseline_buf[k])

    for k, v in zip(["CC", "RC", "SC"], [cc, rc, sc]):
        avg_buf[k].append(v)
        if len(avg_buf[k]) > AVG_SAMPLES:
            avg_buf[k].pop(0)

        if len(avg_buf[k]) == AVG_SAMPLES:
            avg = np.mean(avg_buf[k])
            avg_out[f"{k}_avg"] = avg
            delta_out[f"Δ{k}"] = avg - baseline_val[k]

    return avg_out, delta_out

# =====================
# メイン
# =====================
def main():
    print("EEG stream 探索中...")
    inlet = StreamInlet([s for s in resolve_streams() if s.type() == "EEG"][0])
    print("ストリーム接続完了")

    buf = np.zeros(WIN)
    start_time = time.time()

    # ---------- CSV ----------
    f = open("eeg_sample_entropy.csv", "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow([
        "timestamp", "elapsed",
        "CC", "RC", "SC",
        "CC_avg", "RC_avg", "SC_avg",
        "ΔCC", "ΔRC", "ΔSC"
    ])

    # ---------- 描画 ----------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_ylim(0, 100)

    lines = {k: ax.plot([], [], label=k)[0] for k in ["CC", "RC", "SC"]}
    ax.legend()

    hist = {k: [] for k in lines}

    def update(_):
        sample, _ = inlet.pull_sample(timeout=0.0)
        if sample is None:
            return lines.values()

        eeg = (sample[CH_AF7] + sample[CH_AF8]) / 2
        buf[:-1] = buf[1:]
        buf[-1] = eeg

        beta = bandpass(buf, 13, 30)
        _, _, sc_raw, se = sample_entropy(beta)

        CC = normalize(se, SE_MIN, SE_MAX)
        RC = normalize(1.0 - se, 1.0 - SE_MAX, 1.0 - SE_MIN)
        SC = normalize(math.log(sc_raw + 1e-12), LOGSC_MIN, LOGSC_MAX)

        elapsed = time.time() - start_time
        avg, delta = update_delta(CC, RC, SC, elapsed)

        for k, v in zip(["CC", "RC", "SC"], [CC, RC, SC]):
            hist[k].append(v)
            if len(hist[k]) > 300:
                hist[k].pop(0)
            lines[k].set_data(range(len(hist[k])), hist[k])

        ax.set_xlim(0, len(hist["CC"]))

        w.writerow([
            time.time(), elapsed,
            CC, RC, SC,
            avg["CC_avg"], avg["RC_avg"], avg["SC_avg"],
            delta["ΔCC"], delta["ΔRC"], delta["ΔSC"]
        ])
        f.flush()

        return lines.values()

    
    ani = FuncAnimation(fig, update, interval=200, cache_frame_data=False)
    plt.show()
    f.close()

if __name__ == "__main__":
    main()


