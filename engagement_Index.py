# engagement_Index.py
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
# 定数
# =====================
FS = 256
WINDOW_SEC = 4
N = FS * WINDOW_SEC
EPS = 1e-6

CH_AF7 = 1
CH_AF8 = 2

# =====================
# 最新EEG状態
# =====================
LATEST_STATE = {"CC": 0.0, "RC": 0.0, "SC": 0.0}

def save_state():
    with open("eeg_state.json", "w", encoding="utf-8") as f:
        json.dump(LATEST_STATE, f, ensure_ascii=False)

# =====================
# 日本語フォント
# =====================
font_path = "C:/Windows/Fonts/meiryo.ttc"
font_prop = font_manager.FontProperties(fname=font_path)
rcParams["font.family"] = font_prop.get_name()

# =====================
# 信号処理
# =====================
def bandpass(data, low, high, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)

def smooth_envelope(env, fs, cutoff=1.5):
    nyq = fs / 2
    b, a = butter(2, cutoff / nyq, btype="low")
    return filtfilt(b, a, env)

def band_envelope(data, band, fs):
    return smooth_envelope(
        np.abs(hilbert(bandpass(data, band[0], band[1], fs))),
        fs
    )

# =====================
# EIベース指標
# =====================
def normalize(x, xmin, xmax):
    x = max(xmin, min(x, xmax))
    return int((x - xmin) / (xmax - xmin) * 100)

def CC(alpha_r, beta_r, theta_r):
    return normalize(beta_r / (alpha_r + theta_r + EPS), 0.3, 1.0)

def RC(alpha_r, beta_r, theta_r):
    return normalize(alpha_r / (beta_r + theta_r + EPS), 0.3, 1.2)

def SC(alpha_r, beta_r):
    return normalize(beta_r / (alpha_r + EPS), 0.6, 1.8)

# =====================
# Δ算出（10秒平均）
# =====================
AVG_WINDOW_SEC = 10
UPDATE_HZ = 20
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
    print("接続完了")

    eeg_buf = np.zeros(N)

    # --- ベースライン（周波数） ---
    theta_b, alpha_b, beta_b = [], [], []
    start = time.time()

    while time.time() - start < 10:
        sample, _ = inlet.pull_sample()
        eeg = (sample[CH_AF7] + sample[CH_AF8]) / 2
        eeg_buf = np.roll(eeg_buf, -1)
        eeg_buf[-1] = eeg

        theta_b.append(np.mean(band_envelope(eeg_buf, (4, 7), FS)))
        alpha_b.append(np.mean(band_envelope(eeg_buf, (8, 13), FS)))
        beta_b.append(np.mean(band_envelope(eeg_buf, (13, 30), FS)))

    bt, ba, bb = map(lambda x: max(np.mean(x), EPS), [theta_b, alpha_b, beta_b])
    start_time = time.time()

    # --- CSV ---
    f = open("eeg_engagement_Index.csv", "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow([
        "timestamp", "elapsed",
        "CC", "RC", "SC",
        "CC_avg", "RC_avg", "SC_avg",
        "ΔCC", "ΔRC", "ΔSC"
    ])

    # --- 描画 ---
    fig, ax = plt.subplots()
    ax.set_ylim(0, 100)
    lines = {k: ax.plot([], [], label=k)[0] for k in ["CC", "RC", "SC"]}
    ax.legend()
    hist = {k: [] for k in lines}

    def update(_):
        sample, _ = inlet.pull_sample(timeout=0.0)
        if sample is None:
            return lines.values()

        eeg = (sample[CH_AF7] + sample[CH_AF8]) / 2
        eeg_buf[:-1] = eeg_buf[1:]
        eeg_buf[-1] = eeg

        theta = np.mean(band_envelope(eeg_buf, (4, 7), FS)[-FS*3:])
        alpha = np.mean(band_envelope(eeg_buf, (8, 13), FS)[-FS*3:])
        beta  = np.mean(band_envelope(eeg_buf, (13, 30), FS)[-FS*3:])

        cc = CC(alpha/ba, beta/bb, theta/bt)
        rc = RC(alpha/ba, beta/bb, theta/bt)
        sc = SC(alpha/ba, beta/bb)

        elapsed = time.time() - start_time
        avg, delta = update_delta(cc, rc, sc, elapsed)

        LATEST_STATE.update({"CC": cc, "RC": rc, "SC": sc})
        save_state()

        for k, v in zip(["CC", "RC", "SC"], [cc, rc, sc]):
            hist[k].append(v)
            if len(hist[k]) > 300:
                hist[k].pop(0)
            lines[k].set_data(range(len(hist[k])), hist[k])

        ax.set_xlim(0, len(hist["CC"]))

        w.writerow([
            time.time(), elapsed,
            cc, rc, sc,
            avg["CC_avg"], avg["RC_avg"], avg["SC_avg"],
            delta["ΔCC"], delta["ΔRC"], delta["ΔSC"]
        ])
        f.flush()

        return lines.values()

    
    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.show()
    f.close()

if __name__ == "__main__":
    main()
