# ccrcsc.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import font_manager, rcParams
from scipy.signal import butter, filtfilt, hilbert
from pylsl import StreamInlet, resolve_streams
import time
import csv
import math
import json

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

def get_eeg_state():
    return LATEST_STATE.copy()

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
    filtered = bandpass(data, band[0], band[1], fs)
    env = np.abs(hilbert(filtered))
    return smooth_envelope(env, fs)

# =====================
# CC / RC / SC 指標
# =====================
def CC(alpha, beta):
    val = (beta / 2) * (1 + 1 / (alpha + EPS)) * 50
    return int(np.clip(val, 0, 100))

def RC(alpha, beta):
    
    val = (max(0, (1.0 - beta / 3)) + alpha / 2) * 50
    return int(np.clip(val, 0, 100))

def SC(alpha, beta):
    val = (
        max(0, (1.0 - alpha / 3) / 5)
        + (beta / (2 * alpha + EPS)) * 4 / 5
    ) * 100
    return int(np.clip(val, 0, 100))

# =====================
# Δ算出（10秒平均）
# =====================
AVG_WINDOW_SEC = 10
UPDATE_HZ = 20
AVG_SAMPLES = AVG_WINDOW_SEC * UPDATE_HZ

baseline_buffer = {"CC": [], "RC": [], "SC": []}
avg_buffer = {"CC": [], "RC": [], "SC": []}
baseline_value = {"CC": None, "RC": None, "SC": None}

def update_delta(cc, rc, sc, elapsed_sec):
    avg = {"CC": None, "RC": None, "SC": None}
    delta = {"CC": None, "RC": None, "SC": None}

    if elapsed_sec < AVG_WINDOW_SEC:
        baseline_buffer["CC"].append(cc)
        baseline_buffer["RC"].append(rc)
        baseline_buffer["SC"].append(sc)
        return avg, delta

    if baseline_value["CC"] is None:
        baseline_value["CC"] = np.mean(baseline_buffer["CC"])
        baseline_value["RC"] = np.mean(baseline_buffer["RC"])
        baseline_value["SC"] = np.mean(baseline_buffer["SC"])

    for k, v in zip(["CC", "RC", "SC"], [cc, rc, sc]):
        avg_buffer[k].append(v)
        if len(avg_buffer[k]) > AVG_SAMPLES:
            avg_buffer[k].pop(0)

        if len(avg_buffer[k]) == AVG_SAMPLES:
            avg[k] = np.mean(avg_buffer[k])
            delta[k] = avg[k] - baseline_value[k]

    return avg, delta

# =====================
# メイン処理
# =====================
def main():
    print("EEG stream 探索中...")
    streams = resolve_streams()
    eeg_streams = [s for s in streams if s.type() == "EEG"]
    if not eeg_streams:
        raise RuntimeError("EEG stream が見つかりません")

    inlet = StreamInlet(eeg_streams[0])
    print("EEG stream 接続完了")

    eeg_buf = np.zeros(N)

    # ---------- ベースライン ----------
    print("平常時ベースライン取得中（10秒）...")
    alpha_base, beta_base = [], []
    start = time.time()

    while time.time() - start < 10:
        sample, _ = inlet.pull_sample(timeout=1.0)
        if sample is None:
            continue

        eeg = (sample[CH_AF7] + sample[CH_AF8]) / 2
        eeg_buf = np.roll(eeg_buf, -1)
        eeg_buf[-1] = eeg

        alpha_base.append(np.mean(band_envelope(eeg_buf, (8, 13), FS)))
        beta_base.append(np.mean(band_envelope(eeg_buf, (13, 30), FS)))

    baseline_alpha = max(np.mean(alpha_base), EPS)
    baseline_beta = max(np.mean(beta_base), EPS)

    print("ベースライン取得完了")
    start_time = time.time()

    # ---------- CSV ----------
    csv_file = open("eeg_cc_rc_sc.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow([
        "timestamp", "elapsed_sec",
        "CC", "RC", "SC",
        "CC_avg", "RC_avg", "SC_avg",
        "ΔCC", "ΔRC", "ΔSC"
    ])

    # ---------- 描画 ----------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_ylim(0, 100)
    line_cc, = ax.plot([], [], label="CC")
    line_rc, = ax.plot([], [], label="RC")
    line_sc, = ax.plot([], [], label="SC")
    ax.legend()

    history = {"CC": [], "RC": [], "SC": []}
    text = ax.text(0.02, 0.85, "", transform=ax.transAxes)

    def update(frame):
        sample, _ = inlet.pull_sample(timeout=0.0)
        if sample is None:
            return

        eeg = (sample[CH_AF7] + sample[CH_AF8]) / 2
        eeg_buf[:] = np.roll(eeg_buf, -1)
        eeg_buf[-1] = eeg

        alpha = np.mean(band_envelope(eeg_buf, (8, 13), FS)[-FS*3:])
        beta  = np.mean(band_envelope(eeg_buf, (13, 30), FS)[-FS*3:])

        ar = alpha / baseline_alpha
        br = beta / baseline_beta

        cc = CC(ar, br)
        rc = RC(ar, br)
        sc = SC(ar, br)

        elapsed = time.time() - start_time
        avg, delta = update_delta(cc, rc, sc, elapsed)

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

        text.set_text(f"CC:{cc}  RC:{rc}  SC:{sc}")

        writer.writerow([
            time.time(), elapsed,
            cc, rc, sc,
            avg["CC"], avg["RC"], avg["SC"],
            delta["CC"], delta["RC"], delta["SC"]
        ])
        csv_file.flush()

        return line_cc, line_rc, line_sc, text

    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    csv_file.close()

if __name__ == "__main__":
    main()



