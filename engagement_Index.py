#engagement_Index.py
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
# CC / RC / SC（完全EI版）
# =====================
EPS = 1e-6

def normalize(x, xmin, xmax):
    x = max(xmin, min(x, xmax))
    return min(100, max(0, int((x - xmin) / (xmax - xmin) * 100)))

def CC(alpha, beta, theta):
    # 集中度（Engagement Index）
    ei = beta / (alpha + theta + EPS)
    return normalize(ei, 0.2, 2.5)

def RC(alpha, beta, theta):
    # リラックス度（Alpha dominance）
    rc = alpha / (beta + theta + EPS)
    return normalize(rc, 0.3, 2.0)

def SC(alpha, beta):
    # ストレス度（Beta dominance）
    sc = beta / (alpha + EPS)
    return normalize(sc, 0.3, 3.0)

# =====================
# メイン処理
# =====================
def main():
    global inlet, eeg_buf, writer, csv_file

    # ---------- LSL接続 ----------
    print("EEG stream 探索中...")
    streams = resolve_streams()
    eeg_streams = [s for s in streams if s.type() == "EEG"]

    if not eeg_streams:
        raise RuntimeError("EEG stream が見つかりません")

    inlet = StreamInlet(eeg_streams[0])
    print("EEG stream 接続完了")

    eeg_buf = np.zeros(N)

    # ---------- ベースライン取得 ----------
    print("平常時ベースライン取得中（10秒）...")
    theta_base, alpha_base, beta_base = [], [], []
    start = time.time()

    while time.time() - start < 10:
        sample, _ = inlet.pull_sample(timeout=1.0)
        if sample is None:
            continue

        eeg = (sample[CH_AF7] + sample[CH_AF8]) / 2
        eeg_buf = np.roll(eeg_buf, -1)
        eeg_buf[-1] = eeg

        theta_env = band_envelope(eeg_buf, (4, 7), FS)
        alpha_env = band_envelope(eeg_buf, (8, 13), FS)
        beta_env  = band_envelope(eeg_buf, (13, 30), FS)

        theta_base.append(np.mean(theta_env))
        alpha_base.append(np.mean(alpha_env))
        beta_base.append(np.mean(beta_env))

    baseline_theta = np.mean(theta_base)
    baseline_alpha = np.mean(alpha_base)
    baseline_beta  = np.mean(beta_base)
    print("ベースライン取得完了")
    start_time = time.time()


    # ---------- CSV ----------
    csv_file = open("eeg_engagement_Index.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow([
    "timestamp",
    "elapsed_sec",
    "theta_raw",
    "alpha_raw",
    "beta_raw",
    "theta_ratio",
    "alpha_ratio",
    "beta_ratio",
    "CC",
    "RC",
    "SC"
])


    # ---------- 描画 ----------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_ylim(0, 100)

    line_cc, = ax.plot([], [], label="集中 CC", color="red")
    line_rc, = ax.plot([], [], label="リラックス RC", color="green")
    line_sc, = ax.plot([], [], label="ストレス SC", color="blue")
    ax.legend()

    history = {"CC": [], "RC": [], "SC": []}
    state_text = ax.text(0.02, 0.85, "", transform=ax.transAxes, fontsize=14)

    # ---------- 更新 ----------
    def update(frame):
        sample, _ = inlet.pull_sample(timeout=0.0)
        if sample is None:
            return

        eeg = (sample[CH_AF7] + sample[CH_AF8]) / 2
        eeg_buf[:] = np.roll(eeg_buf, -1)
        eeg_buf[-1] = eeg

        theta = np.mean(band_envelope(eeg_buf, (4, 7), FS)[-FS*3:])
        alpha = np.mean(band_envelope(eeg_buf, (8, 13), FS)[-FS*3:])
        beta  = np.mean(band_envelope(eeg_buf, (13, 30), FS)[-FS*3:])

        theta_r = theta / baseline_theta
        alpha_r = alpha / baseline_alpha
        beta_r  = beta  / baseline_beta

        cc = CC(alpha_r, beta_r, theta_r)
        rc = RC(alpha_r, beta_r, theta_r)
        sc = SC(alpha_r, beta_r)

        LATEST_STATE["CC"] = cc
        LATEST_STATE["RC"] = rc
        LATEST_STATE["SC"] = sc
        save_state()

        for k, v in zip(["CC", "RC", "SC"], [cc, rc, sc]):
            history[k].append(v)
            if len(history[k]) > 300:
                history[k].pop(0)

        line_cc.set_data(range(len(history["CC"])), history["CC"])
        line_rc.set_data(range(len(history["RC"])), history["RC"])
        line_sc.set_data(range(len(history["SC"])), history["SC"])
        ax.set_xlim(0, len(history["CC"]))

        state_text.set_text(
            f"集中 CC : {cc}\n"
            f"リラックス RC : {rc}\n"
            f"ストレス SC : {sc}"
        )

        writer.writerow([time.time(), theta_r, alpha_r, beta_r, cc, rc, sc])
        csv_file.flush()

        return line_cc, line_rc, line_sc, state_text

    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    csv_file.close()

if __name__ == "__main__":
    main()
