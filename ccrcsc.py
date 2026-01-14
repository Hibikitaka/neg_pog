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
import os

# =====================
# 最新EEG状態
# =====================
LATEST_STATE = {"CC": 0.0, "RC": 0.0, "SC": 0.0}

def get_eeg_state():
    """外部呼び出し用：最新EEG状態を返す"""
    return LATEST_STATE.copy()

def save_state():
    """JSONに最新EEG状態を保存"""
    with open("eeg_state.json", "w", encoding="utf-8") as f:
        json.dump(LATEST_STATE, f, ensure_ascii=False)

# =====================
# 日本語フォント（Windows）
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
# CC / RC / SC 計算
# =====================
def CC(alpha, beta):
    value = (beta / 2) * (1 + 1 / alpha) * 50
    return min(100, math.floor(value))

def RC(alpha, beta):
    value = (max(0, (1.0 - beta / 3)) + alpha / 2) * 50
    return min(100, math.floor(value))

def SC(alpha, beta):
    value = (max(0, (1.0 - alpha / 3) / 5) + ((beta / (2 * alpha)) * 4 / 5)) * 100
    return min(100, math.floor(value))

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
        raise RuntimeError("EEG stream が見つかりません（muselsl を起動してください）")

    inlet = StreamInlet(eeg_streams[0])
    print("EEG stream 接続完了")

    eeg_buf = np.zeros(N)

    # ---------- ベースライン取得 ----------
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

        alpha_env = band_envelope(eeg_buf, (8, 13), FS)
        beta_env  = band_envelope(eeg_buf, (13, 30), FS)

        alpha_base.append(np.mean(alpha_env))
        beta_base.append(np.mean(beta_env))

    baseline_alpha = np.mean(alpha_base)
    baseline_beta  = np.mean(beta_base)
    print("ベースライン取得完了")

    # ---------- CSV準備 ----------
    csv_file = open("eeg_cc_rc_sc.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp", "alpha_ratio", "beta_ratio", "CC", "RC", "SC"])

    # ---------- 描画 ----------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_ylim(0, 100)

    line_cc, = ax.plot([], [], label="集中 CC", color="red")
    line_rc, = ax.plot([], [], label="リラックス RC", color="green")
    line_sc, = ax.plot([], [], label="ストレス SC", color="blue")
    ax.legend()

    history = {"CC": [], "RC": [], "SC": []}
    state_text = ax.text(0.02, 0.85, "", transform=ax.transAxes, fontsize=14)

    # ---------- 更新関数 ----------
    def update(frame):
        sample, _ = inlet.pull_sample(timeout=0.0)
        if sample is None:
            return

        eeg = (sample[CH_AF7] + sample[CH_AF8]) / 2
        eeg_buf[:] = np.roll(eeg_buf, -1)
        eeg_buf[-1] = eeg

        alpha = np.mean(band_envelope(eeg_buf, (8, 13), FS)[-FS*3:])
        beta  = np.mean(band_envelope(eeg_buf, (13, 30), FS)[-FS*3:])

        alpha_ratio = alpha / baseline_alpha
        beta_ratio  = beta  / baseline_beta

        cc = CC(alpha_ratio, beta_ratio)
        rc = RC(alpha_ratio, beta_ratio)
        sc = SC(alpha_ratio, beta_ratio)

        # 最新EEG状態更新
        LATEST_STATE["CC"] = cc
        LATEST_STATE["RC"] = rc
        LATEST_STATE["SC"] = sc
        save_state()  # JSON更新

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

        writer.writerow([time.time(), alpha_ratio, beta_ratio, cc, rc, sc])
        csv_file.flush()

        return line_cc, line_rc, line_sc, state_text

    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

    csv_file.close()


if __name__ == "__main__":
    main()



