from pylsl import StreamInlet, resolve_streams
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
from scipy.signal import butter, filtfilt
import csv

# -----------------------
# 設定
# -----------------------
FS = 256
WINDOW_SEC = 5
BUFFER_SIZE = FS * WINDOW_SEC
CHANNEL = 1  # AF7（0=TP9, 1=AF7, 2=AF8, 3=TP10）

# α波とβ波のみ
BANDS = {
    "Alpha (8-13Hz)": (8, 13),
    "Beta (13-30Hz)": (13, 30),
}

CSV_FILE = "alpha_beta.csv"

# -----------------------
# バンドパスフィルタ
# -----------------------
def bandpass(data, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, data)

# -----------------------
# EEG stream を待つ
# -----------------------
print("EEG stream を待っています...")
while True:
    streams = resolve_streams()
    eeg_streams = [s for s in streams if s.type() == "EEG"]
    if eeg_streams:
        break
    time.sleep(0.5)

inlet = StreamInlet(eeg_streams[0])
print("EEG stream 接続完了")

# -----------------------
# バッファ
# -----------------------
buffer = deque(maxlen=BUFFER_SIZE)
time_buffer = deque(maxlen=BUFFER_SIZE)
start_time = time.time()

# -----------------------
# matplotlib
# -----------------------
plt.ion()
fig, axes = plt.subplots(len(BANDS), 1, sharex=True, figsize=(10, 6))
lines = []

for ax, name in zip(axes, BANDS.keys()):
    line, = ax.plot([], [])
    ax.set_ylabel(name)
    ax.set_ylim(-50, 50)
    lines.append(line)

axes[-1].set_xlabel("Time (s)")
fig.suptitle("EEG Band-pass Filtered Waves (AF7)")

# -----------------------
# CSV準備
# -----------------------
with open(CSV_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Time (s)"] + list(BANDS.keys()))

# -----------------------
# メインループ
# -----------------------
try:
    while True:
        sample, ts = inlet.pull_sample(timeout=1.0)
        if sample:
            buffer.append(sample[CHANNEL])
            t = time.time() - start_time
            time_buffer.append(t)

            if len(buffer) == BUFFER_SIZE:
                data = np.array(buffer)
                filtered_dict = {}

                for i, ((low, high), line) in enumerate(zip(BANDS.values(), lines)):
                    filtered = bandpass(data, FS, low, high)
                    line.set_data(time_buffer, filtered)
                    filtered_dict[list(BANDS.keys())[i]] = filtered[-1]  # 最新値だけCSVに書き込む

                axes[0].set_xlim(t - WINDOW_SEC, t)
                plt.pause(0.01)

                # CSVに追記
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    row = [t] + [filtered_dict[name] for name in BANDS.keys()]
                    writer.writerow(row)

except KeyboardInterrupt:
    print("Stopped")

plt.ioff()
plt.show()


