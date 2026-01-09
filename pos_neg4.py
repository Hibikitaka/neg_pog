import random
import time
import csv
from datetime import datetime

CSV_PATH = "eeg_log.csv"

def get_eeg_state():
    """
    実際は muselsl / pylsl の値に置き換える
    今はテスト用ダミー
    """
    cc = random.uniform(40, 100)   # 集中
    rc = random.uniform(0, 40)     # リラックス
    sc = random.uniform(20, 100)   # ストレス

    return {
        "CC": cc,
        "RC": rc,
        "SC": sc
    }

def save_eeg(state):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            state["CC"],
            state["RC"],
            state["SC"]
        ])

if __name__ == "__main__":
    print("EEG テスト開始（Ctrl+Cで終了）")
    while True:
        s = get_eeg_state()
        save_eeg(s)
        print(s)
        time.sleep(1)

