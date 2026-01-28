# eeg_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# =====================
# 日本語フォント設定（Windows）
# =====================
font_path = "C:/Windows/Fonts/meiryo.ttc"
font_prop = font_manager.FontProperties(fname=font_path)
rcParams["font.family"] = font_prop.get_name()

# =====================
# 設定
# =====================
FILES = {
    "CRS": "eeg_cc_rc_sc.csv",
    "改良CRS": "eeg_cc_rc_sc_all_sessions.csv",
    "EI": "eeg_engagement_Index.csv",
    "FAA": "eeg_faa.csv",
    "SampEn": "eeg_sample_entropy.csv", 
    
}

DELTA_COLS = {
    "CC": "ΔCC",
    "RC": "ΔRC",
    "SC": "ΔSC"
}

# =====================
# CSV 読み込み
# =====================
def load_data(path):
    df = pd.read_csv(path)

    # ---- 時間軸の統一 ----
    if "elapsed" in df.columns:
        df = df.sort_values("elapsed")
        df["time"] = df["elapsed"]

    elif "timestamp" in df.columns:
        df = df.sort_values("timestamp")
        t0 = df["timestamp"].iloc[0]
        df["time"] = df["timestamp"] - t0

    else:
        raise ValueError(f"{path} に elapsed も timestamp も存在しません")

    return df


# =====================
# Δ平均集計（従来通り）
# =====================
summary = []

for method, path in FILES.items():
    df = load_data(path)

    for metric, dcol in DELTA_COLS.items():
        if dcol not in df.columns:
            continue

        mean_delta = df[dcol].dropna().mean()

        summary.append([method, metric, mean_delta])

summary_df = pd.DataFrame(
    summary,
    columns=["Method", "Metric", "Delta"]
)

print(summary_df)

# =====================
# Δ比較表（Method × Metric）
# =====================
delta_table = summary_df.pivot(
    index="Metric",
    columns="Method",
    values="Delta"
)

print("\nΔ値比較表")
print(delta_table)

# =====================
# 可視化：Δ平均 棒グラフ
# =====================
METHOD_ORDER = ["EI", "FAA", "SampEn", "CRS", "改良CRS"]

delta_table = delta_table[METHOD_ORDER]

delta_table.plot(kind="bar", figsize=(10, 5))

plt.ylabel("Δ Value（測定時の値 − 平常時の値）")
plt.legend(
    title="手法",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)
plt.grid(True)
plt.tight_layout()
plt.show()



# =====================
# 可視化：ヒートマップ
# =====================
plt.figure(figsize=(8, 4))
plt.imshow(delta_table.values, aspect="auto")
plt.xticks(range(len(delta_table.columns)), delta_table.columns)
plt.yticks(range(len(delta_table.index)), delta_table.index)
plt.colorbar(label="Δ Value")
plt.title("Method × Metric Heatmap")
plt.tight_layout()
plt.show()

# =====================
# 可視化：ΔValue 時系列
# =====================
for method, path in FILES.items():
    df = load_data(path)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"{method} : ΔValue Time Series")

    for ax, (metric, dcol) in zip(axes, DELTA_COLS.items()):
        if dcol not in df.columns:
            ax.set_visible(False)
            continue

        ax.plot(df["time"], df[dcol], linewidth=1)
        ax.set_ylabel(f"Δ{metric}")
        ax.grid(True)

    axes[-1].set_xlabel("Elapsed Time [sec]")
    plt.tight_layout()
    plt.show()

    


