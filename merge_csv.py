# merge_csv.py
import pandas as pd
import glob
import os

# CSVが置いてあるフォルダ（必要なら変更）
CSV_DIR = "./csv_data"

# 出力ファイル名
OUTPUT_FILE = "eeg_cc_rc_sc_all_sessions2.csv"

# 対象CSVを取得
csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "eeg_ccrcsc_measurement2*.csv")))

if not csv_files:
    raise RuntimeError("対象となるCSVファイルが見つかりません")

dfs = []

for file in csv_files:
    df = pd.read_csv(file)

    # セッション識別子としてファイル名を追加
    df["session"] = os.path.basename(file).replace(".csv", "")

    dfs.append(df)

# 縦に結合
merged_df = pd.concat(dfs, ignore_index=True)

# 保存
merged_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"統合完了: {OUTPUT_FILE}")
print(f"結合ファイル数: {len(csv_files)}")
print(f"総行数: {len(merged_df)}")
