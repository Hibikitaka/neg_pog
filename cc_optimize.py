import pandas as pd
import numpy as np
import itertools

EPS = 1e-6
THRESHOLD = 1.769820  # 現行CCとの差分基準

def CC(alpha, beta, a, b):
    val = (beta / a) * (1 + 1 / (alpha + EPS)) * (100 / b)
    val[val > 100] = np.nan
    val[val < 0] = np.nan
    return val


# CSV読み込み
df = pd.read_csv("alpha_beta.csv")
df.columns = df.columns.str.strip().str.lower()
alpha_values = df["alpha"].values
beta_values = df["beta"].values

# 元のパラメータ
a_orig, b_orig = 2.0, 2.0

# 探索範囲 ±1, 0.1刻み
a_range = np.arange(a_orig - 1, a_orig + 1.01, 0.1)
b_range = np.arange(b_orig - 1, b_orig + 1.01, 0.1)

# ベースライン（NaN除外）
baseline = np.nanmean(CC(alpha_values, beta_values, a_orig, b_orig))

# ΔCC 探索
best_combis = []

for a, b in itertools.product(a_range, b_range):
    cc_values = CC(alpha_values, beta_values, a, b)
    delta_cc = np.nanmean(cc_values) - baseline

    if delta_cc >= THRESHOLD:
        best_combis.append((a, b, delta_cc))

# ソート
best_combis_sorted = sorted(best_combis, key=lambda x: x[2], reverse=True)

# 表示
print(f"ΔCC >= {THRESHOLD:.3f} の組み合わせ（上位50件）")
for a, b, dcc in best_combis_sorted[:50]:
    print(f"a={a:.1f}, b={b:.1f}, ΔCC={dcc:.3f}")

