import pandas as pd
import numpy as np
import itertools

EPS = 1e-6

def CC(alpha, beta, a, b):
    val = (beta / a) * (1 + 1 / (alpha + EPS)) * (100 / b)
    return np.clip(val, 0, 100)

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

# ベースライン
baseline = np.mean(CC(alpha_values, beta_values, a_orig, b_orig))

# ΔCC >= 10 となる組み合わせ探索
best_combis = []

for a, b in itertools.product(a_range, b_range):
    cc_values = CC(alpha_values, beta_values, a, b)
    delta_cc = np.mean(cc_values) - baseline
    if delta_cc >= 1.769820:
        best_combis.append((a, b, delta_cc))

# 結果を ΔCC の大きい順にソート
best_combis_sorted = sorted(best_combis, key=lambda x: x[2], reverse=True)

# 上位10件を表示
print("ΔCC >= 10 の組み合わせ（上位10件）")
for combi in best_combis_sorted[:10]:
    print(f"a={combi[0]:.1f}, b={combi[1]:.1f}, ΔCC={combi[2]:.2f}")
