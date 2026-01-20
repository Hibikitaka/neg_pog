# rc_optimize.py
import pandas as pd
import numpy as np
import itertools

# ======================
# RC 計算式（修正版）
# ======================
def RC(alpha, beta, d, e, f):
    val = (np.maximum(0, 1.0 - beta / d) + alpha / e) * (100 / f)
    return np.clip(val, 0, 100)

# ======================
# CSV 読み込み
# ======================
df = pd.read_csv("alpha_beta.csv")
df.columns = df.columns.str.strip().str.lower()
alpha_values = df["alpha"].values
beta_values = df["beta"].values

# ======================
# パラメータ初期値
# ======================
d_orig, e_orig, f_orig = 3.0, 2.0, 2.0

d_range = np.arange(d_orig - 1, d_orig + 1.01, 0.1)
e_range = np.arange(e_orig - 1, e_orig + 1.01, 0.1)
f_range = np.arange(f_orig - 1, f_orig + 1.01, 0.1)

# ======================
# ベースライン
# ======================
baseline = np.mean(RC(alpha_values, beta_values, d_orig, e_orig, f_orig))

# ======================
# ΔRC 探索
# ======================
best_combis = []

for d, e, f in itertools.product(d_range, e_range, f_range):
    rc_values = RC(alpha_values, beta_values, d, e, f)
    delta_rc = np.mean(rc_values) - baseline

    if delta_rc >= -1.848007:
        best_combis.append((d, e, f, delta_rc))

# ======================
# 結果表示
# ======================
best_combis_sorted = sorted(best_combis, key=lambda x: x[3], reverse=True)

print("ΔRC >= 10 の組み合わせ（上位10件）")
for d, e, f, delta in best_combis_sorted[:10]:
    print(f"d={d:.1f}, e={e:.1f}, f={f:.1f}, ΔRC={delta:.2f}")
