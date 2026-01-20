# sc_optimize.py
import pandas as pd
import numpy as np
import itertools

EPS = 1e-6

# ======================
# SC 計算式（kなし）
# ======================
def SC(alpha, beta, g, h):
    val = (
        np.maximum(0, (1.0 - alpha / g) / 5.0)
        + (beta / (h * alpha + EPS)) * 4.0 / 5.0
    ) * 100.0
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
g_orig, h_orig = 2.0, 2.0

g_range = np.arange(g_orig - 1, g_orig + 1.01, 0.1)
h_range = np.arange(h_orig - 1, h_orig + 1.01, 0.1)

# ======================
# ベースライン
# ======================
baseline = np.mean(SC(alpha_values, beta_values, g_orig, h_orig))

# ======================
# ΔSC 探索
# ======================
best_combis = []

for g, h in itertools.product(g_range, h_range):
    sc_values = SC(alpha_values, beta_values, g, h)
    delta_sc = np.mean(sc_values) - baseline

    if delta_sc >= 3.5:
        best_combis.append((g, h, delta_sc))

# ======================
# 結果表示
# ======================
best_combis_sorted = sorted(best_combis, key=lambda x: x[2], reverse=True)

print("ΔSC >= 10 の組み合わせ（上位10件）")
for g, h, delta in best_combis_sorted[:10]:
    print(f"g={g:.1f}, h={h:.1f}, ΔSC={delta:.2f}")

