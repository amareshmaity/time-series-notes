"""
code/01_granger_causality.py
==============================
Module 12 — Multivariate & Advanced Topics
Practical: Granger causality testing and causal discovery.

Demonstrates:
  - Bivariate Granger test with statsmodels
  - VAR-based multivariate Granger test
  - Granger causality matrix (D×D) with heatmap
  - Stationarity preprocessing
  - Transfer entropy (nonlinear Granger proxy)
  - PCMCI placeholder
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.var_model import VAR

# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate Synthetic Causal System
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
N = 500

# True causal structure:
#   X1 → X2 (lag 2, coeff 0.6)
#   X2 → X3 (lag 1, coeff 0.5)
#   X4: independent (no causal links)
#   X1 ↔ X5: bidirectional (X1→X5 lag 1, X5→X1 lag 3)

noise_std = 0.5

x1 = np.random.normal(0, 1, N)
x5 = np.zeros(N)
x5[0:3] = np.random.normal(0, 1, 3)

for t in range(3, N):
    x5[t] = 0.4 * x1[t-1] + np.random.normal(0, noise_std)

x2 = np.zeros(N)
for t in range(2, N):
    x2[t] = 0.6 * x1[t-2] + np.random.normal(0, noise_std)

x3 = np.zeros(N)
for t in range(1, N):
    x3[t] = 0.5 * x2[t-1] + np.random.normal(0, noise_std)

x4 = np.random.normal(0, 1, N)   # completely independent

df_raw = pd.DataFrame({
    "X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5
})

print("Synthetic causal system (N=500):")
print("True links: X1→X2(lag2), X2→X3(lag1), X1→X5(lag1)")
print("No links:   X4 (independent)")
print(f"Shape: {df_raw.shape}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stationarity Check
# ─────────────────────────────────────────────────────────────────────────────

print("ADF stationarity tests (p-value):")
df_stat = df_raw.copy()
for col in df_raw.columns:
    p = adfuller(df_raw[col])[1]
    status = "✅ stationary" if p < 0.05 else "⚠️ non-stationary → differencing"
    print(f"  {col}: p={p:.4f}  {status}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bivariate Granger Tests
# ─────────────────────────────────────────────────────────────────────────────

def granger_pair(x, y, max_lag=5, alpha=0.05):
    """Test if x Granger-causes y. Returns min p-value and best lag."""
    data    = np.column_stack([y, x])
    results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    p_vals  = {lag: results[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag+1)}
    best_lag = min(p_vals, key=p_vals.get)
    min_p    = p_vals[best_lag]
    return {"p_min": round(min_p, 5), "best_lag": best_lag, "causal": min_p < alpha}


print("\nBivariate Granger tests (max_lag=5):")
pairs_to_test = [("X1","X2"), ("X2","X3"), ("X1","X3"), ("X4","X1"), ("X1","X5"), ("X5","X1")]
for cause, effect in pairs_to_test:
    r = granger_pair(df_stat[cause].values, df_stat[effect].values, max_lag=5)
    mark = "✓ CAUSAL" if r["causal"] else "  no link"
    print(f"  {cause} → {effect}: p={r['p_min']:.5f} (lag={r['best_lag']})  {mark}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full Causality Matrix
# ─────────────────────────────────────────────────────────────────────────────

cols  = df_stat.columns.tolist()
D     = len(cols)
pmat  = pd.DataFrame(np.ones((D, D)), index=cols, columns=cols)

print(f"\nComputing {D}×{D} pairwise Granger matrix...")
for cause in cols:
    for effect in cols:
        if cause == effect:
            pmat.loc[cause, effect] = np.nan
            continue
        try:
            r = granger_pair(df_stat[cause].values, df_stat[effect].values)
            pmat.loc[cause, effect] = r["p_min"]
        except Exception:
            pass

print("\nP-value matrix (row → column):")
print(pmat.round(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 5. VAR-Based Multivariate Granger Test
# ─────────────────────────────────────────────────────────────────────────────

print("\nVAR-based multivariate Granger tests:")
model_sel = VAR(df_stat)
sel       = model_sel.select_order(maxlags=6)
best_lag  = max(1, sel.bic)
best_lag  = min(int(best_lag), 5)

var_model   = VAR(df_stat)
var_results = var_model.fit(best_lag, ic=None)

var_rows = []
for cause in cols:
    for effect in cols:
        if cause == effect:
            continue
        try:
            test = var_results.test_causality(effect, [cause], kind="f")
            var_rows.append({
                "cause": cause, "effect": effect,
                "F":     round(test.test_statistic, 3),
                "p":     round(test.pvalue, 5),
                "sig":   test.pvalue < 0.05,
            })
        except Exception:
            pass

var_df = pd.DataFrame(var_rows)
sig    = var_df[var_df["sig"]].sort_values("p")
print(f"Significant links (VAR, p<0.05):")
print(sig[["cause","effect","F","p"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Transfer Entropy (Nonlinear)
# ─────────────────────────────────────────────────────────────────────────────

def transfer_entropy_binning(x, y, lag=1, n_bins=8, eps=1e-10):
    """
    Estimate TE(X→Y) = H(Y_t|Y_{t-1}) - H(Y_t|Y_{t-1}, X_{t-1}) via binning.
    Positive TE → X provides information about future Y beyond Y's own past.
    """
    n    = len(x) - lag
    xt_l = x[:-lag]; yt   = y[lag:]
    yt_l = y[:-lag]

    # Bin all variables
    def digitize(a):
        edges = np.percentile(a, np.linspace(0, 100, n_bins+1))
        edges[0] -= 1e-10; edges[-1] += 1e-10
        return np.digitize(a, edges) - 1

    xbl = digitize(xt_l); yb  = digitize(yt); ybl = digitize(yt_l)

    # H(Y_t, Y_{t-1})
    p_yyt = np.zeros((n_bins, n_bins))
    for i in range(len(yb)):
        p_yyt[yb[i], ybl[i]] += 1
    p_yyt /= (p_yyt.sum() + eps)

    # H(Y_t, Y_{t-1}, X_{t-1})
    p_xyyt = np.zeros((n_bins, n_bins, n_bins))
    for i in range(len(yb)):
        p_xyyt[yb[i], ybl[i], xbl[i]] += 1
    p_xyyt /= (p_xyyt.sum() + eps)

    def cond_ent_2(p2):
        """H(Y|X) from 2D joint p(Y,X)."""
        px   = p2.sum(axis=0) + eps
        term = np.where(p2 > eps, p2 * np.log(p2 / px), 0.0)
        return -term.sum()

    def cond_ent_3(p3):
        """H(Y|X,Z) from 3D joint p(Y,X,Z)."""
        pxz  = p3.sum(axis=0) + eps
        term = np.where(p3 > eps, p3 * np.log(p3 / pxz), 0.0)
        return -term.sum()

    h_y_given_yl  = cond_ent_2(p_yyt)
    h_y_given_yxl = cond_ent_3(p_xyyt)
    return max(0.0, h_y_given_yl - h_y_given_yxl)


print("\nTransfer Entropy (nonlinear Granger proxy, lag=2):")
te_pairs = [("X1","X2"), ("X2","X3"), ("X4","X1"), ("X5","X1"), ("X1","X5")]
for cause, effect in te_pairs:
    te = transfer_entropy_binning(df_stat[cause].values, df_stat[effect].values, lag=2)
    print(f"  TE({cause}→{effect}) = {te:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualization
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Time series plot
ax = axes[0]
colors = plt.cm.tab10(np.linspace(0, 0.5, D))
for i, col in enumerate(cols):
    ax.plot(df_raw[col].values[:150], label=col, color=colors[i], linewidth=1.0, alpha=0.8)
ax.set_title("Synthetic Causal System (first 150 steps)", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 2: P-value heatmap
ax = axes[1]
data = pmat.values.astype(float)
im   = ax.imshow(data, vmin=0, vmax=0.1, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(D)); ax.set_xticklabels(cols, rotation=30, ha="right")
ax.set_yticks(range(D)); ax.set_yticklabels(cols)
ax.set_title("Bivariate Granger P-Values\n(row → column)", fontsize=11)
ax.set_xlabel("Effect"); ax.set_ylabel("Cause")
plt.colorbar(im, ax=ax, shrink=0.8)
for i in range(D):
    for j in range(D):
        v = data[i,j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v < 0.03 else "black")

# Panel 3: VAR significant links
ax = axes[2]
sig_mat = np.zeros((D, D))
for _, row in var_df[var_df["sig"]].iterrows():
    i = cols.index(row["cause"]); j = cols.index(row["effect"])
    sig_mat[i, j] = 1 - row["p"]

np.fill_diagonal(sig_mat, np.nan)
im2 = ax.imshow(sig_mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
ax.set_xticks(range(D)); ax.set_xticklabels(cols, rotation=30, ha="right")
ax.set_yticks(range(D)); ax.set_yticklabels(cols)
ax.set_title("VAR Significant Links\n(row → column)", fontsize=11)
ax.set_xlabel("Effect"); ax.set_ylabel("Cause")
for i in range(D):
    for j in range(D):
        if sig_mat[i,j] > 0:
            ax.text(j, i, "✓", ha="center", va="center", fontsize=14, color="white")

plt.suptitle("Granger Causality Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("granger_causality_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: granger_causality_analysis.png")
