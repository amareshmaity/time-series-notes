"""
code/03_statistical_tests.py
===============================
Module 08 — Evaluation & Metrics
Practical: Statistical tests for forecast model comparison.

Demonstrates:
  - Diebold-Mariano test (DM) for two-model comparison
  - Modified DM test (MDM) for small-sample robustness
  - Model Confidence Set (MCS) for multi-model comparison
  - Paired t-test as simple baseline
  - Full comparison report across 5 models
  - Visualization of loss differentials and p-value distribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# 1. Test Implementations
# ─────────────────────────────────────────────────────────────────────────────

def dm_test(errors_a, errors_b, h=1, loss="mae", alternative="two-sided"):
    """
    Diebold-Mariano test for equal predictive accuracy.

    errors_a, errors_b : forecast errors (y_true - y_pred), shape (T,)
    h                  : forecast horizon (HAC lag truncation)
    loss               : 'mae' or 'mse'
    alternative        : 'two-sided' | 'less' | 'greater'
    """
    ea, eb = np.asarray(errors_a, float), np.asarray(errors_b, float)
    T = len(ea)

    d     = np.abs(ea) - np.abs(eb) if loss == "mae" else ea**2 - eb**2
    d_bar = d.mean()

    # HAC variance (Newey-West)
    gamma0  = ((d - d_bar)**2).mean()
    hac_var = gamma0
    for lag in range(1, h):
        w      = 1 - lag / h
        gamma  = ((d[lag:] - d_bar) * (d[:-lag] - d_bar)).mean()
        hac_var += 2 * w * gamma

    se  = np.sqrt(max(hac_var, 1e-16) / T)
    dm  = d_bar / se

    if alternative == "two-sided":
        p = 2 * (1 - stats.norm.cdf(abs(dm)))
    elif alternative == "less":
        p = stats.norm.cdf(dm)
    else:
        p = 1 - stats.norm.cdf(dm)

    return {"dm": float(dm), "p": float(p), "d_bar": float(d_bar), "T": T,
            "reject_5pct": bool(p < 0.05)}


def mdm_test(errors_a, errors_b, h=1, loss="mae"):
    """
    Modified DM test (Harvey, Leybourne & Newbold, 1997).
    Better for small samples — uses t(T-1) distribution.
    """
    ea, eb = np.asarray(errors_a, float), np.asarray(errors_b, float)
    T  = len(ea)
    d  = np.abs(ea) - np.abs(eb) if loss == "mae" else ea**2 - eb**2
    d_bar = d.mean()

    gamma0  = ((d - d_bar)**2).mean()
    hac_var = gamma0
    for lag in range(1, h):
        w = 1 - lag / h
        hac_var += 2 * w * ((d[lag:] - d_bar) * (d[:-lag] - d_bar)).mean()

    se          = np.sqrt(max(hac_var, 1e-16) / T)
    dm          = d_bar / se
    correction  = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    mdm         = dm * correction
    p           = 2 * (1 - stats.t.cdf(abs(mdm), df=T-1))

    return {"mdm": float(mdm), "dm": float(dm), "p": float(p),
            "d_bar": float(d_bar), "reject_5pct": bool(p < 0.05)}


def paired_ttest_forecast(errors_a, errors_b, loss="mae", alpha=0.05):
    """Paired t-test on absolute errors."""
    la = np.abs(errors_a) if loss == "mae" else errors_a**2
    lb = np.abs(errors_b) if loss == "mae" else errors_b**2
    d  = la - lb
    t, p = stats.ttest_1samp(d, popmean=0)
    return {"t": float(t), "p": float(p), "mean_diff": float(d.mean()),
            "reject_5pct": bool(p < alpha)}


def model_confidence_set(losses, alpha=0.10, B=1000, model_names=None):
    """
    Model Confidence Set (Hansen, Lunde & Nason, 2011).
    Returns subset of models not significantly worse than the best.

    losses      : (T, M) matrix of per-period loss values
    alpha       : significance level (0.10 = 90% MCS)
    B           : number of bootstrap replications
    model_names : list of model name strings
    """
    losses = np.asarray(losses, float)
    T, M   = losses.shape
    if model_names is None:
        model_names = [f"M{i}" for i in range(M)]

    surviving   = list(range(M))
    elim_order  = []

    while len(surviving) > 1:
        idx       = np.array(surviving)
        sub       = losses[:, idx]        # (T, n_surv)
        n_s       = len(idx)

        # Relative loss vs group mean at each period
        group_mean = sub.mean(axis=1, keepdims=True)
        rel        = sub - group_mean      # (T, n_s)
        mu         = rel.mean(axis=0)      # (n_s,)

        # Bootstrap variance
        boot_mu = np.zeros((B, n_s))
        for b in range(B):
            bsample    = rel[np.random.choice(T, T, replace=True)]
            boot_mu[b] = bsample.mean(axis=0)
        std_boot = boot_mu.std(axis=0) + 1e-12

        t_stats = np.abs(mu / std_boot)
        TR      = t_stats.max()

        boot_TR = np.abs(boot_mu / std_boot).max(axis=1)
        crit    = np.quantile(boot_TR, 1 - alpha)

        if TR <= crit:
            break   # remaining models form the MCS

        worst_local  = t_stats.argmax()
        worst_global = idx[worst_local]
        surviving.remove(worst_global)
        elim_order.append(model_names[worst_global])

    return {
        "mcs_members":      [model_names[i] for i in surviving],
        "eliminated":       elim_order,
        "alpha":            alpha,
        "n_total":          M,
        "n_in_mcs":         len(surviving),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Synthetic Forecast Scenario
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(99)
T = 120   # number of test periods

# True signal
y_true = np.cumsum(np.random.randn(T) * 0.5) + 50

# Five model forecasts with different accuracy levels
models = {
    "Best":     y_true + np.random.normal(0, 0.8, T),   # near-perfect
    "Good":     y_true + np.random.normal(0, 1.5, T),   # good
    "Mediocre": y_true + np.random.normal(0, 2.5, T),   # moderate
    "Naïve":    np.roll(y_true, 1),                     # random walk naïve
    "Biased":   y_true + 3.0,                           # constant +3 bias
}
models["Naïve"][0] = y_true[0]

print(f"Test period: T={T}")
print("\nModel MAE values:")
for name, preds in models.items():
    print(f"  {name:10s}: MAE={np.abs(y_true - preds).mean():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Diebold-Mariano Tests — All vs. Naïve
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("DIEBOLD-MARIANO TESTS (all models vs. Naïve, h=1, loss=mae)")
print("="*65)
print(f"{'Model':<12} {'DM stat':>9} {'MDM stat':>10} {'p-value':>10} {'Sig?':>6}")
print("-"*65)

e_naive  = y_true - models["Naïve"]
dm_rows  = []

for name, preds in models.items():
    if name == "Naïve":
        continue
    e_model  = y_true - preds
    dm_res   = dm_test(e_model, e_naive, h=1, loss="mae")
    mdm_res  = mdm_test(e_model, e_naive, h=1, loss="mae")
    sig      = "✅" if dm_res["reject_5pct"] else "ns"
    print(f"{name:<12} {dm_res['dm']:>+9.3f} {mdm_res['mdm']:>+10.3f} {dm_res['p']:>10.4f} {sig:>6}")
    dm_rows.append({"Model": name, **dm_res, "mdm": mdm_res["mdm"],
                     "p_mdm": mdm_res["p"]})

dm_df = pd.DataFrame(dm_rows).set_index("Model")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-Horizon DM Test (h=6)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("MODIFIED DM TEST — 6-STEP HORIZON (h=6 HAC correction)")
print("="*65)
print(f"{'Model':<12} {'MDM stat':>10} {'p-value':>10} {'Sig?':>6}")
print("-"*65)

for name, preds in models.items():
    if name == "Naïve":
        continue
    e_model = y_true - preds
    res     = mdm_test(e_model, e_naive, h=6, loss="mae")
    sig     = "✅" if res["reject_5pct"] else "ns"
    print(f"{name:<12} {res['mdm']:>+10.3f} {res['p']:>10.4f} {sig:>6}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model Confidence Set
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("MODEL CONFIDENCE SET (α=0.10, 90% MCS)")
print("="*60)

# Build loss matrix: (T, M)
names_list = list(models.keys())
loss_matrix = np.column_stack([
    np.abs(y_true - models[n]) for n in names_list
])

mcs_90 = model_confidence_set(loss_matrix, alpha=0.10,
                               model_names=names_list, B=2000)
mcs_25 = model_confidence_set(loss_matrix, alpha=0.25,
                               model_names=names_list, B=2000)

print(f"\n90% MCS members  (α=0.10): {mcs_90['mcs_members']}")
print(f"75% MCS members  (α=0.25): {mcs_25['mcs_members']}")
print(f"\nElimination order (worst first): {mcs_90['eliminated']}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Paired t-Test
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("PAIRED t-TEST (Good vs. Mediocre)")
print("="*60)

e_good     = y_true - models["Good"]
e_mediocre = y_true - models["Mediocre"]
t_res      = paired_ttest_forecast(e_good, e_mediocre, loss="mae")

print(f"  t statistic : {t_res['t']:+.4f}")
print(f"  p-value     : {t_res['p']:.4f}")
print(f"  Mean |err| diff (Good - Mediocre): {t_res['mean_diff']:.4f}")
print(f"  Significant at 5%? {'Yes ✅' if t_res['reject_5pct'] else 'No ❌'}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full Comparison Report
# ─────────────────────────────────────────────────────────────────────────────

def full_comparison_report(y_true, models, baseline_name, h=1):
    """Generate comprehensive report: metrics + DM tests vs. baseline."""
    e_base = y_true - models[baseline_name]
    rows   = []
    for name, preds in models.items():
        e_m  = y_true - preds
        mae_ = np.abs(e_m).mean()
        rmse_= np.sqrt((e_m**2).mean())

        if name == baseline_name:
            row = {"Model": name, "MAE": round(mae_, 4),
                   "RMSE": round(rmse_, 4), "DM": "—", "p": "—",
                   "Sig": "baseline", "Rank": "—"}
        else:
            dm  = dm_test(e_m, e_base, h=h, loss="mae")
            row = {"Model": name, "MAE": round(mae_, 4),
                   "RMSE": round(rmse_, 4), "DM": round(dm["dm"], 3),
                   "p": round(dm["p"], 4),
                   "Sig": "✅" if dm["reject_5pct"] else "ns",
                   "Rank": "better" if dm["d_bar"] < 0 else "worse"}
        rows.append(row)

    return pd.DataFrame(rows).set_index("Model")

report = full_comparison_report(y_true, models, baseline_name="Naïve", h=1)
print("\n" + "="*70)
print("FULL COMPARISON REPORT (DM < 0 = model better than baseline)")
print("="*70)
print(report.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 8. Visualization
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 8a. Loss differentials over time (model vs. naïve)
axes[0,0].axhline(0, color="black", linestyle="--", linewidth=1)
for name in ["Best", "Good", "Mediocre", "Biased"]:
    diff = np.abs(y_true - models[name]) - np.abs(y_true - models["Naïve"])
    axes[0,0].plot(diff, alpha=0.7, label=name, linewidth=1.5)
axes[0,0].set_title("MAE Loss Differential vs. Naïve (negative = better)", fontsize=11)
axes[0,0].set_xlabel("Test Step")
axes[0,0].set_ylabel("Δ|Error| (model − naïve)")
axes[0,0].legend(fontsize=9)
axes[0,0].grid(alpha=0.3)

# 8b. DM statistic bar chart
dm_stats = {}
for name in ["Best", "Good", "Mediocre", "Biased"]:
    e  = y_true - models[name]
    dm_stats[name] = dm_test(e, e_naive, h=1, loss="mae")["dm"]

names_  = list(dm_stats.keys())
vals_   = [dm_stats[n] for n in names_]
colors_ = ["#4CAF50" if v < -1.96 else ("#F44336" if v > 1.96 else "#FF9800")
           for v in vals_]
bars = axes[0,1].bar(names_, vals_, color=colors_, width=0.5)
axes[0,1].axhline(-1.96, color="green", linestyle="--", linewidth=1,
                   label="±1.96 (5% two-sided)")
axes[0,1].axhline(+1.96, color="red",   linestyle="--", linewidth=1)
axes[0,1].axhline(0,     color="black", linestyle="-",  linewidth=0.8)
axes[0,1].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
axes[0,1].set_title("DM Statistic vs. Naïve (< -1.96 → significantly better)", fontsize=11)
axes[0,1].set_ylabel("DM statistic")
axes[0,1].legend(fontsize=9)
axes[0,1].grid(alpha=0.3, axis="y")

# 8c. MCS membership visualization
all_model_names = list(models.keys())
mcs_members_90  = set(mcs_90["mcs_members"])
mcs_members_25  = set(mcs_25["mcs_members"])

mcs_data = pd.DataFrame({
    "Model": all_model_names,
    "MAE":   [np.abs(y_true - models[n]).mean() for n in all_model_names],
    "In 90% MCS": [1 if n in mcs_members_90 else 0 for n in all_model_names],
    "In 75% MCS": [1 if n in mcs_members_25 else 0 for n in all_model_names],
}).sort_values("MAE")

bar_c = ["#2196F3" if v else "#9E9E9E" for v in mcs_data["In 90% MCS"]]
axes[1,0].barh(mcs_data["Model"], mcs_data["MAE"], color=bar_c, height=0.5)
axes[1,0].set_title("Model MAE (blue = in 90% MCS)", fontsize=11)
axes[1,0].set_xlabel("MAE")
axes[1,0].grid(alpha=0.3, axis="x")
for i, (_, row) in enumerate(mcs_data.iterrows()):
    label = "✅ MCS" if row["In 90% MCS"] else "❌"
    axes[1,0].text(row["MAE"] + 0.02, i, label, va="center", fontsize=9)

# 8d. Bootstrap p-value distribution under H0 (demonstration)
np.random.seed(123)
n_boot = 1000
null_dm_stats = []
for _ in range(n_boot):
    e1 = np.random.normal(0, 1.5, T)
    e2 = np.random.normal(0, 1.5, T)
    null_dm_stats.append(dm_test(e1, e2, h=1, loss="mae")["dm"])

null_dm = np.array(null_dm_stats)
axes[1,1].hist(null_dm, bins=50, color="#2196F3", alpha=0.7,
               density=True, label="Bootstrap DM under H₀")
x = np.linspace(-4, 4, 200)
axes[1,1].plot(x, stats.norm.pdf(x), "r-", linewidth=2, label="N(0,1)")
axes[1,1].axvline(-1.96, color="orange", linestyle="--", label="±1.96")
axes[1,1].axvline(+1.96, color="orange", linestyle="--")
axes[1,1].set_title("DM Statistic Distribution Under H₀ (equal accuracy)", fontsize=11)
axes[1,1].set_xlabel("DM Statistic")
axes[1,1].legend(fontsize=9)
axes[1,1].grid(alpha=0.3)

plt.suptitle("Statistical Tests for Forecast Model Comparison",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("statistical_tests_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: statistical_tests_comparison.png")
