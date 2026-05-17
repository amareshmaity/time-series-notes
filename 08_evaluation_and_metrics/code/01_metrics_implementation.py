"""
code/01_metrics_implementation.py
===================================
Module 08 — Evaluation & Metrics
Practical: All forecast metrics implemented from scratch + benchmarked against sklearn.

Demonstrates:
  - MAE, MSE, RMSE, MAPE, SMAPE, MdAE, MASE, wMAPE, OWA
  - Skill scores vs. naïve baseline
  - Metric failure modes (zeros, outliers, asymmetry)
  - Comparison table across multiple mock models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_pinball_loss

# ─────────────────────────────────────────────────────────────────────────────
# 1. Metric Implementations
# ─────────────────────────────────────────────────────────────────────────────

def mae(y_true, y_pred):
    return float(np.abs(np.asarray(y_true) - np.asarray(y_pred)).mean())

def mse(y_true, y_pred):
    return float(((np.asarray(y_true) - np.asarray(y_pred))**2).mean())

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def mape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)))

def smape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom))

def mdae(y_true, y_pred):
    return float(np.median(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def wmape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.abs(y_true - y_pred).sum() / (np.abs(y_true).sum() + eps))

def mase(y_train, y_test, y_pred, seasonality=1, eps=1e-12):
    y_train = np.asarray(y_train, dtype=float)
    y_test  = np.asarray(y_test,  dtype=float)
    y_pred  = np.asarray(y_pred,  dtype=float)
    if seasonality == 1:
        scale = np.abs(np.diff(y_train)).mean()
    else:
        scale = np.abs(y_train[seasonality:] - y_train[:-seasonality]).mean()
    return float(np.abs(y_test - y_pred).mean() / (scale + eps))

def skill_score_mae(y_true, y_pred_model, y_pred_baseline):
    mae_m = mae(y_true, y_pred_model)
    mae_b = mae(y_true, y_pred_baseline)
    return float(1 - mae_m / (mae_b + 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Synthetic Dataset
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
N  = 300
t  = np.arange(N)

# Monthly series with trend and seasonality
y  = 100 + 0.2*t + 15*np.sin(2*np.pi*t/12) + np.random.normal(0, 3, N)

SPLIT   = 250
y_train = y[:SPLIT]
y_test  = y[SPLIT:]
h       = len(y_test)

print(f"Train: {SPLIT} obs | Test: {h} obs")
print(f"Test  — mean: {y_test.mean():.1f}, std: {y_test.std():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Naive Baseline Forecasts
# ─────────────────────────────────────────────────────────────────────────────

# Non-seasonal naïve: repeat last value
naive_random_walk = np.full(h, y_train[-1])

# Seasonal naïve: repeat last season (monthly, s=12)
S = 12
tail         = y_train[-S:]
naive_seasonal = np.tile(tail, (h // S) + 1)[:h]

print(f"\nNaïve (random walk) MAE: {mae(y_test, naive_random_walk):.3f}")
print(f"Naïve (seasonal)    MAE: {mae(y_test, naive_seasonal):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Mock Model Forecasts
# ─────────────────────────────────────────────────────────────────────────────

# Simulate different quality forecasts
np.random.seed(1)
preds_excellent = y_test + np.random.normal(0, 1.0, h)    # tight errors
preds_good      = y_test + np.random.normal(0, 3.0, h)    # moderate errors
preds_biased    = y_test + 8.0                             # constant bias
preds_outlier   = y_test + np.random.normal(0, 2.0, h)    # with one huge error
preds_outlier[10] += 100                                   # inject outlier


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full Metric Table
# ─────────────────────────────────────────────────────────────────────────────

def metrics_row(name, preds, y_test, y_train, baseline_preds, s=12):
    return {
        "Model":       name,
        "MAE":         round(mae(y_test, preds), 3),
        "MdAE":        round(mdae(y_test, preds), 3),
        "RMSE":        round(rmse(y_test, preds), 3),
        "MAPE (%)":    round(100 * mape(y_test, preds), 2),
        "SMAPE (%)":   round(100 * smape(y_test, preds), 2),
        "wMAPE (%)":   round(100 * wmape(y_test, preds), 2),
        "MASE":        round(mase(y_train, y_test, preds, seasonality=s), 4),
        "Skill (MAE)": round(skill_score_mae(y_test, preds, baseline_preds), 4),
        "RMSE/MAE":    round(rmse(y_test, preds) / (mae(y_test, preds) + 1e-12), 3),
    }

rows = [
    metrics_row("Naïve (rw)",   naive_random_walk, y_test, y_train, naive_random_walk),
    metrics_row("Naïve (seas)", naive_seasonal,    y_test, y_train, naive_random_walk),
    metrics_row("Excellent",    preds_excellent,   y_test, y_train, naive_random_walk),
    metrics_row("Good",         preds_good,        y_test, y_train, naive_random_walk),
    metrics_row("Biased",       preds_biased,      y_test, y_train, naive_random_walk),
    metrics_row("Outlier",      preds_outlier,     y_test, y_train, naive_random_walk),
]

df = pd.DataFrame(rows).set_index("Model")
print("\n" + "="*90)
print("METRIC COMPARISON TABLE")
print("="*90)
print(df.to_string())
print(f"\nModels with MASE < 1.0 (beat naïve): {df[df['MASE']<1.0].index.tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Failure Mode Demonstrations
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("METRIC FAILURE MODES")
print("="*60)

# 6a. MAPE with near-zero actuals
y_zero = np.array([0.1, 100, 200, 0.05, 150])
y_zero_pred = np.array([1.0, 90, 210, 1.0, 140])
print(f"\n⚠  MAPE with near-zero actuals: {100*mape(y_zero, y_zero_pred):.1f}%")
print(f"   SMAPE (more robust):          {100*smape(y_zero, y_zero_pred):.1f}%")
print(f"   MAE (no division):            {mae(y_zero, y_zero_pred):.2f}")

# 6b. RMSE vs MAE with outlier
errors_clean   = np.ones(100)               # 100 errors of magnitude 1
errors_outlier = np.ones(100); errors_outlier[50] = 50  # one spike

print(f"\n⚠  Outlier effect on RMSE vs MAE:")
print(f"   Clean  — MAE: {errors_clean.mean():.2f},   RMSE: {np.sqrt((errors_clean**2).mean()):.2f}, ratio: {np.sqrt((errors_clean**2).mean())/errors_clean.mean():.2f}")
print(f"   Outlier— MAE: {errors_outlier.mean():.2f},   RMSE: {np.sqrt((errors_outlier**2).mean()):.2f}, ratio: {np.sqrt((errors_outlier**2).mean())/errors_outlier.mean():.2f}")

# 6c. MAPE asymmetry
y_ref  = np.array([100.0])
y_over = np.array([150.0])   # over-forecast by 50%
y_under= np.array([50.0])    # under-forecast by 50%
print(f"\n⚠  MAPE asymmetry (ref=100):")
print(f"   Over-forecast  +50% → MAPE={100*mape(y_ref, y_over):.0f}%")
print(f"   Under-forecast -50% → MAPE={100*mape(y_ref, y_under):.0f}%  (same!)")
print(f"   Under-forecast  -90%→ MAPE={100*mape(y_ref, np.array([10.0])):.0f}% (bounded at 100%)")
print(f"   Over-forecast  +900%→ MAPE={100*mape(y_ref, np.array([1000.0])):.0f}% (unbounded!)")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Sklearn Cross-Check
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("SKLEARN CROSS-CHECK (should match our implementations)")
print("="*60)

preds = preds_good
print(f"MAE  (ours): {mae(y_test, preds):.6f}")
print(f"MAE  (sklearn): {mean_absolute_error(y_test, preds):.6f}")
print(f"RMSE (ours): {rmse(y_test, preds):.6f}")
print(f"RMSE (sklearn): {np.sqrt(mean_squared_error(y_test, preds)):.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Visualization
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 8a. Error distributions
axes[0,0].set_title("Error Distributions by Model", fontsize=12)
for name, preds in [("Excellent", preds_excellent), ("Good", preds_good),
                     ("Biased", preds_biased), ("Outlier", preds_outlier)]:
    errors = y_test - preds
    axes[0,0].hist(errors, bins=20, alpha=0.5, label=name, density=True)
axes[0,0].axvline(0, color="black", linestyle="--")
axes[0,0].set_xlabel("Error (y_true - y_pred)")
axes[0,0].legend(fontsize=8)
axes[0,0].grid(alpha=0.3)

# 8b. MAE vs RMSE scatter (outlier sensitivity)
models  = ["Naïve(rw)", "Naïve(seas)", "Excellent", "Good", "Biased", "Outlier"]
all_preds = [naive_random_walk, naive_seasonal,
             preds_excellent, preds_good, preds_biased, preds_outlier]
maes  = [mae(y_test, p) for p in all_preds]
rmses = [rmse(y_test, p) for p in all_preds]

axes[0,1].scatter(maes, rmses, s=80, zorder=3)
for m, r, name in zip(maes, rmses, models):
    axes[0,1].annotate(name, (m, r), textcoords="offset points",
                       xytext=(5, 3), fontsize=8)
mn = max(maes + rmses)
axes[0,1].plot([0, mn], [0, mn], "k--", alpha=0.4, label="RMSE = MAE")
axes[0,1].set_xlabel("MAE")
axes[0,1].set_ylabel("RMSE")
axes[0,1].set_title("MAE vs RMSE (above diagonal = outlier-sensitive)", fontsize=11)
axes[0,1].legend(fontsize=9)
axes[0,1].grid(alpha=0.3)

# 8c. MASE bar chart
mases = [mase(y_train, y_test, p, seasonality=12) for p in all_preds]
colors = ["#4CAF50" if m < 1 else "#F44336" for m in mases]
bars = axes[1,0].bar(models, mases, color=colors, width=0.6)
axes[1,0].axhline(1.0, color="black", linestyle="--", label="Naïve threshold")
axes[1,0].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
axes[1,0].set_title("MASE (< 1.0 = beats naïve baseline)", fontsize=12)
axes[1,0].set_ylabel("MASE")
axes[1,0].tick_params(axis="x", rotation=20)
axes[1,0].legend(fontsize=9)
axes[1,0].grid(alpha=0.3, axis="y")

# 8d. MAPE vs wMAPE
mapes  = [100*mape(y_test, p) for p in all_preds]
wmapes = [100*wmape(y_test, p) for p in all_preds]
x = np.arange(len(models))
w = 0.35
axes[1,1].bar(x - w/2, mapes,  w, label="MAPE",  color="#2196F3")
axes[1,1].bar(x + w/2, wmapes, w, label="wMAPE", color="#FF9800")
axes[1,1].set_xticks(x)
axes[1,1].set_xticklabels(models, rotation=20, fontsize=9)
axes[1,1].set_ylabel("%")
axes[1,1].set_title("MAPE vs wMAPE Comparison", fontsize=12)
axes[1,1].legend()
axes[1,1].grid(alpha=0.3, axis="y")

plt.suptitle("Forecast Metric Comparison and Failure Mode Analysis",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: metrics_comparison.png")
