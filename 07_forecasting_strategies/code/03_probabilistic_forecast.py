"""
code/03_probabilistic_forecast.py
===================================
Module 07 — Forecasting Strategies
Practical: Probabilistic forecasting with quantile regression + conformal prediction.

Demonstrates:
  - Training multi-quantile LightGBM models (q10, q50, q90)
  - Split conformal prediction intervals (coverage-guaranteed)
  - Adaptive Conformal Inference (ACI) for non-stationary series
  - Conformalized Quantile Regression (CQR)
  - Coverage evaluation and calibration plots
  - Side-by-side comparison of all three PI methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import lightgbm as lgb
from sklearn.metrics import mean_pinball_loss, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Non-Stationary Dataset
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(0)
N = 700

t       = np.arange(N)
trend   = 0.06 * t
season  = 12 * np.sin(2 * np.pi * t / 52)    # 52-week annual cycle

# Increasing volatility in the second half (regime shift)
sigma   = np.where(t < N // 2, 2.0, 4.5)
noise   = np.random.normal(0, sigma, N)
series  = 100 + trend + season + noise

print(f"Series: N={N}, mean={series.mean():.1f}, std={series.std():.1f}")
print(f"Regime 1 (first half) std: {series[:N//2].std():.1f}")
print(f"Regime 2 (second half) std: {series[N//2:].std():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

LAGS = [1, 2, 3, 7, 14, 21]

def build_features(series: np.ndarray, lags: list) -> tuple[np.ndarray, np.ndarray]:
    """Leakage-free lag feature matrix and targets."""
    max_lag = max(lags)
    X, y    = [], []
    for i in range(max_lag, len(series)):
        X.append([series[i - lag] for lag in lags])
        y.append(series[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


X_all, y_all = build_features(series, LAGS)
n_all        = len(X_all)

# Time-ordered splits: train | calibration | test
N_TEST = 150
N_CAL  = 100

X_train = X_all[:n_all - N_TEST - N_CAL]
y_train = y_all[:n_all - N_TEST - N_CAL]
X_cal   = X_all[n_all - N_TEST - N_CAL: n_all - N_TEST]
y_cal   = y_all[n_all - N_TEST - N_CAL: n_all - N_TEST]
X_test  = X_all[n_all - N_TEST:]
y_test  = y_all[n_all - N_TEST:]

print(f"\nSplit: train={len(X_train)}, cal={len(X_cal)}, test={len(X_test)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Method A — Quantile Regression (LightGBM)
# ─────────────────────────────────────────────────────────────────────────────

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
LGB_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
                  min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                  verbose=-1)

print("\nTraining quantile regression models...")
qr_models = {}
for tau in QUANTILES:
    model = lgb.LGBMRegressor(objective="quantile", alpha=tau, **LGB_PARAMS)
    model.fit(X_train, y_train,
              eval_set=[(X_cal, y_cal)],
              callbacks=[lgb.early_stopping(40, verbose=False)])
    qr_models[tau] = model

# Predictions
qr_preds = {tau: model.predict(X_test) for tau, model in qr_models.items()}

# Enforce quantile monotonicity (fix crossing)
sorted_taus = sorted(QUANTILES)
for i in range(1, len(sorted_taus)):
    t1, t2 = sorted_taus[i-1], sorted_taus[i]
    qr_preds[t2] = np.maximum(qr_preds[t2], qr_preds[t1])

# Evaluate
print("\nQuantile Regression — Pinball Loss (test set):")
for tau in [0.10, 0.50, 0.90]:
    pb = mean_pinball_loss(y_test, qr_preds[tau], alpha=tau)
    print(f"  τ={tau:.2f}: {pb:.4f}")

def coverage(y_true, lower, upper):
    return ((y_true >= lower) & (y_true <= upper)).mean()

cov_qr_80 = coverage(y_test, qr_preds[0.10], qr_preds[0.90])
cov_qr_90 = coverage(y_test, qr_preds[0.05], qr_preds[0.95])
print(f"\nQuantile Regression Coverage:")
print(f"  80% PI (q10–q90): {cov_qr_80:.1%} (nominal 80%)")
print(f"  90% PI (q05–q95): {cov_qr_90:.1%} (nominal 90%)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Method B — Split Conformal Prediction
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing Split Conformal intervals...")

# Base model (point forecaster — use q50 model)
base_model = qr_models[0.50]

# Nonconformity scores on calibration set
cal_preds  = base_model.predict(X_cal)
cal_scores = np.abs(y_cal - cal_preds)

def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n   = len(scores)
    lvl = np.ceil((1 - alpha) * (n + 1)) / n
    return float(np.quantile(scores, min(lvl, 1.0)))

q_hat_90 = split_conformal_quantile(cal_scores, alpha=0.10)
q_hat_80 = split_conformal_quantile(cal_scores, alpha=0.20)

test_preds  = base_model.predict(X_test)
sc_lower_90 = test_preds - q_hat_90
sc_upper_90 = test_preds + q_hat_90
sc_lower_80 = test_preds - q_hat_80
sc_upper_80 = test_preds + q_hat_80

cov_sc_90 = coverage(y_test, sc_lower_90, sc_upper_90)
cov_sc_80 = coverage(y_test, sc_lower_80, sc_upper_80)
print(f"\nSplit Conformal Coverage:")
print(f"  90% PI: {cov_sc_90:.1%} (nominal 90%, guaranteed ≥ 90%)")
print(f"  80% PI: {cov_sc_80:.1%} (nominal 80%, guaranteed ≥ 80%)")
print(f"  Interval half-width (90%): ±{q_hat_90:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Method C — Adaptive Conformal Inference (ACI)
# ─────────────────────────────────────────────────────────────────────────────

print("\nRunning Adaptive Conformal Inference (ACI)...")

TARGET_ALPHA = 0.10   # target 90% coverage
GAMMA        = 0.02   # adaptation step size

# Initialize with calibration scores
alpha_t      = TARGET_ALPHA
score_buffer = list(cal_scores)   # seed with calibration scores
aci_lower    = []
aci_upper    = []
aci_alphas   = []
aci_covered  = []

for i in range(len(X_test)):
    # Current conformal quantile
    n     = len(score_buffer)
    lvl   = np.ceil((1 - alpha_t) * (n + 1)) / n
    q_aci = float(np.quantile(score_buffer, min(lvl, 1.0)))

    y_hat = test_preds[i]
    lo    = y_hat - q_aci
    hi    = y_hat + q_aci
    aci_lower.append(lo)
    aci_upper.append(hi)
    aci_alphas.append(alpha_t)

    # Observe true value and update
    err       = 0 if lo <= y_test[i] <= hi else 1
    alpha_t   = np.clip(alpha_t + GAMMA * (TARGET_ALPHA - err), 0.01, 0.99)
    score_buffer.append(abs(y_test[i] - y_hat))
    aci_covered.append(lo <= y_test[i] <= hi)

aci_lower   = np.array(aci_lower)
aci_upper   = np.array(aci_upper)
aci_covered = np.array(aci_covered)

cov_aci = aci_covered.mean()
print(f"\nACI Coverage:")
print(f"  Long-run empirical coverage: {cov_aci:.1%} (target 90%)")
print(f"  Final alpha_t: {aci_alphas[-1]:.4f}")
print(f"  Mean interval width: {(aci_upper - aci_lower).mean():.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Method D — Conformalized Quantile Regression (CQR)
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing CQR intervals...")

q_low_cal  = qr_models[0.05].predict(X_cal)
q_high_cal = qr_models[0.95].predict(X_cal)

# CQR nonconformity score: max undershoot from low, overshoot from high
cqr_scores = np.maximum(q_low_cal - y_cal, y_cal - q_high_cal)

# Conformal adjustment
q_cqr = split_conformal_quantile(cqr_scores, alpha=0.10)
print(f"  CQR conformal adjustment: {q_cqr:.4f}")

q_low_test  = qr_preds[0.05]
q_high_test = qr_preds[0.95]
cqr_lower   = q_low_test  - q_cqr
cqr_upper   = q_high_test + q_cqr

cov_cqr = coverage(y_test, cqr_lower, cqr_upper)
print(f"\nCQR Coverage:")
print(f"  90% PI: {cov_cqr:.1%} (nominal 90%)")
print(f"  Mean interval width: {(cqr_upper - cqr_lower).mean():.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Coverage Summary Table
# ─────────────────────────────────────────────────────────────────────────────

width_qr_90  = (qr_preds[0.95]  - qr_preds[0.05]).mean()
width_sc_90  = (sc_upper_90 - sc_lower_90).mean()
width_aci    = (aci_upper   - aci_lower).mean()
width_cqr    = (cqr_upper   - cqr_lower).mean()

summary = pd.DataFrame({
    "Method":            ["QR (q5–q95)",     "Split Conformal", "ACI",          "CQR"],
    "Nominal Coverage":  ["90%",             "90%",             "90%",          "90%"],
    "Empirical Coverage":[f"{cov_qr_90:.1%}", f"{cov_sc_90:.1%}", f"{cov_aci:.1%}", f"{cov_cqr:.1%}"],
    "Mean Width":        [f"{width_qr_90:.2f}", f"{width_sc_90:.2f}", f"{width_aci:.2f}", f"{width_cqr:.2f}"],
    "Coverage Guaranteed":["❌ (asymptotic)", "✅ (finite-sample)", "✅ (long-run)", "✅ (finite-sample)"],
})
print("\n" + "="*75)
print("PROBABILISTIC FORECAST METHOD COMPARISON")
print("="*75)
print(summary.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Visualization — Comparison Plot
# ─────────────────────────────────────────────────────────────────────────────

test_idx = np.arange(len(y_test))
SHOW     = 100   # show last 100 test points for clarity

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

configs = [
    ("Quantile Regression (q5–q95)", qr_lower := qr_preds[0.05], qr_upper := qr_preds[0.95],
     test_preds, "#9C27B0", cov_qr_90, width_qr_90),
    ("Split Conformal (90%)",        sc_lower_90, sc_upper_90,
     test_preds, "#2196F3", cov_sc_90, width_sc_90),
    ("Adaptive Conformal (ACI)",     aci_lower, aci_upper,
     test_preds, "#FF9800", cov_aci, width_aci),
    ("Conformalized QR (CQR)",       cqr_lower, cqr_upper,
     test_preds, "#4CAF50", cov_cqr, width_cqr),
]

for ax, (title, lower, upper, point, color, cov, width) in zip(axes, configs):
    sl = slice(len(y_test) - SHOW, None)
    ax.fill_between(test_idx[sl], lower[sl], upper[sl],
                    color=color, alpha=0.25, label=f"PI (90%)")
    ax.plot(test_idx[sl], point[sl], color=color, linewidth=1.5, label="Point")
    ax.scatter(test_idx[sl], y_test[sl], color="black", s=10, zorder=5, label="Actual")

    missed = ~((y_test[sl] >= lower[sl]) & (y_test[sl] <= upper[sl]))
    ax.scatter(test_idx[sl][missed], y_test[sl][missed],
               color="red", s=30, zorder=6, marker="x", label="Missed")

    ax.set_title(f"{title}\nCoverage: {cov:.1%} | Width: {width:.2f}", fontsize=10)
    ax.set_xlabel("Test Step")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

plt.suptitle("Probabilistic Forecast Comparison — 90% Prediction Intervals",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("probabilistic_forecast_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: probabilistic_forecast_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Calibration Plot — Rolling Coverage Over Time
# ─────────────────────────────────────────────────────────────────────────────

WINDOW = 50

fig, ax = plt.subplots(figsize=(13, 5))

# Rolling coverage for each method
roll_kw  = dict(window=WINDOW, min_periods=10)
roll_qr  = pd.Series((y_test >= qr_preds[0.05]) & (y_test <= qr_preds[0.95])).rolling(**roll_kw).mean()
roll_sc  = pd.Series((y_test >= sc_lower_90) & (y_test <= sc_upper_90)).rolling(**roll_kw).mean()
roll_aci = pd.Series(aci_covered).rolling(**roll_kw).mean()
roll_cqr = pd.Series((y_test >= cqr_lower) & (y_test <= cqr_upper)).rolling(**roll_kw).mean()

ax.plot(roll_qr,  label="Quantile Regression", color="#9C27B0", linewidth=2)
ax.plot(roll_sc,  label="Split Conformal",      color="#2196F3", linewidth=2)
ax.plot(roll_aci, label="ACI",                  color="#FF9800", linewidth=2)
ax.plot(roll_cqr, label="CQR",                  color="#4CAF50", linewidth=2)
ax.axhline(0.90, color="red", linestyle="--", linewidth=1.5, label="Target 90%")

ax.set_title(f"Rolling {WINDOW}-Step Coverage — Regime Shift at Step {N_TEST//2}",
             fontsize=12)
ax.set_xlabel("Test Step")
ax.set_ylabel("Rolling Coverage")
ax.set_ylim(0.5, 1.02)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.axvline(x=N_TEST // 2, color="gray", linestyle=":", linewidth=1.5,
           label="Regime Shift")
plt.tight_layout()
plt.savefig("rolling_coverage_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved: rolling_coverage_diagnostics.png")
