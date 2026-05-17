"""
code/01_stock_project/main.py
===============================
Module 13 — Projects & Case Studies
Project 1: Stock Price Forecasting

End-to-end pipeline:
  - Synthetic stock data (GBM) or yfinance
  - Feature engineering (technical indicators, lags)
  - Walk-forward CV: Random Walk, Ridge, LightGBM comparison
  - Bootstrap prediction intervals + coverage evaluation
  - MLflow tracking (optional)
  - Visualization dashboard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Stock Data (GBM)
# ─────────────────────────────────────────────────────────────────────────────

N      = 1000
mu     = 0.0004    # daily drift
sigma  = 0.018     # daily volatility
S0     = 150.0

rets   = np.random.normal(mu, sigma, N)
prices = S0 * np.exp(np.cumsum(rets))
dates  = pd.bdate_range("2020-01-01", periods=N)

df = pd.DataFrame({
    "close":  prices,
    "volume": np.random.randint(5_000_000, 15_000_000, N),
}, index=dates)

print(f"Stock dataset: {N} trading days, "
      f"price range [{df['close'].min():.2f}, {df['close'].max():.2f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

c = df["close"]
log_ret = np.log(c).diff()

feat = pd.DataFrame(index=df.index)
feat["target"] = log_ret.shift(-1)   # next-day log return (target)

# Lag features
for lag in [1, 2, 3, 5, 10, 21]:
    feat[f"ret_lag_{lag}"] = log_ret.shift(lag)

# Moving average divergence
for w in [5, 10, 21]:
    feat[f"price_vs_ma{w}"] = c / c.rolling(w).mean() - 1

# Realized volatility
for w in [5, 10, 21]:
    feat[f"rvol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

# RSI (14)
delta = c.diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
feat["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-12)))

# Bollinger band position
ma20   = c.rolling(20).mean()
std20  = c.rolling(20).std()
feat["bb_pos"] = (c - ma20) / (2 * std20 + 1e-12)

# MACD
ema12  = c.ewm(span=12).mean()
ema26  = c.ewm(span=26).mean()
macd   = ema12 - ema26
feat["macd"] = macd
feat["macd_signal"] = macd.ewm(span=9).mean()

# Volume
feat["vol_vs_ma20"] = df["volume"] / df["volume"].rolling(20).mean()

# Calendar
feat["day_of_week"] = df.index.dayofweek.astype(float)
feat["month"]       = df.index.month.astype(float)

feat_df = feat.dropna()
feature_cols = [c for c in feat_df.columns if c != "target"]
print(f"Features: {len(feature_cols)} | Samples: {len(feat_df)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Walk-Forward Splits
# ─────────────────────────────────────────────────────────────────────────────

def make_splits(n, n_splits=5, test_size=21, gap=1):
    step    = (n - test_size * n_splits) // (n_splits + 1)
    splits  = []
    for i in range(n_splits):
        tr_end   = step + i * step + step
        te_start = tr_end + gap
        te_end   = te_start + test_size
        if te_end <= n:
            splits.append((tr_end, te_start, te_end))
    return splits


X = feat_df[feature_cols].values
y = feat_df["target"].values
n = len(X)
splits = make_splits(n, n_splits=5, test_size=21)
print(f"\n{len(splits)} walk-forward folds (test_size=21 days each)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model Comparison
# ─────────────────────────────────────────────────────────────────────────────

results = {}

# Random Walk baseline
rw_preds, rw_true = [], []
for tr_end, te_start, te_end in splits:
    rw_preds.extend([0.0] * (te_end - te_start))  # log return = 0
    rw_true.extend(y[te_start:te_end])
results["Random Walk"] = {
    "mae":  round(mean_absolute_error(rw_true, rw_preds), 6),
    "preds": np.array(rw_preds), "true": np.array(rw_true),
}
print(f"\nRandom Walk:   MAE = {results['Random Walk']['mae']:.6f}")

# Ridge Regression
ridge_preds, ridge_true = [], []
for tr_end, te_start, te_end in splits:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
    pipe.fit(X[:tr_end], y[:tr_end])
    preds = pipe.predict(X[te_start:te_end])
    ridge_preds.extend(preds); ridge_true.extend(y[te_start:te_end])
results["Ridge"] = {
    "mae":  round(mean_absolute_error(ridge_true, ridge_preds), 6),
    "preds": np.array(ridge_preds), "true": np.array(ridge_true),
}
print(f"Ridge:         MAE = {results['Ridge']['mae']:.6f}")

# LightGBM (if available)
try:
    import lightgbm as lgb
    lgbm_preds, lgbm_true = [], []
    for tr_end, te_start, te_end in splits:
        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X[:tr_end])
        X_te_s  = scaler.transform(X[te_start:te_end])
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                                num_leaves=31, random_state=42, verbose=-1)
        m.fit(X_tr_s, y[:tr_end])
        lgbm_preds.extend(m.predict(X_te_s)); lgbm_true.extend(y[te_start:te_end])
    results["LightGBM"] = {
        "mae":   round(mean_absolute_error(lgbm_true, lgbm_preds), 6),
        "preds": np.array(lgbm_preds), "true": np.array(lgbm_true),
    }
    print(f"LightGBM:      MAE = {results['LightGBM']['mae']:.6f}")
except ImportError:
    print("LightGBM not installed (pip install lightgbm)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Bootstrap Prediction Intervals
# ─────────────────────────────────────────────────────────────────────────────

# Fit Ridge on last training fold; compute bootstrap PI for last test fold
tr_end, te_start, te_end = splits[-1]
scaler_pi = StandardScaler()
X_tr_pi = scaler_pi.fit_transform(X[:tr_end])
X_te_pi = scaler_pi.transform(X[te_start:te_end])

ridge_pi = Ridge(alpha=10.0)
ridge_pi.fit(X_tr_pi, y[:tr_end])
residuals = y[:tr_end] - ridge_pi.predict(X_tr_pi)
y_hat_pi  = ridge_pi.predict(X_te_pi)
y_te_pi   = y[te_start:te_end]

n_boot = 500
boot   = np.array([y_hat_pi + np.random.choice(residuals, len(y_hat_pi), replace=True)
                    for _ in range(n_boot)])
lower_95 = np.percentile(boot, 2.5, axis=0)
upper_95 = np.percentile(boot, 97.5, axis=0)
coverage = float(((y_te_pi >= lower_95) & (y_te_pi <= upper_95)).mean())

print(f"\nBootstrap PI (last fold):")
print(f"  95% coverage: {100*coverage:.1f}% (target: ~95%)")
print(f"  Avg interval width: {(upper_95 - lower_95).mean():.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualization
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Price series
ax = axes[0, 0]
ax.plot(df.index, df["close"], color="#2196F3", linewidth=1, alpha=0.8)
ax.set_title("Synthetic Stock Price (GBM)", fontsize=11)
ax.set_ylabel("Price ($)"); ax.grid(alpha=0.3)

# Panel 2: MAE comparison
ax = axes[0, 1]
names  = list(results.keys())
maes   = [results[m]["mae"] for m in names]
colors = ["#FF5722","#2196F3","#4CAF50"][:len(names)]
bars   = ax.bar(names, maes, color=colors, width=0.4)
ax.bar_label(bars, fmt="%.6f", padding=3, fontsize=9)
ax.set_ylabel("Walk-Forward MAE (log-return space)")
ax.set_title("Model Comparison", fontsize=11); ax.grid(alpha=0.3, axis="y")

# Panel 3: Prediction vs. actual (Ridge, last fold)
ax = axes[1, 0]
t_  = np.arange(len(y_te_pi))
ax.plot(t_, y_te_pi,  color="#2196F3", linewidth=1.5, label="Actual")
ax.plot(t_, y_hat_pi, color="#FF5722", linewidth=1.5, linestyle="--", label="Ridge forecast")
ax.fill_between(t_, lower_95, upper_95, alpha=0.2, color="#FF5722", label="95% PI")
ax.set_title(f"Ridge Forecast + Bootstrap PI (last fold)\n"
             f"Coverage = {100*coverage:.1f}%", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_xlabel("Days in test window"); ax.set_ylabel("Log return")

# Panel 4: Return distribution
ax = axes[1, 1]
ret_vals = log_ret.dropna().values
ax.hist(ret_vals, bins=50, color="#9C27B0", edgecolor="white", alpha=0.8, density=True)
x_ = np.linspace(ret_vals.min(), ret_vals.max(), 200)
ax.plot(x_, norm.pdf(x_, ret_vals.mean(), ret_vals.std()), "r-", linewidth=2, label="Normal fit")
ax.set_title("Log-Return Distribution (fat tails)", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle("Stock Price Forecasting Project", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("stock_forecast_results.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("STOCK PROJECT SUMMARY")
print("="*55)
print(f"{'Model':<15} {'MAE':>12}")
print("-"*30)
for name in results:
    print(f"  {name:<13} {results[name]['mae']:>12.6f}")
print(f"\nBootstrap PI coverage: {100*coverage:.1f}%")
print("Key takeaway: Hard to consistently beat random walk on log-returns.")
print("Plot saved: stock_forecast_results.png")
