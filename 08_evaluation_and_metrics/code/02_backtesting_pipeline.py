"""
code/02_backtesting_pipeline.py
=================================
Module 08 — Evaluation & Metrics
Practical: Rolling-origin backtesting framework with multiple models.

Demonstrates:
  - Expanding window walk-forward validation
  - Fixed-window sliding backtesting
  - Gap strategy (production data lag simulation)
  - Per-horizon MASE and MAE comparison
  - Backtesting of multiple models in one pass
  - Result visualization: accuracy vs. horizon, model ranking
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Optional

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Dataset — Monthly retail sales with trend + seasonality
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
N = 480   # 40 years of monthly data
t = np.arange(N)

trend    = 0.08 * t
seasonal = 20 * np.sin(2 * np.pi * t / 12) + 5 * np.sin(4 * np.pi * t / 12)
noise    = np.random.normal(0, 4, N)
series   = 200 + trend + seasonal + noise

dates = pd.date_range("1985-01", periods=N, freq="ME")
df    = pd.DataFrame({"ds": dates, "y": series})

print(f"Series: N={N}, mean={series.mean():.1f}, std={series.std():.1f}")
print(f"Date range: {dates[0].date()} → {dates[-1].date()}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Forecast Model Functions
# ─────────────────────────────────────────────────────────────────────────────

def naive_rw(train: np.ndarray, h: int, **_) -> np.ndarray:
    """Naïve (random walk): repeat last value."""
    return np.full(h, train[-1])


def naive_seasonal(train: np.ndarray, h: int, s: int = 12, **_) -> np.ndarray:
    """Seasonal naïve: repeat last season."""
    tail = train[-s:]
    return np.tile(tail, (h // s) + 1)[:h]


def ses_forecast(train: np.ndarray, h: int, alpha: float = 0.3, **_) -> np.ndarray:
    """Simple exponential smoothing — flat forecast."""
    level = train[0]
    for y in train:
        level = alpha * y + (1 - alpha) * level
    return np.full(h, level)


def holt_forecast(train: np.ndarray, h: int,
                   alpha: float = 0.3, beta: float = 0.1, **_) -> np.ndarray:
    """Holt's linear trend method — handles trend."""
    level = train[0]
    trend = train[1] - train[0]
    for y in train[1:]:
        prev_level = level
        level = alpha * y + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return np.array([level + j * trend for j in range(1, h + 1)])


def theta_forecast(train: np.ndarray, h: int, **_) -> np.ndarray:
    """
    Simplified Theta method (Assimakopoulos & Nikolopoulos, 2000).
    Decomposes series into theta=0 (trend) and theta=2 (fluctuations).
    Combined forecast = 0.5 * (theta0 + theta2_ses).
    """
    n     = len(train)
    times = np.arange(1, n + 1)

    # Theta-0 line: OLS trend through the data
    m, b = np.polyfit(times, train, 1)
    theta0_fc = np.array([m * (n + j) + b for j in range(1, h + 1)])

    # Theta-2 SES on the series
    theta2_fc = ses_forecast(train, h, alpha=0.3)

    return 0.5 * theta0_fc + 0.5 * theta2_fc


def lgbm_forecast(train: np.ndarray, h: int, lags: list = None, **_) -> np.ndarray:
    """LightGBM recursive forecaster with lag features."""
    try:
        import lightgbm as lgb
    except ImportError:
        return ses_forecast(train, h)  # fallback

    if lags is None:
        lags = [1, 2, 3, 6, 12]

    max_lag = max(lags)
    if len(train) < max_lag + 10:
        return ses_forecast(train, h)

    X, y = [], []
    for i in range(max_lag, len(train)):
        X.append([train[i - lag] for lag in lags])
        y.append(train[i])
    X, y = np.array(X), np.array(y)

    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1,
                               num_leaves=15, verbose=-1)
    model.fit(X, y)

    from collections import deque
    buffer = deque(train[-max_lag:], maxlen=max_lag)
    preds  = []
    for _ in range(h):
        feat = np.array([[list(buffer)[-lag] for lag in lags]])
        p    = model.predict(feat)[0]
        preds.append(p)
        buffer.append(p)
    return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Backtesting Engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    horizon:      int
    n_origins:    int
    min_train:    int
    step:         Optional[int] = None
    gap:          int           = 0
    window_type:  str           = "expanding"   # 'expanding' or 'fixed'
    fixed_size:   Optional[int] = None
    seasonality:  int           = 1


def run_backtest_multi_model(
    series: np.ndarray,
    models: dict,
    cfg: BacktestConfig,
) -> dict:
    """
    Walk-forward backtest for multiple models simultaneously.

    Parameters
    ----------
    series : full time series (1D array)
    models : dict {model_name: callable(train, h) → forecast_array}
    cfg    : BacktestConfig object

    Returns
    -------
    dict {model_name: BacktestResult dict}
    """
    step = cfg.step or cfg.horizon
    n    = len(series)

    # Generate evaluation origins (train_end indices)
    max_train_end = n - 1 - cfg.gap - cfg.horizon
    min_train_end = cfg.min_train - 1
    origins       = np.linspace(min_train_end, max_train_end,
                                 cfg.n_origins, dtype=int)

    # Compute MASE denominator scale (once, from full training data)
    train_end_idx = origins[0]
    if cfg.seasonality == 1:
        scale = np.abs(np.diff(series[:train_end_idx+1])).mean() + 1e-12
    else:
        s = series[:train_end_idx+1]
        scale = np.abs(s[cfg.seasonality:] - s[:-cfg.seasonality]).mean() + 1e-12

    results = {name: {"preds": [], "actuals": []} for name in models}

    for i, train_end in enumerate(origins):
        test_start = train_end + cfg.gap + 1
        test_end   = test_start + cfg.horizon

        if test_end > n:
            break

        # Determine training slice
        if cfg.window_type == "fixed" and cfg.fixed_size:
            tr_start = max(0, train_end + 1 - cfg.fixed_size)
        else:
            tr_start = 0

        train  = series[tr_start:train_end + 1]
        actual = series[test_start:test_end]

        for name, model_fn in models.items():
            try:
                fc = model_fn(train, cfg.horizon)[:len(actual)]
            except Exception as e:
                fc = np.full(len(actual), train[-1])  # fallback on failure
            results[name]["preds"].append(fc)
            results[name]["actuals"].append(actual)

        if (i + 1) % 5 == 0:
            print(f"  Origin {i+1}/{len(origins)}: train[{tr_start}..{train_end}] → test[{test_start}..{test_end-1}]")

    # Compute summary metrics
    summary = {}
    for name in models:
        preds   = np.array(results[name]["preds"])    # (n_origins, H)
        actuals = np.array(results[name]["actuals"])  # (n_origins, H)
        errors  = np.abs(actuals - preds)

        summary[name] = {
            "mean_mae":       float(errors.mean()),
            "mean_mase":      float(errors.mean() / scale),
            "mae_per_step":   errors.mean(axis=0),
            "rmse_per_step":  np.sqrt((actuals - preds)**2).mean(axis=0),
            "mase_per_step":  errors.mean(axis=0) / scale,
            "preds":          preds,
            "actuals":        actuals,
        }

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 4. Run Backtests
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "Naïve RW":       lambda tr, h: naive_rw(tr, h),
    "Naïve Seasonal": lambda tr, h: naive_seasonal(tr, h, s=12),
    "SES":            lambda tr, h: ses_forecast(tr, h, alpha=0.25),
    "Holt":           lambda tr, h: holt_forecast(tr, h, alpha=0.3, beta=0.1),
    "Theta":          lambda tr, h: theta_forecast(tr, h),
    "LightGBM":       lambda tr, h: lgbm_forecast(tr, h, lags=[1,2,3,6,12]),
}

CFG = BacktestConfig(
    horizon=12,
    n_origins=20,
    min_train=60,
    gap=0,
    window_type="expanding",
    seasonality=12,
)

print("\n" + "="*70)
print("Running walk-forward backtest (expanding window, 20 origins)...")
print("="*70)
bt_results = run_backtest_multi_model(series, MODELS, CFG)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary Table
# ─────────────────────────────────────────────────────────────────────────────

rows = []
for name, res in bt_results.items():
    rows.append({
        "Model":       name,
        "Mean MAE":    round(res["mean_mae"], 3),
        "Mean MASE":   round(res["mean_mase"], 4),
        "Beats Naïve": "✅" if res["mean_mase"] < 1.0 else "❌",
        "Best Step":   int(np.argmin(res["mae_per_step"])) + 1,
        "Worst Step":  int(np.argmax(res["mae_per_step"])) + 1,
    })

summary_df = pd.DataFrame(rows).set_index("Model").sort_values("Mean MASE")
print("\nBacktest Summary (sorted by MASE):")
print(summary_df.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fixed vs. Expanding Window Comparison
# ─────────────────────────────────────────────────────────────────────────────

CFG_FIXED = BacktestConfig(
    horizon=12, n_origins=20, min_train=60, gap=0,
    window_type="fixed", fixed_size=120, seasonality=12,
)

print("\nRunning fixed-window backtest (window=120)...")
bt_fixed = run_backtest_multi_model(series, {"LightGBM": MODELS["LightGBM"]}, CFG_FIXED)

lgbm_mase_expanding = bt_results["LightGBM"]["mean_mase"]
lgbm_mase_fixed     = bt_fixed["LightGBM"]["mean_mase"]
print(f"\nLightGBM — Expanding window MASE: {lgbm_mase_expanding:.4f}")
print(f"LightGBM — Fixed window MASE:     {lgbm_mase_fixed:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualization
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "Naïve RW":       "#9E9E9E",
    "Naïve Seasonal": "#607D8B",
    "SES":            "#FF9800",
    "Holt":           "#4CAF50",
    "Theta":          "#9C27B0",
    "LightGBM":       "#2196F3",
}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: MAE per horizon step
for name, res in bt_results.items():
    axes[0].plot(range(1, CFG.horizon+1), res["mae_per_step"],
                 marker="o", label=name, color=COLORS.get(name, "black"),
                 linewidth=2, markersize=5)
axes[0].set_title("MAE per Forecast Horizon Step", fontsize=13)
axes[0].set_xlabel("Horizon Step")
axes[0].set_ylabel("MAE")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Right: MASE bar chart
mases = {name: res["mean_mase"] for name, res in bt_results.items()}
sorted_names = sorted(mases, key=lambda x: mases[x])
bar_colors = ["#4CAF50" if mases[n] < 1 else "#F44336" for n in sorted_names]
bars = axes[1].barh(sorted_names, [mases[n] for n in sorted_names],
                    color=bar_colors, height=0.5)
axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="Naïve threshold")
axes[1].bar_label(bars, fmt="%.4f", padding=5, fontsize=9)
axes[1].set_title("Overall MASE (< 1.0 = beats naïve)", fontsize=13)
axes[1].set_xlabel("MASE")
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3, axis="x")

plt.suptitle("Walk-Forward Backtesting Results — Monthly Sales Series",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("backtesting_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: backtesting_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Gap Strategy Demo
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("Gap Strategy Comparison (SES model, gap=0 vs gap=3)")
print("="*60)

for gap in [0, 3]:
    cfg_g = BacktestConfig(
        horizon=12, n_origins=15, min_train=60, gap=gap, seasonality=12
    )
    bt_g = run_backtest_multi_model(series, {"SES": MODELS["SES"]}, cfg_g)
    print(f"  Gap={gap}: MAE={bt_g['SES']['mean_mae']:.3f}, MASE={bt_g['SES']['mean_mase']:.4f}")

print("\n→ Larger gap → harder problem → higher MAE/MASE")
print("  Ignoring the gap (gap=0 in backtest but gap=3 in production)")
print("  → Optimistic evaluation that won't reproduce in production")
