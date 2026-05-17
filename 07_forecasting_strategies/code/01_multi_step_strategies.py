"""
code/01_multi_step_strategies.py
=================================
Module 07 — Forecasting Strategies
Practical: Direct vs. Recursive vs. MIMO comparison on the same dataset.

Demonstrates:
  - Building lag features without data leakage
  - Training Recursive (1 model), Direct (H models), and MIMO strategies
  - Rolling-origin evaluation across horizon steps
  - Side-by-side accuracy comparison plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Dataset
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
N = 600
t = np.arange(N)

# Trend + weekly seasonality + noise
trend    = 0.05 * t
seasonal = 8 * np.sin(2 * np.pi * t / 7)
noise    = np.random.normal(0, 1.5, N)
series   = 50 + trend + seasonal + noise

dates = pd.date_range("2020-01-01", periods=N, freq="D")
df    = pd.DataFrame({"ds": dates, "y": series})

print(f"Series length: {N}")
print(f"Date range   : {dates[0].date()} → {dates[-1].date()}")
print(df.tail(3))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering — Lag Features (No Leakage)
# ─────────────────────────────────────────────────────────────────────────────

LAGS    = [1, 2, 3, 7, 14]
HORIZON = 14

def build_lag_features(series: np.ndarray, lags: list) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) matrices for one-step-ahead training."""
    max_lag = max(lags)
    X, y    = [], []
    for i in range(max_lag, len(series)):
        X.append([series[i - lag] for lag in lags])
        y.append(series[i])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Strategy A — Recursive
# ─────────────────────────────────────────────────────────────────────────────

def train_recursive(series: np.ndarray, lags: list) -> LGBMRegressor:
    X, y  = build_lag_features(series, lags)
    model = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                          num_leaves=31, verbose=-1)
    model.fit(X, y)
    return model


def forecast_recursive(model, series_end: np.ndarray, lags: list, h: int) -> np.ndarray:
    """Multi-step recursive forecast — predictions fed back as inputs."""
    max_lag = max(lags)
    buffer  = deque(series_end[-max_lag:], maxlen=max_lag)
    preds   = []
    for _ in range(h):
        feat = np.array([[list(buffer)[-lag] for lag in lags]])
        p    = model.predict(feat)[0]
        preds.append(p)
        buffer.append(p)
    return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Strategy B — Direct
# ─────────────────────────────────────────────────────────────────────────────

def train_direct(series: np.ndarray, lags: list, horizon: int) -> dict:
    """Train one LightGBM model per horizon step."""
    max_lag = max(lags)
    models  = {}
    for h in range(1, horizon + 1):
        X_h, y_h = [], []
        for i in range(max_lag, len(series) - h + 1):
            X_h.append([series[i - lag] for lag in lags])
            y_h.append(series[i + h - 1])
        model = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, verbose=-1)
        model.fit(np.array(X_h), np.array(y_h))
        models[h] = model
    return models


def forecast_direct(models: dict, series_end: np.ndarray, lags: list, horizon: int) -> np.ndarray:
    """Generate direct forecasts — each model predicts its horizon step independently."""
    feat = np.array([[series_end[-lag] for lag in lags]])
    return np.array([models[h].predict(feat)[0] for h in range(1, horizon + 1)])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Strategy C — MIMO (Multi-Input Multi-Output)
# ─────────────────────────────────────────────────────────────────────────────

def train_mimo(series: np.ndarray, lags: list, horizon: int) -> MultiOutputRegressor:
    """Train a single multi-output regressor predicting all H steps at once."""
    max_lag = max(lags)
    X, Y    = [], []
    for i in range(max_lag, len(series) - horizon + 1):
        X.append([series[i - lag] for lag in lags])
        Y.append([series[i + h] for h in range(horizon)])
    model = MultiOutputRegressor(
        LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1),
        n_jobs=-1,
    )
    model.fit(np.array(X), np.array(Y))
    return model


def forecast_mimo(model, series_end: np.ndarray, lags: list) -> np.ndarray:
    """MIMO predicts the full horizon vector in a single forward pass."""
    feat = np.array([[series_end[-lag] for lag in lags]])
    return model.predict(feat).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Rolling-Origin Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def rolling_origin_eval(series: np.ndarray, lags: list, horizon: int,
                         n_origins: int = 20, min_train: int = 200) -> pd.DataFrame:
    """
    Walk-forward evaluation across all three strategies.

    Returns per-horizon MAE for each strategy.
    """
    origins = np.linspace(min_train, len(series) - horizon, n_origins, dtype=int)

    results = {strat: np.zeros((n_origins, horizon))
               for strat in ["Recursive", "Direct", "MIMO"]}
    actuals = np.zeros((n_origins, horizon))

    for idx, origin in enumerate(origins):
        train  = series[:origin]
        actual = series[origin:origin + horizon]
        actuals[idx] = actual

        # ── Recursive ──
        m_rec             = train_recursive(train, lags)
        results["Recursive"][idx] = forecast_recursive(m_rec, train, lags, horizon)

        # ── Direct ──
        m_dir             = train_direct(train, lags, horizon)
        results["Direct"][idx]    = forecast_direct(m_dir, train, lags, horizon)

        # ── MIMO ──
        m_mimo            = train_mimo(train, lags, horizon)
        results["MIMO"][idx]      = forecast_mimo(m_mimo, train, lags)

        if (idx + 1) % 5 == 0:
            print(f"  Evaluated {idx+1}/{n_origins} origins")

    # Compute per-horizon MAE
    mae_records = []
    for strat, preds in results.items():
        for h in range(horizon):
            mae_records.append({
                "strategy":  strat,
                "horizon":   h + 1,
                "mae":       mean_absolute_error(actuals[:, h], preds[:, h]),
            })

    return pd.DataFrame(mae_records)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Run Evaluation
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("Rolling-origin evaluation (this may take ~1 minute)...")
print("="*60)

s = df["y"].values
mae_df = rolling_origin_eval(s, lags=LAGS, horizon=HORIZON, n_origins=15, min_train=250)

# Pivot for display
pivot = mae_df.pivot(index="horizon", columns="strategy", values="mae")
print("\nMAE per Horizon Step:")
print(pivot.round(4))
print(f"\nOverall mean MAE:")
print(mae_df.groupby("strategy")["mae"].mean().round(4))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Visualization
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {"Recursive": "#E84646", "Direct": "#4CAF50", "MIMO": "#2196F3"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: MAE per horizon step
for strat in ["Recursive", "Direct", "MIMO"]:
    sub = mae_df[mae_df["strategy"] == strat]
    axes[0].plot(sub["horizon"], sub["mae"], marker="o", label=strat,
                 color=COLORS[strat], linewidth=2)
axes[0].set_title("MAE per Forecast Horizon Step", fontsize=13)
axes[0].set_xlabel("Horizon Step")
axes[0].set_ylabel("MAE")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Mean MAE bar chart
mean_mae = mae_df.groupby("strategy")["mae"].mean()
bars = axes[1].bar(mean_mae.index, mean_mae.values,
                   color=[COLORS[s] for s in mean_mae.index], width=0.5)
axes[1].bar_label(bars, fmt="%.3f", padding=3, fontweight="bold")
axes[1].set_title("Overall Mean MAE by Strategy", fontsize=13)
axes[1].set_ylabel("Mean MAE")
axes[1].set_ylim(0, mean_mae.max() * 1.25)
axes[1].grid(alpha=0.3, axis="y")

plt.suptitle("Direct vs. Recursive vs. MIMO Multi-Step Forecast Comparison",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("multi_step_strategies_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: multi_step_strategies_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Sample Forecast Visualization
# ─────────────────────────────────────────────────────────────────────────────

train_data = s[:500]
test_data  = s[500:500 + HORIZON]

m_rec  = train_recursive(train_data, LAGS)
m_dir  = train_direct(train_data, LAGS, HORIZON)
m_mimo = train_mimo(train_data, LAGS, HORIZON)

preds_rec  = forecast_recursive(m_rec,  train_data, LAGS, HORIZON)
preds_dir  = forecast_direct(m_dir,     train_data, LAGS, HORIZON)
preds_mimo = forecast_mimo(m_mimo,      train_data, LAGS)

fig, ax = plt.subplots(figsize=(12, 5))
ctx = 60
ax.plot(range(-ctx, 0), train_data[-ctx:], color="black", linewidth=2, label="History")
ax.plot(range(HORIZON),  test_data,        color="black", linewidth=2, linestyle=":", label="Actual")
ax.plot(range(HORIZON),  preds_rec,  color="#E84646", marker="o", linewidth=2, label="Recursive")
ax.plot(range(HORIZON),  preds_dir,  color="#4CAF50", marker="s", linewidth=2, label="Direct")
ax.plot(range(HORIZON),  preds_mimo, color="#2196F3", marker="^", linewidth=2, label="MIMO")
ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
ax.set_title("14-Day Forecast — Strategy Comparison", fontsize=13)
ax.set_xlabel("Steps from Forecast Origin")
ax.set_ylabel("Value")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("multi_step_sample_forecast.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved: multi_step_sample_forecast.png")
