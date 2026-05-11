"""
02_ets_models.py
================
Module 03 — Statistical Models
Topic   : Exponential Smoothing (SES, Holt, Holt-Winters, ETS)

Covers:
  - Simple Exponential Smoothing (SES)
  - Holt's linear trend (with and without damping)
  - Additive and multiplicative Holt-Winters
  - ETS framework with automatic model selection
  - Model comparison and diagnostics
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE = "#2C7BB6", "#D7191C", "#1A9641", "#F07D00"


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_airline():
    try:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        s = df.squeeze(); s.index.freq = "MS"
    except Exception:
        df = sns.load_dataset("flights")
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str), format="%Y-%B")
        s = df.set_index("date")["passengers"].sort_index(); s.index.freq = "MS"
    s.name = "Passengers"
    return s

series = load_airline()
train = series[:-24]
test  = series[-24:]
h = len(test)
print(f"Train: {len(train)} | Test: {h} | Seasonal period: 12")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SIMPLE EXPONENTIAL SMOOTHING (SES)
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- SES ---")
ses_model = SimpleExpSmoothing(train, initialization_method="estimated")
ses_fit = ses_model.fit(optimized=True)
alpha = ses_fit.params["smoothing_level"]
print(f"Optimal α = {alpha:.4f}")
ses_forecast = ses_fit.forecast(steps=h)


# ─────────────────────────────────────────────────────────────────────────────
# 3. HOLT'S LINEAR TREND
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Holt Linear + Damped ---")

holt_fit = ExponentialSmoothing(
    train, trend="add", seasonal=None, initialization_method="estimated"
).fit(optimized=True)
print(f"Holt: α={holt_fit.params['smoothing_level']:.4f}, β={holt_fit.params['smoothing_trend']:.4f}")

damped_fit = ExponentialSmoothing(
    train, trend="add", damped_trend=True, seasonal=None, initialization_method="estimated"
).fit(optimized=True)
print(f"Damped: φ={damped_fit.params['damping_trend']:.4f}")

holt_forecast  = holt_fit.forecast(h)
damped_forecast = damped_fit.forecast(h)


# ─────────────────────────────────────────────────────────────────────────────
# 4. HOLT-WINTERS
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Holt-Winters ---")

hw_add = ExponentialSmoothing(
    train, trend="add", seasonal="add", seasonal_periods=12,
    initialization_method="estimated"
).fit(optimized=True)
print(f"HW-Add: α={hw_add.params['smoothing_level']:.4f}, β={hw_add.params['smoothing_trend']:.4f}, γ={hw_add.params['smoothing_seasonal']:.4f}")

hw_mul = ExponentialSmoothing(
    train, trend="add", seasonal="mul", seasonal_periods=12,
    initialization_method="estimated"
).fit(optimized=True)
print(f"HW-Mul: α={hw_mul.params['smoothing_level']:.4f}, β={hw_mul.params['smoothing_trend']:.4f}, γ={hw_mul.params['smoothing_seasonal']:.4f}")

hw_add_fcst = hw_add.forecast(h)
hw_mul_fcst = hw_mul.forecast(h)


# ─────────────────────────────────────────────────────────────────────────────
# 5. AUTOMATIC ETS SELECTION
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Automatic ETS Model Selection ---")

best_aic = np.inf
best_config = None
best_ets = None

configs = [
    ("add", None, False),
    ("add", "add", False),
    ("add", "mul", False),
    ("add", "add", True),
    ("add", "mul", True),
    (None, None, False),
    (None, "add", False),
    (None, "mul", False),
]

for trend, seasonal, damped in configs:
    try:
        m = ExponentialSmoothing(
            train, trend=trend, seasonal=seasonal, seasonal_periods=12 if seasonal else None,
            damped_trend=damped, initialization_method="estimated"
        ).fit(optimized=True, disp=False)
        label = f"trend={trend}, seasonal={seasonal}, damped={damped}"
        print(f"  {label:<55} AIC={m.aic:.2f}")
        if m.aic < best_aic:
            best_aic = m.aic
            best_config = (trend, seasonal, damped)
            best_ets = m
    except Exception:
        continue

print(f"\nBest ETS: trend={best_config[0]}, seasonal={best_config[1]}, damped={best_config[2]}")
print(f"Best AIC: {best_aic:.2f}")
ets_fcst = best_ets.forecast(h)


# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def rmse(actual, pred):
    return np.sqrt(((actual.values - pred.values[:len(actual)]) ** 2).mean())

def mape(actual, pred):
    return (np.abs(actual.values - pred.values[:len(actual)]) / actual.values).mean() * 100

models_eval = {
    "SES":        ses_forecast,
    "Holt":       holt_forecast,
    "Holt-Damped": damped_forecast,
    "HW-Add":     hw_add_fcst,
    "HW-Mul":     hw_mul_fcst,
    "Auto-ETS":   ets_fcst,
}

print("\n--- Model Comparison ---")
results = pd.DataFrame([
    {"Model": name, "RMSE": round(rmse(test, fc), 2), "MAPE": round(mape(test, fc), 2), "AIC": None}
    for name, fc in models_eval.items()
]).set_index("Model").sort_values("RMSE")
print(results.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(13, 10))

# Full comparison
axes[0].plot(train[-36:], color="gray", linewidth=1.2, label="Train")
axes[0].plot(test, color="black", linewidth=2.5, label="Actual")
colors = [BLUE, RED, GREEN, ORANGE, "purple", "brown"]
for (name, fc), color in zip(models_eval.items(), colors):
    axes[0].plot(fc, color=color, linewidth=1.5, linestyle="--", label=name)
axes[0].axvline(train.index[-1], color="black", linewidth=0.8, linestyle=":", alpha=0.5)
axes[0].legend(fontsize=8)
axes[0].set_title("ETS Model Family — Forecast Comparison")

# Best model with prediction interval (HW-Mul usually best for airline)
best_model = hw_mul
best_forecast = hw_mul_fcst
simulations = best_model.simulate(nsimulations=h, repetitions=200, error="mul")
lower = np.percentile(simulations, 2.5, axis=1)
upper = np.percentile(simulations, 97.5, axis=1)

axes[1].plot(train[-36:], color="gray", linewidth=1.2, label="Train")
axes[1].plot(test, color="black", linewidth=2.5, label="Actual")
axes[1].plot(best_forecast, color=RED, linewidth=2, linestyle="--", label="HW-Multiplicative")
axes[1].fill_between(best_forecast.index, lower, upper, color=RED, alpha=0.15, label="95% Prediction Interval")
axes[1].axvline(train.index[-1], color="black", linewidth=0.8, linestyle=":", alpha=0.5)
axes[1].legend(fontsize=9)
axes[1].set_title("Holt-Winters (Multiplicative) with Prediction Intervals")

plt.suptitle("Exponential Smoothing Models — Airline Passengers", fontweight="bold")
plt.tight_layout()
plt.savefig("01_ets_comparison.png", bbox_inches="tight")
plt.show()

print(f"\n✅ ETS demo complete.")
print(f"   Best model: HW-Multiplicative | RMSE={rmse(test, hw_mul_fcst):.2f} | MAPE={mape(test, hw_mul_fcst):.2f}%")
