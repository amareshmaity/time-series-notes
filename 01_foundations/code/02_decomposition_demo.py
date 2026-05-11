"""
02_decomposition_demo.py
========================
Module 01 — Foundations of Time Series
Topic   : Time Series Decomposition

Covers:
  - Classical decomposition (additive and multiplicative)
  - STL decomposition (single seasonality)
  - MSTL decomposition (multiple seasonalities)
  - Measuring trend and seasonal strength
  - Deseasonalized forecasting workflow

Dataset: Monthly airline passengers (single seasonality)
         Hourly electricity (multiple seasonality — simulated)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# ── Plotting defaults ────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
BLUE  = "#2C7BB6"
RED   = "#D7191C"
GREEN = "#1A9641"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Load airline passengers data
# ─────────────────────────────────────────────────────────────────────────────

def load_airline() -> pd.Series:
    """Load monthly airline passenger data."""
    try:
        url = (
            "https://raw.githubusercontent.com/jbrownlee/Datasets"
            "/master/airline-passengers.csv"
        )
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        s = df.squeeze()
        s.index.freq = "MS"
    except Exception:
        df = sns.load_dataset("flights")
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str), format="%Y-%B"
        )
        s = df.set_index("date")["passengers"].sort_index()
        s.index.freq = "MS"
    s.name = "Passengers"
    return s


series = load_airline()
print(f"Loaded: {series.name} | {len(series)} observations | {series.index[0].date()} → {series.index[-1].date()}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLASSICAL DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  1. Classical Decomposition")
print("="*55)

# Additive decomposition
result_add = seasonal_decompose(series, model="additive", period=12)
fig_add = result_add.plot()
fig_add.suptitle("Classical Additive Decomposition — Airline Passengers", fontweight="bold")
plt.tight_layout()
plt.savefig("01_classical_additive.png", bbox_inches="tight")
plt.show()

# Multiplicative decomposition
result_mul = seasonal_decompose(series, model="multiplicative", period=12)
fig_mul = result_mul.plot()
fig_mul.suptitle("Classical Multiplicative Decomposition — Airline Passengers", fontweight="bold")
plt.tight_layout()
plt.savefig("02_classical_multiplicative.png", bbox_inches="tight")
plt.show()

print("\nAirline passengers have MULTIPLICATIVE seasonality:")
print("  → Peaks grow larger as the total number of passengers increases")
print("  → Seasonal amplitude is proportional to the level")

# Check residuals
add_resid_std = result_add.resid.dropna().std()
mul_resid_std = result_mul.resid.dropna().std()
print(f"\nResidual std — Additive:        {add_resid_std:.2f}")
print(f"Residual std — Multiplicative:  {mul_resid_std:.4f}")
print("(Lower residual std = better fit for this dataset)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. STL DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  2. STL Decomposition")
print("="*55)

# STL on log-transformed series (log makes multiplicative → additive)
series_log = np.log(series)
series_log.name = "log(Passengers)"

stl = STL(series_log, period=12, robust=True)
stl_result = stl.fit()

# Plot
fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)

axes[0].plot(series_log, color=BLUE, linewidth=1.5, label="log(Passengers)")
axes[0].set_title("Original (log scale)")
axes[0].legend(loc="upper left")

axes[1].plot(stl_result.trend, color=RED, linewidth=2, label="Trend")
axes[1].set_title("Trend Component")
axes[1].legend(loc="upper left")

axes[2].plot(stl_result.seasonal, color=GREEN, linewidth=1.5, label="Seasonal")
axes[2].axhline(0, color="black", linewidth=0.6, linestyle="--")
axes[2].set_title("Seasonal Component (period=12)")
axes[2].legend(loc="upper left")

axes[3].plot(stl_result.resid, color="gray", linewidth=1, label="Residual")
axes[3].axhline(0, color="black", linewidth=0.6, linestyle="--")
axes[3].set_title("Residual Component")
axes[3].legend(loc="upper left")

plt.suptitle("STL Decomposition — log(Airline Passengers)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("03_stl_decomposition.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 3. STL PARAMETER COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  3. STL Parameter Sensitivity")
print("="*55)

seasonal_smoothness_values = [7, 13, 25, 51]

fig, axes = plt.subplots(len(seasonal_smoothness_values), 1, figsize=(13, 10), sharex=True)

for i, sw in enumerate(seasonal_smoothness_values):
    r = STL(series_log, period=12, seasonal=sw, robust=True).fit()
    axes[i].plot(r.seasonal, label=f"seasonal={sw}", color=plt.cm.viridis(i / len(seasonal_smoothness_values)))
    axes[i].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[i].set_ylabel("Seasonal")
    axes[i].legend(loc="upper right", fontsize=9)

plt.suptitle("Effect of `seasonal` Parameter on STL Seasonal Component",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("04_stl_parameter_sensitivity.png", bbox_inches="tight")
plt.show()

print("Observation:")
print("  seasonal=7   → seasonal component can change rapidly over time")
print("  seasonal=51  → seasonal component changes very slowly (more stable)")
print("  In practice, start with seasonal=13 and adjust based on domain knowledge")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MSTL — MULTIPLE SEASONALITY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  4. MSTL — Multiple Seasonality")
print("="*55)

# Simulate hourly data with daily (24) + weekly (168) seasonality
np.random.seed(42)
n_hours = 24 * 7 * 8   # 8 weeks of hourly data

time_index = pd.date_range(start="2023-01-02", periods=n_hours, freq="h")
t = np.arange(n_hours)

# Components
trend_component    = 100 + 0.05 * t
daily_seasonal     = 20 * np.sin(2 * np.pi * t / 24)           # peaks at midday
weekly_seasonal    = 10 * np.sin(2 * np.pi * t / (24 * 7))     # peaks on weekdays
noise              = np.random.normal(0, 3, n_hours)

series_hourly = pd.Series(
    trend_component + daily_seasonal + weekly_seasonal + noise,
    index=time_index,
    name="Electricity Load (MW)"
)

print(f"\nSimulated hourly electricity demand: {len(series_hourly)} observations")
print(f"Period: {series_hourly.index[0]} → {series_hourly.index[-1]}")

# MSTL decomposition
mstl = MSTL(series_hourly, periods=[24, 168])
mstl_result = mstl.fit()

# Plot
fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True)

axes[0].plot(series_hourly.values[:24*7*2], color=BLUE, linewidth=1, label="Observed")
axes[0].set_title("Original (first 2 weeks shown)")
axes[0].legend(loc="upper right")

axes[1].plot(mstl_result.trend[:24*7*2], color=RED, linewidth=2, label="Trend")
axes[1].set_title("Trend Component")
axes[1].legend(loc="upper right")

axes[2].plot(mstl_result.seasonal["seasonal_24"][:24*7*2], color=GREEN,
             linewidth=1.5, label="Daily Seasonal (s=24)")
axes[2].axhline(0, color="black", linewidth=0.5, linestyle="--")
axes[2].set_title("Daily Seasonal Component")
axes[2].legend(loc="upper right")

axes[3].plot(mstl_result.seasonal["seasonal_168"][:24*7*2], color="purple",
             linewidth=1.5, label="Weekly Seasonal (s=168)")
axes[3].axhline(0, color="black", linewidth=0.5, linestyle="--")
axes[3].set_title("Weekly Seasonal Component")
axes[3].legend(loc="upper right")

axes[4].plot(mstl_result.resid[:24*7*2], color="gray", linewidth=1, label="Residual")
axes[4].axhline(0, color="black", linewidth=0.5, linestyle="--")
axes[4].set_title("Residual Component")
axes[4].legend(loc="upper right")

plt.suptitle("MSTL Decomposition — Simulated Hourly Electricity Load\n(Daily s=24 + Weekly s=168)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("05_mstl_decomposition.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5. MEASURING TREND AND SEASONAL STRENGTH
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  5. Component Strength Measurement")
print("="*55)

def compute_strength(stl_fitted) -> dict:
    """Compute trend and seasonal strength from an STL result."""
    R = stl_fitted.resid
    T = stl_fitted.trend
    S = stl_fitted.seasonal

    var_R  = np.var(R, ddof=1)
    var_TR = np.var(T + R, ddof=1)
    var_SR = np.var(S + R, ddof=1)

    trend_strength    = max(0.0, 1 - var_R / var_TR)
    seasonal_strength = max(0.0, 1 - var_R / var_SR)

    return {
        "trend_strength":    round(trend_strength, 4),
        "seasonal_strength": round(seasonal_strength, 4),
    }


# Measure on airline data (log-transformed)
strengths = compute_strength(stl_result)
print(f"\nAirline Passengers (log scale):")
print(f"  Trend strength    : {strengths['trend_strength']:.4f}  ← {'Strong' if strengths['trend_strength'] > 0.6 else 'Moderate/Weak'}")
print(f"  Seasonal strength : {strengths['seasonal_strength']:.4f}  ← {'Strong' if strengths['seasonal_strength'] > 0.6 else 'Moderate/Weak'}")

# Interpretation guide
print("\nStrength Interpretation Guide:")
print("  > 0.8  → Very strong — must model explicitly")
print("  0.5–0.8 → Moderate — include in model")
print("  0.2–0.5 → Weak — consider including")
print("  < 0.2  → Very weak — likely safe to ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 6. DESEASONALIZED FORECASTING WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  6. Deseasonalized Forecasting Workflow")
print("="*55)

# Use log-transformed airline data
series_log = np.log(series)

# Train/test split (last 12 months = test)
train = series_log[:-12]
test  = series_log[-12:]

# Step 1: Decompose training series
stl_train = STL(train, period=12, robust=True).fit()

# Step 2: Deseasonalize training series
seasonal_train = stl_train.seasonal
deseasonalized_train = train - seasonal_train

print(f"\nTrain period : {train.index[0].date()} → {train.index[-1].date()} ({len(train)} obs)")
print(f"Test period  : {test.index[0].date()} → {test.index[-1].date()} ({len(test)} obs)")

# Step 3: Fit ARIMA on deseasonalized training series
model = ARIMA(deseasonalized_train, order=(1, 1, 1))
fitted_model = model.fit()
print(f"\nARIMA(1,1,1) fitted on deseasonalized series")
print(f"  AIC: {fitted_model.aic:.2f} | BIC: {fitted_model.bic:.2f}")

# Step 4: Forecast deseasonalized values (12 steps ahead)
forecast_deseason = fitted_model.forecast(steps=12)

# Step 5: Re-add seasonal component
# Use the seasonal values from the last year of training data
last_year_seasonal = seasonal_train[-12:].values
forecast_final_log = forecast_deseason.values + last_year_seasonal

# Convert back from log scale
forecast_final = np.exp(forecast_final_log)
actual_final   = np.exp(test.values)

# Compute RMSE
rmse = np.sqrt(np.mean((forecast_final - actual_final) ** 2))
mape = np.mean(np.abs((actual_final - forecast_final) / actual_final)) * 100
print(f"\n  RMSE : {rmse:.2f} passengers")
print(f"  MAPE : {mape:.2f}%")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(13, 9))

# Deseasonalized forecast
axes[0].plot(train.index, deseasonalized_train, color=BLUE, linewidth=1.5, label="Deseasonalized Train")
forecast_index = test.index
axes[0].plot(forecast_index, forecast_deseason.values, color=RED, linewidth=2,
             linestyle="--", label="Deseasonalized Forecast")
axes[0].axvline(train.index[-1], color="black", linewidth=1, linestyle=":", alpha=0.7)
axes[0].set_title("Step 3–4: ARIMA Forecast on Deseasonalized Series (log scale)")
axes[0].legend()

# Final forecast vs actual (original scale)
axes[1].plot(series.index, series.values, color=BLUE, linewidth=1.5, label="Actual (full series)")
axes[1].plot(forecast_index, forecast_final, color=RED, linewidth=2.5,
             linestyle="--", label=f"Final Forecast (RMSE={rmse:.0f})")
axes[1].axvline(train.index[-1], color="black", linewidth=1, linestyle=":", alpha=0.7)
axes[1].set_title("Step 5: Re-Seasonalized Forecast vs. Actual (original scale)")
axes[1].legend()

plt.suptitle("Deseasonalized Forecasting Workflow — Airline Passengers",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("06_deseasonalized_forecast.png", bbox_inches="tight")
plt.show()

print("\n✅ Decomposition demonstration complete.")
print("Summary of methods covered:")
print("  1. Classical decomposition   — simple but limited (NaN edges, fixed seasonality)")
print("  2. STL decomposition         — robust, flexible, single seasonality")
print("  3. MSTL decomposition        — handles multiple seasonal periods")
print("  4. Component strength        — quantify trend and seasonal dominance")
print("  5. Deseasonalized workflow   — practical production forecasting pattern")
