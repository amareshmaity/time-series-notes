# 01 — Naive Baseline Models

> **Module**: 03 Statistical Models | **File**: 1 of 6
>
> Before fitting any complex model, you must establish a baseline. Naive models are deceptively competitive — if your sophisticated model cannot beat a seasonal naive forecast, it has no value.

---

## Table of Contents

1. [Why Baselines Matter](#1-why-baselines-matter)
2. [Mean Forecast](#2-mean-forecast)
3. [Naive Forecast (Random Walk)](#3-naive-forecast-random-walk)
4. [Drift Forecast](#4-drift-forecast)
5. [Seasonal Naive Forecast](#5-seasonal-naive-forecast)
6. [Comparing Baselines](#6-comparing-baselines)
7. [When Baselines Are Hard to Beat](#7-when-baselines-are-hard-to-beat)

---

## 1. Why Baselines Matter

A baseline model:
- Provides a **lower bound on model quality** — any model worse than a baseline is useless
- Exposes whether the data has **any predictable structure** at all
- Acts as a **sanity check** before investing time in complex methods
- Is used to compute **skill scores** (relative metrics)

> **Industry rule**: If your production model doesn't beat seasonal naive by at least 10% on MAPE, question whether the extra complexity is worth it.

```
Model Value = Performance(Advanced Model) - Performance(Baseline)
If this difference is small → the baseline is already capturing most of the signal
```

---

## 2. Mean Forecast

### 2.1 Definition

Forecast all future values as the **historical mean**:

```
ŷ(t+h) = ȳ = (1/T) Σₜ₌₁ᵀ y(t)   for all h ≥ 1
```

### 2.2 When It Makes Sense

- Stationary series with no trend or seasonality
- Long-horizon forecasting when the series mean-reverts strongly
- As a naive benchmark for any series

### 2.3 Implementation

```python
import numpy as np
import pandas as pd

def mean_forecast(train: pd.Series, h: int) -> pd.Series:
    """Forecast the next h periods as the historical mean."""
    forecast_value = train.mean()
    forecast_idx = pd.date_range(
        start=train.index[-1] + train.index.freq,
        periods=h,
        freq=train.index.freq,
    )
    return pd.Series(forecast_value, index=forecast_idx, name="mean_forecast")

forecast_mean = mean_forecast(train, h=12)
```

### 2.4 Properties

| Property | Value |
|----------|-------|
| Parameters | 0 (just the mean) |
| Trend handling | ❌ None |
| Seasonality | ❌ None |
| Prediction intervals | Based on sample variance |

---

## 3. Naive Forecast (Random Walk)

### 3.1 Definition

Forecast all future values as the **last observed value**:

```
ŷ(t+h) = y(T)   for all h ≥ 1
```

This is equivalent to saying the series is a **random walk** — the best predictor of tomorrow is today.

### 3.2 When It Makes Sense

- Financial prices (approximately random walks)
- Series with no detectable trend or seasonality
- When the series changes unpredictably

```python
def naive_forecast(train: pd.Series, h: int) -> pd.Series:
    """Forecast next h periods as last observed value."""
    last_value = train.iloc[-1]
    forecast_idx = pd.date_range(
        start=train.index[-1] + train.index.freq,
        periods=h,
        freq=train.index.freq,
    )
    return pd.Series(last_value, index=forecast_idx, name="naive_forecast")
```

### 3.3 Prediction Intervals

For a naive forecast, prediction intervals grow with horizon:

```
95% PI: ŷ(t+h) ± 1.96 · σ · √h

Where σ = standard deviation of first differences
```

```python
sigma = train.diff().dropna().std()
h_vals = np.arange(1, h + 1)
margin = 1.96 * sigma * np.sqrt(h_vals)
upper = last_value + margin
lower = last_value - margin
```

---

## 4. Drift Forecast

### 4.1 Definition

Forecast future values by **extending the line** between the first and last observation:

```
ŷ(t+h) = y(T) + h · [(y(T) - y(1)) / (T - 1)]

Equivalently: y(T) + h · (average change per period)
```

This is the naive forecast with a **drift term** — it captures the long-run average change.

### 4.2 When It Makes Sense

- Strongly trended series where the trend is relatively linear
- When you expect recent momentum to continue

```python
def drift_forecast(train: pd.Series, h: int) -> pd.Series:
    """Forecast next h periods using the average drift."""
    T = len(train)
    drift = (train.iloc[-1] - train.iloc[0]) / (T - 1)
    last_value = train.iloc[-1]
    forecast_idx = pd.date_range(
        start=train.index[-1] + train.index.freq,
        periods=h,
        freq=train.index.freq,
    )
    forecasts = [last_value + step * drift for step in range(1, h + 1)]
    return pd.Series(forecasts, index=forecast_idx, name="drift_forecast")
```

---

## 5. Seasonal Naive Forecast

### 5.1 Definition

Forecast each future period using the **value from the same period in the last season**:

```
ŷ(t+h) = y(T + h - m · ⌈h/m⌉)

Where m = seasonal period (e.g., 12 for monthly, 7 for daily)
```

More intuitively: the forecast for next January = last January's value, next Monday = last Monday's value, etc.

### 5.2 When It Makes Sense

- Any series with clear, stable seasonality
- Short-term forecasting of retail sales, energy demand, tourism
- **Always try this before SARIMA** — it often performs surprisingly well

```python
def seasonal_naive_forecast(train: pd.Series, h: int, m: int) -> pd.Series:
    """
    Forecast next h periods using last season's values.
    m = seasonal period (e.g., 12 for monthly, 7 for daily)
    """
    forecast_idx = pd.date_range(
        start=train.index[-1] + train.index.freq,
        periods=h,
        freq=train.index.freq,
    )
    # Repeat the last season's values cyclically
    last_season = train.iloc[-m:].values
    forecasts = [last_season[i % m] for i in range(h)]
    return pd.Series(forecasts, index=forecast_idx, name="seasonal_naive_forecast")
```

### 5.3 Prediction Intervals

```
95% PI: ŷ(t+h) ± 1.96 · σₘ · √(1 + ⌊(h-1)/m⌋)

Where σₘ = standard deviation of seasonal differences (y(t) - y(t-m))
```

---

## 6. Comparing Baselines

### 6.1 Evaluation Framework

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

def evaluate_forecast(actual: pd.Series, predicted: pd.Series, model_name: str) -> dict:
    """Compute standard forecast metrics."""
    errors = actual - predicted
    mae  = errors.abs().mean()
    rmse = np.sqrt((errors ** 2).mean())
    mape = (errors.abs() / actual.abs()).mean() * 100
    return {"model": model_name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

# Example usage
results = []
for name, fcst in [
    ("Mean", forecast_mean),
    ("Naive", forecast_naive),
    ("Drift", forecast_drift),
    ("Seasonal Naive", forecast_snaive),
]:
    results.append(evaluate_forecast(test_series, fcst, name))

leaderboard = pd.DataFrame(results).set_index("model").sort_values("RMSE")
print(leaderboard.round(3))
```

### 6.2 Skill Score (Relative Metric)

Measures how much better model A is vs. a baseline B:

```
Skill Score = 1 - (RMSE_model / RMSE_baseline)

Value > 0 → model beats the baseline
Value = 0 → same as baseline
Value < 0 → worse than baseline
```

```python
def skill_score(rmse_model: float, rmse_baseline: float) -> float:
    return 1 - (rmse_model / rmse_baseline)

ss = skill_score(rmse_model=42.3, rmse_baseline=58.7)
print(f"Skill score: {ss:.3f}")   # e.g., 0.280 = 28% improvement over baseline
```

### 6.3 Visual Comparison

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(train_series[-36:], color="gray", label="Train (last 3 years)", linewidth=1)
ax.plot(test_series, color="black", linewidth=2, label="Actual")

colors = ["#2C7BB6", "#D7191C", "#1A9641", "#F07D00"]
for (name, fcst), color in zip(
    [("Mean", forecast_mean), ("Naive", forecast_naive),
     ("Drift", forecast_drift), ("Seasonal Naive", forecast_snaive)],
    colors
):
    ax.plot(fcst, color=color, linewidth=1.8, linestyle="--", label=name)

ax.axvline(train_series.index[-1], color="black", linewidth=1, linestyle=":", alpha=0.5)
ax.legend()
ax.set_title("Baseline Model Comparison")
plt.tight_layout()
plt.show()
```

---

## 7. When Baselines Are Hard to Beat

### 7.1 Financial Prices (Random Walk)

Stock prices are famously close to a random walk in the short term. The naive forecast often wins because:
- Any signal in lagged prices is rapidly arbitraged away
- News (the main driver) is by definition unpredictable

### 7.2 Short Horizons with Strong Seasonality

For 1-step-ahead forecasting of highly seasonal series, seasonal naive often beats ARIMA and even ML models because:
- Last year's same-period value is the dominant signal
- Complex models add noise, not signal

### 7.3 Implication for Model Development

If your model barely beats seasonal naive → ask:
1. Is there genuinely predictable structure beyond seasonal patterns?
2. Am I evaluating correctly (no leakage)?
3. Do I have enough data for the model to learn?

> **Best practice**: Always report baseline metrics alongside your model metrics. A model with 95% MAPE "reduction" from a poorly chosen baseline is meaningless — always use seasonal naive as the benchmark.

---

*← [Module README](./README.md) | Next: [02 — ETS Models](./02_exponential_smoothing_ETS.md) →*
