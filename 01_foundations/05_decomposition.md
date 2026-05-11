# 05 — Time Series Decomposition

> **Module**: 01 Foundations | **File**: 5 of 5
>
> Decomposition separates a time series into interpretable components — trend, seasonality, and remainder. It is the foundation of classical modeling and a powerful preprocessing step for modern methods.

---

## Table of Contents

1. [Why Decompose?](#1-why-decompose)
2. [Classical Decomposition](#2-classical-decomposition)
3. [STL Decomposition](#3-stl-decomposition)
4. [MSTL — Multiple Seasonality](#4-mstl--multiple-seasonality)
5. [Measuring Component Strength](#5-measuring-component-strength)
6. [Deseasonalized Forecasting Workflow](#6-deseasonalized-forecasting-workflow)
7. [Choosing the Right Decomposition Method](#7-choosing-the-right-decomposition-method)

---

## 1. Why Decompose?

Decomposition serves three distinct purposes:

| Purpose | Description |
|---------|-------------|
| **Understanding** | Visualize the trend direction and seasonal cycle of the data |
| **Diagnostics** | Check if the remainder looks like white noise — if not, signal is being missed |
| **Preprocessing** | Remove trend/seasonality before modeling; add back after forecasting |

```
Before decomposition: Y(t) = complex, intertwined patterns
After decomposition:  Y(t) = T(t) + S(t) + R(t)  ← each piece is interpretable
```

---

## 2. Classical Decomposition

### 2.1 How It Works

Classical decomposition uses a **centered moving average (CMA)** to estimate the trend, then derives the seasonal component by averaging.

#### Steps for Additive Decomposition:

```
Step 1: Estimate trend T(t) using a centered moving average of order m
        For odd m:  T(t) = (1/m) Σᵢ₌₋(m-1)/2 to (m-1)/2  Y(t+i)
        For even m: Use a 2×m-MA (average of two overlapping m-MAs)
        Typically m = s (seasonal period) or m = 2s for even periods

Step 2: Detrend the series
        D(t) = Y(t) - T(t)

Step 3: Estimate seasonal component S(t)
        Average D(t) values for each season position
        e.g., average all January values, all February values, ...
        Then adjust so that seasonal factors sum to zero

Step 4: Compute remainder
        R(t) = Y(t) - T(t) - S(t)
```

### 2.2 Python Implementation

```python
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Additive decomposition
result_add = seasonal_decompose(series, model='additive', period=12)
result_add.plot()
plt.suptitle('Classical Additive Decomposition')
plt.tight_layout()
plt.show()

# Multiplicative decomposition
result_mul = seasonal_decompose(series, model='multiplicative', period=12)
result_mul.plot()
plt.suptitle('Classical Multiplicative Decomposition')
plt.tight_layout()
plt.show()

# Access individual components
trend    = result_add.trend       # pandas Series, NaN at edges
seasonal = result_add.seasonal    # pandas Series
residual = result_add.resid       # pandas Series
```

### 2.3 Limitations of Classical Decomposition

| Limitation | Impact |
|-----------|--------|
| **Edge NaNs** | First and last `⌊m/2⌋` observations lose trend estimates |
| **Fixed seasonality** | Assumes seasonal pattern is constant over time — not true in practice |
| **Not robust to outliers** | A single outlier contaminates the moving average window |
| **Symmetric window** | Future values leak into the trend estimate for current period |
| **Single seasonality only** | Cannot handle multiple seasonal periods |

> **Bottom line**: Classical decomposition is useful for teaching but **STL should be used in all practical work**.

---

## 3. STL Decomposition

**STL = Seasonal and Trend decomposition using LOESS** (locally estimated scatterplot smoothing).

STL is the modern, robust, and flexible decomposition method used in production systems.

### 3.1 Key Advantages Over Classical

| Feature | Classical | STL |
|---------|-----------|-----|
| Seasonal pattern can change over time | ❌ No | ✅ Yes |
| Works with any period `s` | Limited | ✅ Yes |
| Robust to outliers | ❌ No | ✅ Yes (`robust=True`) |
| No edge NaN problem | ❌ Yes, NaNs at edges | ✅ No NaNs |
| Handles multiple seasonality | ❌ No | ⚠️ Partial (use MSTL) |
| Missing values | ❌ No | ❌ No (interpolate first) |

### 3.2 How STL Works (Conceptually)

```
STL uses an iterative algorithm (inner loop + outer loop):

Inner Loop (repeated n_inner times):
  1. Compute trend T(t) using LOESS smoothing on de-seasonalized series
  2. Remove trend from original: D(t) = Y(t) - T(t)
  3. For each seasonal sub-series (e.g., all January values),
     fit a LOESS smoother to capture changing seasonality
  4. Average sub-series smoothers to form seasonal component S(t)
  5. Repeat

Outer Loop (repeated n_outer times, only if robust=True):
  6. Compute robustness weights: down-weight outliers based on residual size
  7. Re-run inner loop with these weights
```

### 3.3 Python Implementation

```python
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

stl = STL(
    series,
    period=12,          # seasonal period (required)
    seasonal=13,        # smoothness of seasonal — must be odd, >= 7
    trend=None,         # auto-computed as smallest odd > 1.5 * period / (1 - 1.5/seasonal)
    robust=True,        # use robust LOESS to downweight outliers
)
result = stl.fit()

# Plot all components
result.plot()
plt.suptitle('STL Decomposition')
plt.tight_layout()
plt.show()

# Access components
trend    = result.trend     # pandas Series
seasonal = result.seasonal  # pandas Series
residual = result.resid     # pandas Series
```

### 3.4 STL Parameters Explained

| Parameter | Type | Description | Recommendation |
|-----------|------|-------------|----------------|
| `period` | int | Seasonal period (e.g., 12 for monthly, 7 for daily) | **Required** — set correctly |
| `seasonal` | int (odd) | Controls how much the seasonal component is allowed to change | 7 (fast change) to 51 (slow change) |
| `trend` | int (odd) | Controls trend smoothness — larger = smoother | Auto-computed is usually fine |
| `robust` | bool | Use robust LOESS — protects against outlier distortion | `True` for noisy real-world data |
| `low_pass` | int | Low-pass filter window — controls interaction between trend and seasonal | Auto-computed |

### 3.5 Tuning STL

```python
# More stable (seasonal changes slowly):
stl_stable = STL(series, period=12, seasonal=51, robust=True)

# More responsive (seasonal changes quickly):
stl_responsive = STL(series, period=12, seasonal=7, robust=True)

# Visual comparison
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
for stl_obj, label in [(stl_stable, 'seasonal=51'), (stl_responsive, 'seasonal=7')]:
    r = stl_obj.fit()
    axes[0].plot(r.trend, label=label)
    axes[1].plot(r.seasonal, label=label)
    axes[2].plot(r.resid, label=label)
for ax, title in zip(axes, ['Trend', 'Seasonal', 'Residual']):
    ax.set_title(title)
    ax.legend()
plt.tight_layout()
plt.show()
```

---

## 4. MSTL — Multiple Seasonality

When a series has **more than one seasonal period**, use **MSTL** (Multiple STL):

### 4.1 When to Use MSTL

| Data Frequency | Multiple Seasonality | Periods |
|---------------|---------------------|---------|
| Hourly | Daily + Weekly | `[24, 168]` |
| 30-minute | Daily + Weekly | `[48, 336]` |
| Daily | Weekly + Yearly | `[7, 365]` |
| Monthly | Yearly (only one) | `[12]` — use regular STL |

### 4.2 Python Implementation

```python
from statsmodels.tsa.seasonal import MSTL

# Hourly data with daily + weekly seasonality
mstl = MSTL(
    series_hourly,
    periods=[24, 168],    # daily (24h) and weekly (168h)
)
result = mstl.fit()

# Plot
result.plot()
plt.tight_layout()
plt.show()

# Access seasonal components separately
daily_seasonal  = result.seasonal['seasonal_24']
weekly_seasonal = result.seasonal['seasonal_168']
trend = result.trend
resid = result.resid
```

---

## 5. Measuring Component Strength

A useful quantification of **how dominant** each component is in the series. Defined in Wang, Smith & Hyndman (2006):

### 5.1 Formulas

```
Trend Strength:
  F_T = max(0, 1 - Var(R) / Var(T + R))

Seasonal Strength:
  F_S = max(0, 1 - Var(R) / Var(S + R))

Range: [0, 1]
  Close to 1  →  Component is dominant (explains most variation)
  Close to 0  →  Component is weak or essentially absent
```

### 5.2 Python Implementation

```python
import numpy as np
from statsmodels.tsa.seasonal import STL

result = STL(series, period=12, robust=True).fit()

var_R  = np.var(result.resid, ddof=1)
var_TR = np.var(result.trend + result.resid, ddof=1)
var_SR = np.var(result.seasonal + result.resid, ddof=1)

trend_strength    = max(0, 1 - var_R / var_TR)
seasonal_strength = max(0, 1 - var_R / var_SR)

print(f"Trend strength    : {trend_strength:.3f}")
print(f"Seasonal strength : {seasonal_strength:.3f}")

# Interpretation:
# trend_strength > 0.6    → strong trend, consider detrending or ARIMA d>0
# seasonal_strength > 0.6 → strong seasonality, consider SARIMA or seasonal differencing
```

### 5.3 Interpretation Guide

| Strength Value | Interpretation | Modeling Implication |
|---------------|---------------|----------------------|
| > 0.8 | Very strong | This component dominates — must model it explicitly |
| 0.5–0.8 | Moderate | Include this component in the model |
| 0.2–0.5 | Weak | May or may not be worth modeling |
| < 0.2 | Very weak or absent | Probably safe to ignore this component |

---

## 6. Deseasonalized Forecasting Workflow

A common and robust production pattern:

### 6.1 The Workflow

```
1. Decompose:          Y(t) = T(t) + S(t) + R(t)  using STL
2. Remove seasonality: Y_ds(t) = Y(t) - S(t)       (deseasonalized series)
3. Forecast:           Ŷ_ds(t+h)                    using any model (ARIMA, ML, DL)
4. Re-add seasonality: Ŷ(t+h) = Ŷ_ds(t+h) + S(t+h) (use seasonal component from decomposition)
```

### 6.2 Python Implementation

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Decompose
stl = STL(series, period=12, robust=True)
result = stl.fit()
seasonal_component = result.seasonal

# Step 2: Deseasonalize
series_deseasonalized = series - seasonal_component

# Step 3: Forecast the deseasonalized series
model = ARIMA(series_deseasonalized, order=(1, 1, 1))
fitted = model.fit()
forecast_deseasonalized = fitted.forecast(steps=12)

# Step 4: Re-add seasonal component
# Use the seasonal pattern from the last observed year
last_year_seasonal = seasonal_component[-12:].values
forecast_with_seasonality = forecast_deseasonalized.values + last_year_seasonal

# Wrap result in a pandas Series with correct index
forecast_index = pd.date_range(
    start=series.index[-1] + pd.DateOffset(months=1),
    periods=12,
    freq='MS'
)
final_forecast = pd.Series(forecast_with_seasonality, index=forecast_index)
print(final_forecast)
```

### 6.3 Advantages of This Approach

1. **Stability** — Separating seasonality reduces complexity for the forecasting model
2. **Flexibility** — Any model (ML, DL, statistical) can be used for the deseasonalized forecast
3. **Interpretability** — Easy to explain trend forecast vs. seasonal adjustment separately
4. **Robustness** — STL's robustness protects the seasonal estimate from outliers

---

## 7. Choosing the Right Decomposition Method

| Situation | Recommended Method |
|-----------|-------------------|
| Teaching / quick exploration | Classical `seasonal_decompose` |
| Standard production use (single seasonality) | **STL** with `robust=True` |
| Multiple seasonal periods | **MSTL** |
| Sub-daily data with complex patterns | MSTL or Prophet decomposition |
| Outliers present | **STL** with `robust=True` |
| Need trend and seasonal to be separable | STL (more control via parameters) |
| Non-integer or approximate period | STL (more flexible than classical) |

```python
# Quick decision function
def choose_decomposition(series, period, has_multiple_seasonality=False, has_outliers=True):
    if has_multiple_seasonality:
        print("→ Use MSTL with periods=[...]")
    elif has_outliers:
        print(f"→ Use STL(period={period}, robust=True)")
    else:
        print(f"→ Use STL(period={period}, robust=False)  or  seasonal_decompose(period={period})")
```

---

*← [04 — ACF & PACF](./04_autocorrelation_acf_pacf.md) | [Module README](./README.md) | Next Module: [02 — Data Engineering](../02_data_engineering/README.md) →*
