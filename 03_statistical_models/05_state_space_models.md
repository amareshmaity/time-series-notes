# 05 — State Space Models & Kalman Filter

> **Module**: 03 Statistical Models | **File**: 5 of 6
>
> State Space Models (SSM) provide a unified, flexible framework that subsumes ARIMA, ETS, and many other models. The Kalman filter is the algorithm that makes inference tractable.

---

## Table of Contents

1. [State Space Representation](#1-state-space-representation)
2. [The Kalman Filter](#2-the-kalman-filter)
3. [Local Level Model](#3-local-level-model)
4. [Local Linear Trend Model](#4-local-linear-trend-model)
5. [Structural Time Series Models](#5-structural-time-series-models)
6. [ARIMA as a State Space Model](#6-arima-as-a-state-space-model)
7. [Implementation with statsmodels](#7-implementation-with-statsmodels)

---

## 1. State Space Representation

### 1.1 The General Form

A **State Space Model** separates the observed series from an underlying unobserved state:

```
Observation equation:  y(t) = Z·α(t) + d + ε(t),    ε(t) ~ N(0, H)
State equation:        α(t) = T·α(t-1) + c + R·η(t), η(t) ~ N(0, Q)

Where:
  y(t)   = observed measurement (scalar or vector)
  α(t)   = latent state vector (unobserved)
  Z      = observation matrix (links state to measurement)
  T      = state transition matrix (dynamics of state evolution)
  d, c   = intercept/offset terms
  R      = selection matrix
  H      = observation noise covariance
  Q      = state noise covariance
  ε(t)   = observation noise
  η(t)   = state noise (process noise)
```

### 1.2 Key Properties

| Property | Explanation |
|----------|-------------|
| **Latent state** | The "true" underlying level/trend we cannot observe directly |
| **Observation noise** | Measurement error in what we record |
| **State noise** | Random innovations that change the underlying state |
| **Kalman filter** | Algorithm to estimate α(t) from y(1), ..., y(t) |
| **Flexible** | Can represent ARIMA, ETS, regression, structural breaks |

---

## 2. The Kalman Filter

### 2.1 Purpose

The **Kalman filter** is a recursive algorithm that provides the **optimal linear estimate** of the latent state `α(t)` given all observations up to time `t`:

```
α̂(t|t) = E[α(t) | y(1), y(2), ..., y(t)]   (filtered state)
α̂(t|t-1) = E[α(t) | y(1), y(2), ..., y(t-1)]  (predicted state)
```

### 2.2 The Two-Step Algorithm

**Predict step** (before observing y(t)):
```
α̂(t|t-1) = T · α̂(t-1|t-1) + c              (predicted state)
P(t|t-1)  = T · P(t-1|t-1) · Tᵀ + R·Q·Rᵀ   (predicted covariance)
```

**Update step** (after observing y(t)):
```
v(t) = y(t) - Z · α̂(t|t-1) - d             (innovation / forecast error)
F(t) = Z · P(t|t-1) · Zᵀ + H              (innovation variance)
K(t) = P(t|t-1) · Zᵀ · F(t)⁻¹            (Kalman gain)

α̂(t|t) = α̂(t|t-1) + K(t) · v(t)          (updated state)
P(t|t)  = (I - K(t) · Z) · P(t|t-1)      (updated covariance)
```

### 2.3 Intuition for Kalman Gain

```
K(t) = [Uncertainty in state] / [Uncertainty in measurement]

K → 1:  Trust the measurement more than the model prediction
K → 0:  Trust the model prediction more than the measurement
```

This is why the Kalman filter is robust to missing data — missing observations simply skip the update step.

---

## 3. Local Level Model

### 3.1 Definition

The simplest meaningful state space model — also known as the **random walk plus noise** model:

```
Observation: y(t) = μ(t) + ε(t),  ε(t) ~ N(0, σ²_ε)
State:        μ(t) = μ(t-1) + η(t), η(t) ~ N(0, σ²_η)

Where:
  μ(t) = level (latent state — random walk)
  ε(t) = irregular/noise component
  η(t) = level disturbance (how much the level can change each period)
```

### 3.2 Signal-to-Noise Ratio

```
q = σ²_η / σ²_ε   (signal-to-noise ratio)

q → 0: Level barely changes → forecast ≈ mean forecast (lots of smoothing)
q → ∞: Level changes rapidly → forecast ≈ naive forecast (no smoothing)
```

The Kalman gain `K` equals `q / (1 + q)` at steady state — exactly the SES smoothing parameter `α`!

> **The Local Level Model IS Simple Exponential Smoothing in state space form.**

### 3.3 Implementation

```python
from statsmodels.tsa.statespace.structural import UnobservedComponents

# Local level model (random walk + noise)
model = UnobservedComponents(
    train,
    level="local level",   # μ(t) = μ(t-1) + η
)
fitted = model.fit(disp=False)
print(fitted.summary())

# Filtered state (estimated level)
filtered_level = fitted.filtered_state[0]

# Smooth state (using all data — more accurate, but can't be used in real-time)
smoothed_level = fitted.smoothed_state[0]

# Forecast
forecast = fitted.get_forecast(steps=12)
print(f"12-step forecast: {forecast.predicted_mean.values}")
```

---

## 4. Local Linear Trend Model

### 4.1 Definition

Extends the local level by adding a **stochastic trend** component:

```
Observation: y(t)  = μ(t) + ε(t),   ε(t) ~ N(0, σ²_ε)
Level:       μ(t)  = μ(t-1) + β(t-1) + η(t),  η(t) ~ N(0, σ²_η)
Trend:       β(t)  = β(t-1) + ζ(t),            ζ(t) ~ N(0, σ²_ζ)
```

Both the level AND the trend can drift randomly over time.

**Special cases:**

| σ²_η | σ²_ζ | Equivalent |
|------|------|-----------|
| > 0 | > 0 | Full local linear trend |
| 0 | > 0 | Smooth trend model |
| > 0 | 0 | Random walk trend (Holt's linear) |
| 0 | 0 | Fixed linear trend |

### 4.2 Implementation

```python
# Local linear trend
model_llt = UnobservedComponents(
    train,
    level="local linear trend",   # level + stochastic slope
)
fitted_llt = model_llt.fit(disp=False)

# Decompose into components
level  = fitted_llt.smoothed_state[0]
trend  = fitted_llt.smoothed_state[1]
```

---

## 5. Structural Time Series Models

### 5.1 Full Structural Model

A **Structural Time Series (STS)** model decomposes the series into interpretable components:

```
y(t) = μ(t) + γ(t) + c(t) + ε(t)

Where:
  μ(t) = trend component  (local level or linear trend)
  γ(t) = seasonal component (stochastic or deterministic)
  c(t) = cyclical component (optional)
  ε(t) = irregular component
```

### 5.2 Implementation with Structural Components

```python
from statsmodels.tsa.statespace.structural import UnobservedComponents

# Full structural model: trend + stochastic seasonal
model_structural = UnobservedComponents(
    train,
    level="local linear trend",
    seasonal=12,               # stochastic seasonal with period 12
    freq_seasonal=None,        # use trigonometric seasonal instead
)
fitted_structural = model_structural.fit(disp=False)
print(fitted_structural.summary())

# Access decomposed components
smoothed = fitted_structural.smoothed_state
level_component    = smoothed[0]
trend_component    = smoothed[1]
seasonal_component = smoothed[2:14].sum(axis=0)   # sum seasonal harmonics

# Plot decomposition
import matplotlib.pyplot as plt
fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)

axes[0].plot(train, color="#2C7BB6", label="Observed", linewidth=1)
axes[0].plot(fitted_structural.fittedvalues, color="#D7191C",
             label="Fitted", linewidth=1.5, linestyle="--")
axes[0].set_title("Observed vs. Fitted")
axes[0].legend()

axes[1].plot(level_component, color="#D7191C", label="Level")
axes[1].set_title("Level Component")

axes[2].plot(trend_component, color="#1A9641", label="Trend")
axes[2].set_title("Trend Component")

axes[3].plot(train.index, seasonal_component, color="#F07D00", label="Seasonal")
axes[3].axhline(0, color="black", linewidth=0.5, linestyle="--")
axes[3].set_title("Seasonal Component")

plt.suptitle("Structural Time Series Decomposition", fontweight="bold")
plt.tight_layout()
plt.show()
```

---

## 6. ARIMA as a State Space Model

Every ARIMA model can be written in state space form, which enables:
- Exact likelihood computation via the Kalman filter
- Handling of missing values naturally
- Extension with regression components (ARIMAX/SARIMAX)

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA in state space form (this is how statsmodels always fits ARIMA)
model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    trend="c",
)
fitted = model.fit(disp=False)

# Access state space internals
print(f"State space matrices:")
print(f"  Observation matrix Z: {fitted.model.design}")
print(f"  Transition matrix T:  {fitted.model.transition}")
```

---

## 7. Implementation with statsmodels

### 7.1 Model Comparison

```python
import pandas as pd
import numpy as np

models_to_compare = {
    "Local Level": UnobservedComponents(train, level="local level"),
    "Local Linear Trend": UnobservedComponents(train, level="local linear trend"),
    "LLT + Seasonal": UnobservedComponents(train, level="local linear trend", seasonal=12),
}

results = []
for name, model in models_to_compare.items():
    fitted = model.fit(disp=False)
    forecast = fitted.get_forecast(steps=len(test)).predicted_mean
    rmse = np.sqrt(((test.values - forecast.values) ** 2).mean())
    results.append({
        "Model": name, "AIC": fitted.aic, "BIC": fitted.bic, "Test RMSE": rmse
    })

leaderboard = pd.DataFrame(results).set_index("Model")
print(leaderboard.round(3))
```

### 7.2 Handling Missing Values

State space models handle missing values automatically — simply set them to `NaN`:

```python
series_with_gaps = train.copy()
series_with_gaps.iloc[10:15] = np.nan   # introduce gaps

model = UnobservedComponents(series_with_gaps, level="local level")
fitted = model.fit(disp=False)
# Kalman filter automatically handles NaN — no imputation needed!

# The smoothed state estimates the "true" value even at missing periods
print("Estimated level at missing periods:")
print(fitted.smoothed_state[0, 10:15])
```

---

*← [04 — VAR](./04_var_vector_autoregression.md) | [Module README](./README.md) | Next: [06 — Model Selection & Diagnostics](./06_model_selection_and_diagnostics.md) →*
