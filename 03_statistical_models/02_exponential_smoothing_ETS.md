# 02 — Exponential Smoothing & ETS Models

> **Module**: 03 Statistical Models | **File**: 2 of 6
>
> Exponential smoothing is one of the most widely used and consistently accurate forecasting methods in industry. The ETS framework unifies the entire exponential smoothing family into a coherent state space formulation.

---

## Table of Contents

1. [Core Idea of Exponential Smoothing](#1-core-idea-of-exponential-smoothing)
2. [Simple Exponential Smoothing (SES)](#2-simple-exponential-smoothing-ses)
3. [Holt's Linear Trend Method](#3-holts-linear-trend-method)
4. [Holt-Winters Method (Triple Exponential Smoothing)](#4-holt-winters-method-triple-exponential-smoothing)
5. [The ETS Framework](#5-the-ets-framework)
6. [Automatic Model Selection](#6-automatic-model-selection)
7. [ETS vs. ARIMA](#7-ets-vs-arima)

---

## 1. Core Idea of Exponential Smoothing

### 1.1 The Problem with Simple Averages

A simple moving average gives **equal weight** to all past observations. But intuitively, the value from last month should matter more than the value from 5 years ago.

### 1.2 The Exponential Smoothing Principle

Exponential smoothing assigns **exponentially decreasing weights** to older observations:

```
Weights: α, α(1-α), α(1-α)², α(1-α)³, ...

Where 0 < α < 1 is the smoothing parameter

Weight at lag k = α(1-α)^k

Total sum: Σₖ₌₀^∞ α(1-α)^k = 1  (weights sum to 1)
```

| α value | Effect |
|---------|--------|
| α close to 1 | Recent observations dominate (fast-adapting) |
| α close to 0 | Old observations have significant influence (smooth, slow-adapting) |

---

## 2. Simple Exponential Smoothing (SES)

### 2.1 Use Case

SES is appropriate for series with:
- **No trend** (or very weak trend)
- **No seasonality**

### 2.2 Equations

**Smoothing equation (component form):**
```
Level:    L(t) = α · y(t) + (1-α) · L(t-1)
Forecast: ŷ(t+h) = L(t)    for all h ≥ 1
```

**Error correction form (equivalent):**
```
L(t) = L(t-1) + α · e(t)
Where e(t) = y(t) - L(t-1)  (one-step forecast error)
```

The level is updated by adding a fraction `α` of the most recent forecast error.

### 2.3 Choosing α

- **Estimated by minimizing SSE**: `α* = argmin Σₜ [y(t) - ŷ(t)]²`
- statsmodels optimizes this automatically

### 2.4 Implementation

```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Fit SES
model = SimpleExpSmoothing(train, initialization_method="estimated")
fitted = model.fit(optimized=True)   # auto-optimize alpha

print(f"Optimal α: {fitted.params['smoothing_level']:.4f}")

# Forecast
forecast_ses = fitted.forecast(steps=12)

# In-sample fitted values
fitted_values = fitted.fittedvalues
```

---

## 3. Holt's Linear Trend Method

### 3.1 Use Case

Holt's method extends SES to handle series with a **linear trend** (no seasonality).

### 3.2 Equations

```
Level:    L(t) = α · y(t) + (1-α) · [L(t-1) + T(t-1)]
Trend:    T(t) = β · [L(t) - L(t-1)] + (1-β) · T(t-1)
Forecast: ŷ(t+h) = L(t) + h · T(t)
```

Parameters:
- `α` (0 < α < 1): smoothing for the level
- `β` (0 < β < 1): smoothing for the trend

### 3.3 Damped Trend Variant

Long-run trends often **flatten out** rather than continuing indefinitely. The **damped trend** introduces a damping parameter `φ`:

```
Forecast: ŷ(t+h) = L(t) + (φ + φ² + ... + φʰ) · T(t)

φ = 1       → undamped (standard Holt's)
0 < φ < 1   → trend dampens toward a flat line
φ → 0       → trend disappears quickly
```

The **damped Holt** is typically the most accurate single-series forecasting method for trended data.

### 3.4 Implementation

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Standard Holt (additive trend, no seasonality)
model_holt = ExponentialSmoothing(
    train,
    trend="add",
    seasonal=None,
    initialization_method="estimated",
)
fitted_holt = model_holt.fit(optimized=True)
print(f"α={fitted_holt.params['smoothing_level']:.4f}, β={fitted_holt.params['smoothing_trend']:.4f}")

# Damped Holt
model_damped = ExponentialSmoothing(
    train,
    trend="add",
    damped_trend=True,
    seasonal=None,
    initialization_method="estimated",
)
fitted_damped = model_damped.fit(optimized=True)
print(f"φ (damping)={fitted_damped.params['damping_trend']:.4f}")

# Forecast
forecast_holt   = fitted_holt.forecast(steps=12)
forecast_damped = fitted_damped.forecast(steps=12)
```

---

## 4. Holt-Winters Method (Triple Exponential Smoothing)

### 4.1 Use Case

Holt-Winters extends Holt's method to handle both **trend and seasonality**. It is the industry standard for seasonal univariate forecasting when simplicity is needed.

### 4.2 Additive Holt-Winters

Use when seasonal variation is **constant** (same magnitude regardless of series level):

```
Level:    L(t) = α · [y(t) - S(t-m)] + (1-α) · [L(t-1) + T(t-1)]
Trend:    T(t) = β · [L(t) - L(t-1)] + (1-β) · T(t-1)
Seasonal: S(t) = γ · [y(t) - L(t)] + (1-γ) · S(t-m)
Forecast: ŷ(t+h) = L(t) + h · T(t) + S(t + h - m · ⌈h/m⌉)

Parameters: α (level), β (trend), γ (seasonal smoothing)
m = seasonal period
```

### 4.3 Multiplicative Holt-Winters

Use when seasonal variation **grows with the level** of the series:

```
Level:    L(t) = α · [y(t) / S(t-m)] + (1-α) · [L(t-1) + T(t-1)]
Trend:    T(t) = β · [L(t) - L(t-1)] + (1-β) · T(t-1)
Seasonal: S(t) = γ · [y(t) / L(t)] + (1-γ) · S(t-m)
Forecast: ŷ(t+h) = [L(t) + h · T(t)] · S(t + h - m · ⌈h/m⌉)
```

### 4.4 Implementation

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Additive Holt-Winters
model_add = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="add",
    seasonal_periods=12,           # monthly data: 12 periods per year
    initialization_method="estimated",
    damped_trend=True,             # often improves performance
)
fitted_add = model_add.fit(optimized=True)

# Multiplicative Holt-Winters (airline data is multiplicative)
model_mul = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="mul",
    seasonal_periods=12,
    initialization_method="estimated",
)
fitted_mul = model_mul.fit(optimized=True)

# Print learned parameters
print("Additive HW parameters:")
for k, v in fitted_add.params.items():
    print(f"  {k}: {v:.4f}")

# Forecast
h = 24
forecast_add = fitted_add.forecast(steps=h)
forecast_mul = fitted_mul.forecast(steps=h)

# Access decomposed components
print("\nLast estimated seasonal factors:")
print(fitted_mul.season[-12:])
```

### 4.5 Model Variants Summary

| Model | Trend | Seasonal | Typical Use |
|-------|-------|----------|-------------|
| SES | None | None | Stationary series |
| Holt | Additive | None | Trended, non-seasonal |
| Holt (Damped) | Damped | None | Trended, long horizon |
| Holt-Winters (Add) | Additive | Additive | Constant seasonal swings |
| Holt-Winters (Mul) | Additive | Multiplicative | Growing seasonal swings |

---

## 5. The ETS Framework

### 5.1 What is ETS?

ETS stands for **Error–Trend–Seasonality**. It is a general state space framework that unifies all exponential smoothing methods.

Each component can be:
- **E (Error)**: `A` (Additive) or `M` (Multiplicative)
- **T (Trend)**: `N` (None), `A` (Additive), `Ad` (Additive Damped), or `M` (Multiplicative)
- **S (Seasonal)**: `N` (None), `A` (Additive), or `M` (Multiplicative)

This gives **30 possible combinations** (some are numerically unstable and excluded).

### 5.2 Common ETS Models

| ETS Model | Equivalent To | Use Case |
|-----------|-------------|----------|
| ETS(A,N,N) | SES | No trend, no seasonality |
| ETS(A,A,N) | Holt's linear | Additive trend |
| ETS(A,Ad,N) | Damped Holt | Damped additive trend |
| ETS(A,A,A) | Additive Holt-Winters | Additive trend + seasonality |
| ETS(M,A,M) | Multiplicative Holt-Winters | Multiplicative seasonality |
| ETS(M,Ad,M) | Damped Multiplicative HW | Most common production choice |

### 5.3 State Space Form

The ETS framework allows each model to be written as a state space model:

```
Measurement equation: y(t) = h(x_{t-1}) + k(x_{t-1}) · ε(t)
State equation:       x(t) = f(x_{t-1}) + g(x_{t-1}) · ε(t)

Where:
  x(t) = state vector (level, trend, seasonal factors)
  ε(t) ~ NID(0, σ²)  (Gaussian white noise)
  h, k, f, g = model-specific functions
```

Benefits of the state space form:
- Provides **prediction intervals** naturally
- Enables **likelihood-based parameter estimation**
- Allows formal **model selection** via information criteria (AIC/AICc/BIC)

### 5.4 Automatic ETS in Python

```python
# Using statsforecast for automatic ETS selection (fast)
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

sf = StatsForecast(
    models=[AutoETS(season_length=12, model="ZZZ")],   # ZZZ = auto-select all components
    freq="MS",
)
sf.fit(df_sf)   # df_sf must have columns: unique_id, ds, y
forecast_ets = sf.predict(h=12)

# Or using statsmodels (more control)
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
model = ETSModel(train, error="mul", trend="add", damped_trend=True,
                 seasonal="mul", seasonal_periods=12)
fitted = model.fit(disp=False)
print(fitted.summary())
```

---

## 6. Automatic Model Selection

ETS model selection is done by minimizing the **AICc** (corrected AIC):

```
AICc = AIC + 2k(k+1)/(n-k-1)

Where:
  AIC = -2·log(L) + 2k
  L   = maximum likelihood
  k   = number of parameters
  n   = number of observations
```

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

best_aic = np.inf
best_model = None

configs = [
    ("add", None, False),
    ("add", "add", False),
    ("add", "mul", False),
    ("add", "add", True),    # damped
    ("add", "mul", True),    # damped
]

for trend, seasonal, damped in configs:
    try:
        m = ExponentialSmoothing(
            train, trend=trend, seasonal=seasonal,
            seasonal_periods=12, damped_trend=damped,
            initialization_method="estimated"
        ).fit(optimized=True, disp=False)
        if m.aic < best_aic:
            best_aic = m.aic
            best_model = m
            best_config = (trend, seasonal, damped)
    except Exception:
        continue

print(f"Best ETS config: trend={best_config[0]}, seasonal={best_config[1]}, damped={best_config[2]}")
print(f"Best AIC: {best_aic:.2f}")
```

---

## 7. ETS vs. ARIMA

| Aspect | ETS | ARIMA |
|--------|-----|-------|
| **Interpretability** | High — level/trend/seasonal components | Moderate — AR/MA operators |
| **Automatic selection** | Easy — 30 models via AICc | Harder — requires ACF/PACF reading or auto_arima |
| **Seasonal handling** | Built-in and intuitive | Requires seasonal differencing + seasonal AR/MA |
| **Multiple seasonality** | ❌ (use TBATS or Prophet) | ❌ (use SARIMA with approximate period) |
| **Prediction intervals** | ✅ Exact from state space | ✅ Exact from likelihood |
| **External regressors** | ❌ (use ARIMAX/SARIMAX) | ✅ (ARIMAX/SARIMAX) |
| **Empirical accuracy** | Often better on short series | Often better on long stationary series |
| **Benchmark** | M-competitions winner historically | Strong for economic data |

> **Practical recommendation**: Always try both ETS and ARIMA. For seasonal monthly data, ETS(M,Ad,M) (multiplicative damped) is often the best starting point.

---

*← [01 — Naive Baselines](./01_naive_baseline_models.md) | [Module README](./README.md) | Next: [03 — AR → SARIMA Family](./03_ar_ma_arma_arima_sarima.md) →*
