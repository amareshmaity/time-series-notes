# 03 — AR, MA, ARMA, ARIMA & SARIMA

> **Module**: 03 Statistical Models | **File**: 3 of 6
>
> The ARIMA family is the cornerstone of classical time series analysis. This file walks through the entire model family from first principles — AR → MA → ARMA → ARIMA → SARIMA — with mathematical derivations, identification rules, and implementation.

---

## Table of Contents

1. [The Building Blocks](#1-the-building-blocks)
2. [AR(p) — AutoRegressive Model](#2-arp--autoregressive-model)
3. [MA(q) — Moving Average Model](#3-maq--moving-average-model)
4. [ARMA(p,q)](#4-armapq)
5. [ARIMA(p,d,q)](#5-arimapdq)
6. [SARIMA(p,d,q)(P,D,Q,s)](#6-sarimapdqpdqs)
7. [Box-Jenkins Methodology](#7-box-jenkins-methodology)
8. [auto_arima — Automated Order Selection](#8-auto_arima--automated-order-selection)
9. [Forecasting and Prediction Intervals](#9-forecasting-and-prediction-intervals)
10. [ARIMAX / SARIMAX — Exogenous Regressors](#10-arimax--sarimax--exogenous-regressors)

---

## 1. The Building Blocks

### 1.1 White Noise

```
ε(t) ~ WN(0, σ²)

E[ε(t)] = 0
Var[ε(t)] = σ²
Cov[ε(t), ε(s)] = 0  for t ≠ s
```

White noise is completely unpredictable — it is the fundamental irreducible component.

### 1.2 Backshift (Lag) Operator B

The backshift operator `B` shifts the series back by one period:

```
B · y(t) = y(t-1)
B² · y(t) = y(t-2)
Bᵏ · y(t) = y(t-k)
```

This notation makes ARIMA equations compact:

```
(1-B) · y(t) = y(t) - y(t-1) = Δy(t)   (first difference)
(1-Bˢ) · y(t) = y(t) - y(t-s)           (seasonal difference)
```

### 1.3 The Model Family Tree

```
White Noise ε(t)
     │
     ├── Add autoregressive terms (past y values)
     │         ↓
     │       AR(p)
     │
     ├── Add moving average terms (past ε values)
     │         ↓
     │       MA(q)
     │
     ├── Combine AR + MA
     │         ↓
     │      ARMA(p,q)   ← assumes stationarity
     │
     ├── Add differencing to handle non-stationarity
     │         ↓
     │    ARIMA(p,d,q)
     │
     └── Add seasonal components
               ↓
         SARIMA(p,d,q)(P,D,Q,s)
```

---

## 2. AR(p) — AutoRegressive Model

### 2.1 Definition

An **AR(p)** model expresses the current value as a **linear combination of its own past `p` values** plus white noise:

```
y(t) = c + φ₁·y(t-1) + φ₂·y(t-2) + ... + φₚ·y(t-p) + ε(t)

Using backshift operator:
  φ(B) · y(t) = c + ε(t)
  Where φ(B) = 1 - φ₁B - φ₂B² - ... - φₚBᵖ
```

Parameters:
- `c` — constant (intercept)
- `φ₁, ..., φₚ` — autoregressive coefficients
- `ε(t)` — white noise

### 2.2 Stationarity Condition

An AR(p) process is stationary if and only if **all roots of the characteristic polynomial lie outside the unit circle**:

```
φ(z) = 1 - φ₁z - φ₂z² - ... - φₚzᵖ = 0  →  |z| > 1 for all roots

AR(1) stationarity condition: |φ₁| < 1
AR(2) stationarity condition: φ₁ + φ₂ < 1, φ₂ - φ₁ < 1, |φ₂| < 1
```

### 2.3 ACF/PACF Signature for AR(p)

| | AR(p) |
|--|-------|
| **ACF** | Tails off — exponential or oscillating decay |
| **PACF** | Cuts off sharply after lag p |

```
Example AR(1) with φ₁ = 0.8:
  ACF:  ρ(k) = φ₁^k = 0.8^k  → exponential decay
  PACF: spike at lag 1, near zero for lags 2, 3, ...
```

### 2.4 Interpretation

- `φ₁ > 0` → positive autocorrelation (persistence): high values follow high values
- `φ₁ < 0` → negative autocorrelation (mean-reversion): alternating high-low pattern
- `φ₁ = 1` → random walk (non-stationary)

### 2.5 Implementation

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit AR(2) model
model_ar2 = ARIMA(train, order=(2, 0, 0))
fitted_ar2 = model_ar2.fit()
print(fitted_ar2.summary())

# AR coefficients
phi1, phi2 = fitted_ar2.params[1], fitted_ar2.params[2]
print(f"φ₁={phi1:.4f}, φ₂={phi2:.4f}")
```

---

## 3. MA(q) — Moving Average Model

### 3.1 Definition

An **MA(q)** model expresses the current value as a **linear combination of the current and past `q` white noise errors**:

```
y(t) = μ + ε(t) + θ₁·ε(t-1) + θ₂·ε(t-2) + ... + θq·ε(t-q)

Using backshift operator:
  y(t) = μ + θ(B) · ε(t)
  Where θ(B) = 1 + θ₁B + θ₂B² + ... + θqBᵠ
```

> ⚠️ **Important**: The MA model is about **past errors** (residuals), NOT past values of `y`. This is a key conceptual difference from AR.

### 3.2 Properties

| Property | Value |
|----------|-------|
| Always stationary? | ✅ Yes — MA processes are always stationary |
| Invertibility condition | Roots of θ(z) = 0 must lie outside unit circle |
| Memory | Finite — only `q` periods of error history matter |

### 3.3 ACF/PACF Signature for MA(q)

| | MA(q) |
|--|-------|
| **ACF** | Cuts off sharply after lag q |
| **PACF** | Tails off — exponential or oscillating decay |

```
Example MA(1) with θ₁ = 0.6:
  ACF:  ρ(1) = θ₁/(1 + θ₁²), ρ(k) = 0 for k > 1  → cuts off at lag 1
  PACF: Alternating exponential decay
```

### 3.4 Interpretation

- `θ₁ > 0` → over-differencing type behavior — positive shock in period t gives positive effect in t+1
- `θ₁ < 0` → negative correction — positive shock followed by negative correction

### 3.5 Implementation

```python
# Fit MA(2) model
model_ma2 = ARIMA(train, order=(0, 0, 2))
fitted_ma2 = model_ma2.fit()
print(fitted_ma2.summary())

theta1, theta2 = fitted_ma2.params[1], fitted_ma2.params[2]
print(f"θ₁={theta1:.4f}, θ₂={theta2:.4f}")
```

---

## 4. ARMA(p,q)

### 4.1 Definition

**ARMA(p,q)** combines both AR and MA components:

```
y(t) = c + φ₁·y(t-1) + ... + φₚ·y(t-p) + ε(t) + θ₁·ε(t-1) + ... + θq·ε(t-q)

Compact form:  φ(B)·y(t) = c + θ(B)·ε(t)
```

### 4.2 When to Use ARMA

- Series is **already stationary** (no differencing needed)
- Both ACF and PACF **tail off** (neither cuts off cleanly → ARMA, not pure AR or MA)

### 4.3 ACF/PACF Signature

| | ARMA(p,q) |
|--|-----------|
| **ACF** | Tails off after lag q |
| **PACF** | Tails off after lag p |

When both ACF and PACF tail off, manual order identification is difficult. Use **AIC/BIC** to compare candidate models.

### 4.4 Implementation

```python
# Fit ARMA(1,1) — equivalent to ARIMA(1,0,1)
model_arma = ARIMA(train, order=(1, 0, 1))
fitted_arma = model_arma.fit()
print(fitted_arma.summary())
```

---

## 5. ARIMA(p,d,q)

### 5.1 Definition

**ARIMA** = **I**ntegrated ARMA — adds `d` rounds of differencing to handle non-stationarity:

```
ARIMA(p, d, q) means:
  1. Take d differences of y(t) to make it stationary:  Δᵈy(t)
  2. Fit an ARMA(p,q) to the differenced series

Full equation:
  φ(B) · (1-B)ᵈ · y(t) = c + θ(B) · ε(t)

Where:
  (1-B)ᵈ  = d-th differencing operator
  d=1: Δy(t) = y(t) - y(t-1)
  d=2: Δ²y(t) = Δy(t) - Δy(t-1)
```

### 5.2 Parameter Meaning

| Parameter | Role | How to Determine |
|-----------|------|-----------------|
| `p` | AR order — how many past values to use | PACF: number of significant spikes |
| `d` | Differencing order — to achieve stationarity | ADF + KPSS tests |
| `q` | MA order — how many past errors to use | ACF: number of significant spikes |

### 5.3 Special Cases

| ARIMA Model | Equivalent |
|-------------|-----------|
| ARIMA(0,0,0) | White noise |
| ARIMA(0,1,0) | Random walk |
| ARIMA(0,1,0) + c | Random walk with drift |
| ARIMA(1,0,0) | AR(1) |
| ARIMA(0,0,1) | MA(1) |
| ARIMA(p,1,0) | Differenced AR(p) |

### 5.4 Implementation

```python
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Fit ARIMA(1,1,1)
model = ARIMA(train, order=(1, 1, 1), trend="c")
fitted = model.fit()

print(fitted.summary())
print(f"\nAIC: {fitted.aic:.2f}")
print(f"BIC: {fitted.bic:.2f}")

# Forecast
forecast_result = fitted.get_forecast(steps=12)
forecast_mean   = forecast_result.predicted_mean
forecast_ci     = forecast_result.conf_int(alpha=0.05)   # 95% CI
```

---

## 6. SARIMA(p,d,q)(P,D,Q,s)

### 6.1 Definition

**SARIMA** = Seasonal ARIMA — extends ARIMA with seasonal AR, I, and MA components:

```
SARIMA(p,d,q)(P,D,Q,s) equation:

φ(B) · Φ(Bˢ) · (1-B)ᵈ · (1-Bˢ)ᴰ · y(t) = c + θ(B) · Θ(Bˢ) · ε(t)

Where:
  φ(B)  = non-seasonal AR polynomial (order p)
  Φ(Bˢ) = seasonal AR polynomial (order P)
  (1-B)ᵈ = non-seasonal differencing (d times)
  (1-Bˢ)ᴰ = seasonal differencing (D times)
  θ(B)  = non-seasonal MA polynomial (order q)
  Θ(Bˢ) = seasonal MA polynomial (order Q)
  s     = seasonal period
```

### 6.2 Parameter Guide

| Notation | What It Represents |
|----------|--------------------|
| `p` | Non-seasonal AR order |
| `d` | Non-seasonal differencing |
| `q` | Non-seasonal MA order |
| `P` | Seasonal AR order |
| `D` | Seasonal differencing |
| `Q` | Seasonal MA order |
| `s` | Seasonal period (12=monthly, 7=daily, 4=quarterly) |

### 6.3 Common Seasonal ACF/PACF Patterns

After non-seasonal differencing `(d)`, look at the residual ACF/PACF for seasonal structure:

| ACF at lag s | PACF at lag s | Seasonal component to add |
|-------------|--------------|--------------------------|
| Single spike | Tails off at s, 2s, 3s | Seasonal MA(1): Q=1 |
| Tails off at s, 2s, 3s | Single spike | Seasonal AR(1): P=1 |
| Both tail off at seasonal lags | Both tail off | Seasonal ARMA: P=1, Q=1 |

### 6.4 Implementation

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA(1,1,1)(1,1,1)[12] — common for monthly data
model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True,
)
fitted = model.fit(disp=False, maxiter=200)

print(fitted.summary())
print(f"\nAIC: {fitted.aic:.2f} | BIC: {fitted.bic:.2f}")

# Forecasting with confidence intervals
forecast = fitted.get_forecast(steps=24)
fc_mean  = forecast.predicted_mean
fc_ci    = forecast.conf_int(alpha=0.05)   # 95% prediction intervals
```

### 6.5 Typical Starting Points by Data Type

| Data | Frequency | Good Starting Model |
|------|-----------|---------------------|
| Monthly economic | 12 | SARIMA(1,1,1)(1,1,1)[12] |
| Monthly retail | 12 | SARIMA(0,1,1)(0,1,1)[12] (airline model) |
| Quarterly | 4 | SARIMA(1,1,0)(1,1,0)[4] |
| Daily (weekly season) | 7 | SARIMA(1,1,1)(1,1,1)[7] |
| Hourly (daily season) | 24 | SARIMA(1,1,1)(1,1,1)[24] |

---

## 7. Box-Jenkins Methodology

The **Box-Jenkins** methodology is the systematic 3-step procedure for fitting ARIMA models:

### Step 1: Identification

```
A. Check stationarity:
   → Run ADF + KPSS tests
   → Determine d (number of differences needed)

B. Apply differencing if needed:
   → First difference for trend
   → Seasonal difference for seasonality

C. Plot ACF and PACF of the stationary series:
   → PACF cuts off at p → AR(p)
   → ACF cuts off at q → MA(q)
   → Both tail off → ARMA(p,q) — use AIC to select

D. Note any seasonal spikes at multiples of s
   → Identify P, D, Q using seasonal ACF/PACF rules
```

### Step 2: Estimation

```
A. Fit the identified model via Maximum Likelihood Estimation (MLE)
B. Compare AIC/BIC across a few candidate models (± 1 on each parameter)
C. Select the model with the lowest AIC (or BIC for parsimony)
```

### Step 3: Diagnostic Checking

```
A. Residual plot — should look like white noise (no patterns)
B. ACF of residuals — all bars inside confidence bands
C. Ljung-Box test — p-value > 0.05 (no autocorrelation in residuals)
D. Q-Q plot — residuals approximately normal
E. If diagnostics fail → go back to Step 1 and revise orders
```

```python
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox

# Complete diagnostic check
fitted.plot_diagnostics(figsize=(13, 10))
plt.tight_layout()
plt.show()

# Ljung-Box test on residuals
lb = acorr_ljungbox(fitted.resid, lags=[10, 20], return_df=True)
print("Ljung-Box Test on Residuals:")
print(lb)
# p-value > 0.05 → residuals are white noise ✅
```

---

## 8. auto_arima — Automated Order Selection

### 8.1 pmdarima Auto-ARIMA

```python
import pmdarima as pm

# Automatically search for best ARIMA order
model_auto = pm.auto_arima(
    train,
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    d=None,          # auto-determine d via ADF test
    seasonal=True,
    m=12,            # seasonal period
    start_P=0, max_P=2,
    start_Q=0, max_Q=2,
    D=None,          # auto-determine D
    information_criterion="aic",
    stepwise=True,   # stepwise search (faster)
    trace=True,      # print search progress
    error_action="ignore",
    suppress_warnings=True,
)

print(f"Best model: {model_auto.order} × {model_auto.seasonal_order}")
print(f"AIC: {model_auto.aic():.2f}")

# Forecast
forecast_auto = model_auto.predict(n_periods=12)
```

### 8.2 statsforecast AutoARIMA (Production-Grade, Fast)

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

sf = StatsForecast(
    models=[AutoARIMA(season_length=12)],
    freq="MS",
    n_jobs=-1,   # parallel across series
)
sf.fit(df_sf)
forecast_df = sf.predict(h=12, level=[80, 95])   # with prediction intervals
print(forecast_df)
```

---

## 9. Forecasting and Prediction Intervals

### 9.1 Point Forecast

```python
# Generate forecast (point + intervals)
forecast_result = fitted.get_forecast(steps=h)
forecast_mean = forecast_result.predicted_mean

# Plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(train[-36:], color="gray", label="Train (last 3 years)", linewidth=1)
ax.plot(test, color="black", linewidth=2, label="Actual")
ax.plot(forecast_mean, color="#D7191C", linewidth=2, linestyle="--", label="SARIMA Forecast")

# Confidence interval (shaded)
ci = forecast_result.conf_int(alpha=0.05)
ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                color="#D7191C", alpha=0.15, label="95% Prediction Interval")

ax.axvline(train.index[-1], color="black", linewidth=0.8, linestyle=":", alpha=0.5)
ax.legend()
ax.set_title("SARIMA(1,1,1)(1,1,1)[12] Forecast with 95% Prediction Intervals")
plt.tight_layout()
plt.show()
```

### 9.2 Prediction Intervals Widen with Horizon

```
For ARIMA(p,d,q):
  Var[ê(t+h)] increases with h (uncertainty grows further into the future)
  
At horizon h=1: error ≈ σ² (white noise variance)
At horizon h=∞: error → Var[y(t)] (uncertainty equals full series variance)
```

This is why **short-horizon forecasts are always more accurate** than long-horizon ones.

---

## 10. ARIMAX / SARIMAX — Exogenous Regressors

ARIMAX (ARIMA with eXogenous variables) and SARIMAX extend the ARIMA family to incorporate **external predictors** — making them critical for production forecasting where covariates (promotions, weather, price, calendar events) are available.

### 10.1 Mathematical Extension

```
ARIMAX(p,d,q) with exogenous variables X:

  φ(B) · (1-B)ᵈ · y(t) = c + β·X(t) + θ(B) · ε(t)

SARIMAX(p,d,q)(P,D,Q,s) with exogenous variables:

  φ(B) · Φ(Bˢ) · (1-B)ᵈ · (1-Bˢ)ᴰ · y(t) = c + β·X(t) + θ(B) · Θ(Bˢ) · ε(t)

Where:
  X(t)  = matrix of exogenous variables at time t  (known future values required at forecast time!)
  β     = coefficient vector for exogenous predictors
  All other terms are the same as SARIMA
```

### 10.2 When to Use ARIMAX/SARIMAX

| Scenario | Exogenous Variables |
|----------|---------------------|
| Retail sales forecasting | Promotions, holidays, price, competitor actions |
| Energy demand | Temperature, day-of-week, public holidays |
| Tourism | Airline seat availability, events, school holidays |
| Finance | Macroeconomic indicators, sentiment indices |
| Supply chain | Lead times, inventory levels, promotional calendar |

> **Critical requirement**: At forecast time, you must have **known future values** of ALL exogenous variables for every step you wish to forecast. Unknown future regressors must be separately forecasted first.

### 10.3 Implementation

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Example: Monthly sales with promotions and holiday indicators as covariates
# X_train shape: (n_train, n_covariates)
# X_test  shape: (n_test,  n_covariates) — MUST be fully known

# Build exogenous feature matrix
X_train = train_df[["is_promotion", "is_holiday", "price_discount_pct"]].values
X_test  = test_df[["is_promotion",  "is_holiday", "price_discount_pct"]].values

# Fit SARIMAX with exogenous regressors
model = SARIMAX(
    train_df["sales"],
    exog=X_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True,
)
fitted = model.fit(disp=False, maxiter=300)

print(fitted.summary())
print(f"\nAIC: {fitted.aic:.2f} | BIC: {fitted.bic:.2f}")

# Print exogenous coefficients
print("\nExogenous Variable Coefficients:")
for name, coef, pval in zip(
    ["is_promotion", "is_holiday", "price_discount_pct"],
    fitted.params[1:4],
    fitted.pvalues[1:4],
):
    sig = "✅" if pval < 0.05 else "❌"
    print(f"  {name:<22}: β={coef:+.4f}  p={pval:.4f} {sig}")
```

### 10.4 Forecasting with Future Exogenous Values

```python
# Generate forecast — must supply future exogenous values
forecast_result = fitted.get_forecast(
    steps=len(X_test),
    exog=X_test,          # REQUIRED: future covariate values
)
forecast_mean = forecast_result.predicted_mean
forecast_ci   = forecast_result.conf_int(alpha=0.05)

# Plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(train_df["sales"][-36:], color="gray", linewidth=1, label="Train")
ax.plot(test_df["sales"],  color="black",   linewidth=2, label="Actual")
ax.plot(forecast_mean,     color="#D7191C", linewidth=2, linestyle="--", label="SARIMAX Forecast")
ax.fill_between(
    forecast_ci.index,
    forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
    color="#D7191C", alpha=0.15, label="95% PI"
)
ax.axvline(train_df.index[-1], color="black", linewidth=0.8, linestyle=":")
ax.legend()
ax.set_title("SARIMAX Forecast with Promotion & Holiday Regressors")
plt.tight_layout()
plt.show()
```

### 10.5 Key Distinctions

| Model | Exog Support | Library |
|-------|-------------|--------|
| `ARIMA` | ❌ None | `statsmodels.tsa.arima.model.ARIMA` |
| `SARIMAX` | ✅ Yes | `statsmodels.tsa.statespace.sarimax.SARIMAX` |
| `AutoARIMA` (pmdarima) | ✅ Yes (`exogenous=`) | `pmdarima` |
| `AutoARIMA` (statsforecast) | ✅ Yes (`X_df=`) | `statsforecast` |

> **Production tip**: Always run SARIMAX without exogenous variables first. If exogenous covariates improve AIC AND out-of-sample RMSE, include them. If only AIC improves, they may be overfit.

---

*← [02 — ETS Models](./02_exponential_smoothing_ETS.md) | [Module README](./README.md) | Next: [04 — VAR Models](./04_var_vector_autoregression.md) →*
