# 03 — Stationarity

> **Module**: 01 Foundations | **File**: 3 of 5
>
> Stationarity is the single most important prerequisite concept in classical time series analysis. Almost every statistical model assumes it — yet most real-world series violate it.

---

## Table of Contents

1. [What is Stationarity?](#1-what-is-stationarity)
2. [Types of Non-Stationarity](#2-types-of-non-stationarity)
3. [Visual Diagnosis](#3-visual-diagnosis)
4. [ADF Test](#4-augmented-dickey-fuller-adf-test)
5. [KPSS Test](#5-kpss-test)
6. [ADF + KPSS Combined Decision](#6-adf--kpss-combined-decision)
7. [Making a Series Stationary](#7-making-a-series-stationary)
8. [Integration Order d](#8-integration-order-d)
9. [Practical Checklist](#9-practical-checklist)

---

## 1. What is Stationarity?

### 1.1 Strict Stationarity

A time series `{Y_t}` is **strictly stationary** if its joint probability distribution is invariant to time shifts:

```
(Y_{t₁}, Y_{t₂}, ..., Y_{tₖ}) =ᵈ (Y_{t₁+τ}, Y_{t₂+τ}, ..., Y_{tₖ+τ})  for all τ
```

This is a very strong condition — rarely assumed in practice.

### 1.2 Weak (Covariance) Stationarity

**Weak stationarity** (also called second-order stationarity) requires only:

```
1.  E[Y(t)] = μ            (constant mean — no trend)
2.  Var[Y(t)] = σ²         (constant variance — no heteroskedasticity)
3.  Cov[Y(t), Y(t+k)] = γ(k)  (autocovariance depends only on lag k, not on time t)
```

> This is what **"stationarity" means in practice**. When statisticians and ML engineers say a series is stationary, they mean weak stationarity.

### 1.3 Why Stationarity Matters

| Reason | Explanation |
|--------|-------------|
| **Model validity** | AR, MA, ARMA models are derived under stationarity assumptions |
| **Parameter stability** | If mean/variance change over time, model parameters become meaningless |
| **Prediction reliability** | Stationary series' future behavior resembles its past — enabling extrapolation |
| **Test applicability** | Most statistical tests (t-test, F-test) assume stationarity |

> **Key insight**: If you fit an AR(1) model to a non-stationary series (like a random walk), you'll get a coefficient near 1.0 and "impressive" in-sample fit — but the model will be completely unreliable out-of-sample.

---

## 2. Types of Non-Stationarity

| Type | Description | Example | Fix |
|------|-------------|---------|-----|
| **Trend** | Mean shifts over time | Rising sales, growing population | First differencing or detrending |
| **Seasonality** | Periodic mean shifts | Monthly sales peaks | Seasonal differencing |
| **Unit root** | Random walk — shocks accumulate permanently | Financial prices | First differencing |
| **Heteroskedasticity** | Variance changes over time | Volatile stock returns | Log transform or Box-Cox |
| **Structural break** | Sudden regime change | COVID impact on travel | Segmented models, include break dummy |
| **Explosive process** | Variance grows without bound | Hyperinflation | Differencing, transformation |

### 2.1 The Random Walk — Most Important Non-Stationary Process

```
Y(t) = Y(t-1) + ε(t)    where ε(t) ~ WN(0, σ²)

Properties:
  E[Y(t)] = Y(0)                  → Mean depends on starting point
  Var[Y(t)] = t · σ²              → Variance grows linearly with time!
  Autocorrelation → 1 for all lags → ACF decays extremely slowly
```

A random walk **cannot be predicted** — the best forecast is the last observed value (naïve forecast). This is a key insight for financial markets.

---

## 3. Visual Diagnosis

**Always plot the series before running any test.**

```python
import matplotlib.pyplot as plt
import pandas as pd

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Raw series
axes[0].plot(series)
axes[0].set_title('Raw Series')

# Rolling mean and std (window = seasonal period or longer)
rolling_mean = series.rolling(window=12).mean()
rolling_std  = series.rolling(window=12).std()
axes[1].plot(series, label='Original', alpha=0.5)
axes[1].plot(rolling_mean, label='Rolling Mean (12)', color='red')
axes[1].plot(rolling_std, label='Rolling Std (12)', color='orange')
axes[1].legend()
axes[1].set_title('Rolling Statistics')
plt.tight_layout()
plt.show()
```

**Reading the plot:**

```
Signs of NON-stationarity:
  ✗ Obvious upward or downward drift in the series
  ✗ Rolling mean is not horizontal (it drifts)
  ✗ Rolling std increases or decreases over time
  ✗ ACF decays very slowly (20+ significant lags)
  ✗ Seasonal swings growing in amplitude

Signs of STATIONARITY:
  ✓ Series fluctuates around a constant level
  ✓ Rolling mean is roughly flat
  ✓ Rolling std is roughly constant
  ✓ ACF drops to near zero after a few lags
```

---

## 4. Augmented Dickey-Fuller (ADF) Test

### 4.1 Hypotheses

```
H₀ (Null):      A unit root is present → Series is NON-STATIONARY
H₁ (Alternative): No unit root → Series is STATIONARY

Decision rule:
  p-value < 0.05  →  Reject H₀  →  Evidence for stationarity     ✅
  p-value ≥ 0.05  →  Fail to reject H₀  →  Evidence for non-stationarity  ❌
```

### 4.2 The ADF Regression

The ADF test estimates this regression and tests whether `δ = 0` (unit root):

```
ΔY(t) = α + β·t + δ·Y(t-1) + Σᵢ γᵢ·ΔY(t-i) + ε(t)

Where:
  ΔY(t)   = first difference of Y (the dependent variable)
  α       = constant (drift term)
  β·t     = optional time trend
  δ       = key coefficient: δ = 0 → unit root (non-stationary)
  Σᵢ γᵢ·ΔY(t-i) = lagged differences to correct for autocorrelation in errors
```

### 4.3 Running the ADF Test in Python

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(series, autolag='AIC')

print("=" * 40)
print(f"  ADF Statistic : {result[0]:.4f}")
print(f"  p-value       : {result[1]:.4f}")
print(f"  Lags used     : {result[2]}")
print(f"  Observations  : {result[3]}")
print("\n  Critical Values:")
for key, val in result[4].items():
    print(f"    {key}: {val:.4f}")
print("=" * 40)

# Interpretation
if result[1] < 0.05:
    print("✅ Reject H₀ — Series is likely STATIONARY")
else:
    print("❌ Fail to reject H₀ — Series is likely NON-STATIONARY")
```

### 4.4 ADF Parameters

| Parameter | Options | Recommendation |
|-----------|---------|---------------|
| `autolag` | `'AIC'`, `'BIC'`, `'t-stat'`, integer | Use `'AIC'` (default) |
| `regression` | `'c'` (constant), `'ct'` (constant+trend), `'n'` (none) | `'c'` for most cases |
| `maxlags` | Integer | Let `autolag` decide unless you have reason to fix it |

---

## 5. KPSS Test

### 5.1 Hypotheses (OPPOSITE of ADF)

```
H₀ (Null):      Series IS STATIONARY
H₁ (Alternative): Series is NON-STATIONARY (has a unit root)

Decision rule:
  p-value < 0.05  →  Reject H₀  →  Evidence for NON-stationarity  ❌
  p-value ≥ 0.05  →  Fail to reject H₀  →  Evidence for stationarity  ✅
```

> **Critical**: The KPSS null is the opposite of ADF. Being careful about which direction to read the p-value is essential.

### 5.2 Running the KPSS Test

```python
from statsmodels.tsa.stattools import kpss

stat, p_value, lags, critical_values = kpss(series, regression='c', nlags='auto')

print("=" * 40)
print(f"  KPSS Statistic : {stat:.4f}")
print(f"  p-value        : {p_value:.4f}")
print(f"  Lags used      : {lags}")
print("\n  Critical Values:")
for key, val in critical_values.items():
    print(f"    {key}: {val:.4f}")
print("=" * 40)

# Interpretation
if p_value < 0.05:
    print("❌ Reject H₀ — Series is NON-STATIONARY")
else:
    print("✅ Fail to reject H₀ — Series is STATIONARY")
```

### 5.3 KPSS `regression` Parameter

| Value | Tests For |
|-------|-----------|
| `'c'` | Level stationarity (constant mean, no trend) |
| `'ct'` | Trend stationarity (stationary around a linear trend) |

---

## 6. ADF + KPSS Combined Decision

Using both tests together gives a more reliable conclusion:

| ADF Result | KPSS Result | Conclusion |
|-----------|-------------|-----------|
| Reject H₀ (→ stationary) | Fail to reject H₀ (→ stationary) | ✅ **Stationary** — both tests agree |
| Fail to reject H₀ (→ non-stationary) | Reject H₀ (→ non-stationary) | ❌ **Non-stationary** — both tests agree |
| Reject H₀ (→ stationary) | Reject H₀ (→ non-stationary) | ⚠️ **Trend stationary** — stationary around a trend; detrend it |
| Fail to reject H₀ (→ non-stationary) | Fail to reject H₀ (→ stationary) | ⚠️ **Difference stationary** — has unit root; first-difference it |

```python
def stationarity_report(series):
    from statsmodels.tsa.stattools import adfuller, kpss

    adf_stat, adf_p, *_ = adfuller(series, autolag='AIC')
    kpss_stat, kpss_p, *_ = kpss(series, regression='c', nlags='auto')

    print(f"ADF  p-value : {adf_p:.4f}  → {'Stationary' if adf_p < 0.05 else 'Non-stationary'}")
    print(f"KPSS p-value : {kpss_p:.4f}  → {'Non-stationary' if kpss_p < 0.05 else 'Stationary'}")
```

---

## 7. Making a Series Stationary

### 7.1 Method 1 — First Differencing

Removes **linear trend** and **unit roots**.

```
ΔY(t) = Y(t) - Y(t-1)
```

```python
series_diff = series.diff().dropna()
```

**When to use**: Series has a linear trend or is a random walk (I(1) process).

### 7.2 Method 2 — Seasonal Differencing

Removes **seasonality** of period `s`.

```
Δₛ Y(t) = Y(t) - Y(t-s)
```

```python
s = 12   # seasonal period (12 for monthly data)
series_seasonal_diff = series.diff(s).dropna()
```

**When to use**: Series has a strong, stable seasonal pattern.

### 7.3 Method 3 — Log Transform

Stabilizes **multiplicative variance** (heteroskedasticity) — variance growing with level.

```python
import numpy as np
series_log = np.log(series)   # series must be strictly positive
```

**When to use**: Variance grows proportionally with the mean (common in sales, financial data).

### 7.4 Method 4 — Box-Cox Transform

A generalization of the log transform that finds the optimal power `λ`:

```
Y_transformed = (Y^λ - 1) / λ    if λ ≠ 0
              = log(Y)            if λ = 0
```

```python
from scipy.stats import boxcox
series_bc, lambda_val = boxcox(series)
print(f"Optimal lambda: {lambda_val:.3f}")
# lambda ≈ 0 → log transform
# lambda ≈ 0.5 → square root transform
# lambda ≈ 1 → no transform needed
```

### 7.5 Method 5 — Combined Transform (Common for SARIMA)

```python
# Step 1: Log transform (stabilize variance)
series_log = np.log(series)

# Step 2: First difference (remove trend)
series_log_diff = series_log.diff().dropna()

# Step 3: Seasonal difference (remove seasonality)
series_final = series_log_diff.diff(12).dropna()

# Verify stationarity after transformation
stationarity_report(series_final)
```

> **Rule of thumb**: Rarely need more than one first difference + one seasonal difference. Over-differencing introduces unnecessary MA terms (test: PACF of differenced series will show significant negative spike at lag 1).

---

## 8. Integration Order d

The **order of integration** `d` is the minimum number of first differences required to make a series stationary.

| Notation | `d` | Description | Example |
|----------|-----|-------------|---------|
| **I(0)** | 0 | Already stationary — no differencing needed | White noise, stationary AR process |
| **I(1)** | 1 | One first difference makes it stationary | Most economic and financial series |
| **I(2)** | 2 | Two differences needed | Price acceleration series (rare in practice) |

**Why it matters**: `d` is the middle parameter in `ARIMA(p, d, q)`. Getting it wrong invalidates the entire model.

```
ARIMA(p=1, d=1, q=0) means:
  - First-difference the series once (d=1)
  - Then fit an AR(1) model to the differenced series
```

---

## 9. Practical Checklist

Use this checklist every time you start working with a new time series:

```
□ 1. Plot the raw series — look for obvious trend, seasonality, outliers
□ 2. Plot rolling mean (window=s) and rolling std — check for drift
□ 3. Plot ACF — slow decay → non-stationary, spikes at s → seasonal
□ 4. Run ADF test — note p-value and conclusion
□ 5. Run KPSS test — note p-value and conclusion
□ 6. Use ADF+KPSS table to determine stationarity status
□ 7. If non-stationary:
        - Trend present → first difference
        - Seasonality present → seasonal difference (period s)
        - Variance unstable → log transform first
□ 8. Re-run ADF + KPSS on transformed series — confirm stationarity
□ 9. Note the integration order d for ARIMA modeling
```

---

*← [02 — Components](./02_components_trend_seasonality.md) | [Module README](./README.md) | Next: [04 — ACF & PACF](./04_autocorrelation_acf_pacf.md) →*
