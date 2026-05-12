# 04 — Autocorrelation: ACF & PACF

> **Module**: 01 Foundations | **File**: 4 of 5
>
> ACF and PACF plots are the primary diagnostic tools in time series analysis. They reveal the memory structure of a series and directly guide the selection of AR and MA model orders.

---

## Table of Contents

1. [Autocorrelation](#1-autocorrelation)
2. [Autocorrelation Function (ACF)](#2-autocorrelation-function-acf)
3. [Partial Autocorrelation Function (PACF)](#3-partial-autocorrelation-function-pacf)
4. [ACF vs. PACF — Key Differences](#4-acf-vs-pacf--key-differences)
5. [Reading ACF & PACF Plots](#5-reading-acf--pacf-plots)
6. [Model Order Identification](#6-model-order-identification)
7. [Ljung-Box Test](#7-ljung-box-test)
8. [Lag Plots](#8-lag-plots)
9. [Common ACF/PACF Patterns](#9-common-acfpacf-patterns)
10. [Cross-Correlation Function (CCF)](#10-cross-correlation-function-ccf)

---

## 1. Autocorrelation

### 1.1 Definition

**Autocorrelation** measures the **linear correlation** between a time series and a lagged version of itself.

At lag `k`, it answers: **"How similar is the value today compared to the value `k` periods ago?"**

```
Autocovariance at lag k:
  γ(k) = Cov[Y(t), Y(t-k)] = E[(Y(t) - μ)(Y(t-k) - μ)]

Autocorrelation at lag k:
  ρ(k) = γ(k) / γ(0) = Cov[Y(t), Y(t-k)] / Var[Y(t)]

Range: -1 ≤ ρ(k) ≤ 1
  ρ(k) = +1  → Perfect positive linear relationship at lag k
  ρ(k) =  0  → No linear relationship at lag k
  ρ(k) = -1  → Perfect negative linear relationship at lag k
```

### 1.2 Intuition by Domain

| Series | ρ(1) | Intuition |
|--------|------|-----------|
| Daily temperature | High (~0.9) | Today's temp is very similar to yesterday's |
| Weekly retail sales | Moderate (~0.6) | Last week's sales partially predict this week |
| Retail sales lag 7 | High | Same day last week is the strongest predictor |
| White noise | Near 0 | No relationship with any past value |
| Over-differenced series | Negative | Alternating pattern introduced by excess differencing |

---

## 2. Autocorrelation Function (ACF)

### 2.1 What the ACF Shows

The **ACF plot** shows `ρ(k)` for all lags `k = 0, 1, 2, 3, ...`

```
Important: ACF includes BOTH direct AND indirect effects.

Example at lag 2:
  Y(t) is correlated with Y(t-2) because:
    1. Y(t-2) directly influences Y(t)               ← direct effect
    2. Y(t-2) → Y(t-1) → Y(t)                        ← indirect effect via lag 1
  
  The ACF at lag 2 captures the TOTAL of both effects.
```

### 2.2 Computing and Plotting ACF

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

fig, ax = plt.subplots(figsize=(12, 4))
plot_acf(
    series,
    lags=40,          # how many lags to display
    alpha=0.05,       # significance level for confidence band (95%)
    ax=ax
)
ax.set_title('Autocorrelation Function (ACF)')
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
plt.tight_layout()
plt.show()
```

### 2.3 Computing ACF Values Directly

```python
from statsmodels.tsa.stattools import acf

acf_values, confint = acf(series, nlags=40, alpha=0.05)
# acf_values[0] = 1.0 (lag 0, always)
# acf_values[1] = ρ(1)
# acf_values[k] = ρ(k)
```

### 2.4 The Confidence Band

The blue shaded region (or dashed lines) represents the 95% confidence interval:

```
Approximate 95% confidence band: ±1.96 / √n

Where n = number of observations.
Bars OUTSIDE this band are statistically significant autocorrelations.
```

---

## 3. Partial Autocorrelation Function (PACF)

### 3.1 What the PACF Shows

The **PACF** measures the **direct relationship** between `Y(t)` and `Y(t-k)` after **removing the indirect effect** of all intermediate lags `1, 2, ..., k-1`.

```
PACF at lag k = Correlation(Y(t), Y(t-k) | Y(t-1), Y(t-2), ..., Y(t-k+1))

"The correlation between today's value and the value k periods ago,
 after controlling for everything in between."
```

### 3.2 Intuition

Think of PACF as answering: **"Does lag k add any NEW information about Y(t) beyond what lags 1 through k-1 already explain?"**

```
Example: Daily temperature
  ACF(2) is high because: temp(t) ↔ temp(t-1) ↔ temp(t-2)
  PACF(2) is lower because: most of the correlation with temp(t-2)
           is explained by temp(t-1) acting as a middleman.
```

### 3.3 Computing and Plotting PACF

```python
from statsmodels.graphics.tsaplots import plot_pacf

fig, ax = plt.subplots(figsize=(12, 4))
plot_pacf(
    series,
    lags=40,
    method='ywm',     # Yule-Walker with bias correction (most stable)
    alpha=0.05,
    ax=ax
)
ax.set_title('Partial Autocorrelation Function (PACF)')
ax.set_xlabel('Lag')
ax.set_ylabel('Partial Autocorrelation')
plt.tight_layout()
plt.show()
```

### 3.4 PACF Estimation Methods

| Method | Code | Notes |
|--------|------|-------|
| Yule-Walker | `'ywm'` | Default, most stable for AR processes |
| Levinson-Durbin | `'ld'` | Equivalent to Yule-Walker |
| OLS | `'ols'` | Regress Y on its lags, more flexible |

---

## 4. ACF vs. PACF — Key Differences

| Property | ACF | PACF |
|----------|-----|------|
| **Measures** | Total correlation at lag k (direct + indirect) | Direct-only correlation at lag k |
| **Identifies** | MA(q) order — cuts off after lag q | AR(p) order — cuts off after lag p |
| **For MA(q)** | Cuts off sharply after lag q | Tails off gradually |
| **For AR(p)** | Tails off gradually | Cuts off sharply after lag p |
| **For ARMA** | Both tail off | Both tail off |
| **Lag 0** | Always 1.0 | Always 1.0 |

---

## 5. Reading ACF & PACF Plots

### 5.1 Pattern Recognition Guide

```
"Cuts off" = drops abruptly to near zero and stays there after lag k
"Tails off" = decays gradually and exponentially toward zero
```

| ACF Pattern | Meaning |
|-------------|---------|
| Cuts off after lag q | MA(q) process |
| Tails off slowly (exponential decay) | AR process |
| Tails off with oscillations | AR with negative coefficients |
| Very slow decay (20+ significant lags) | Non-stationary — difference the series first! |
| Significant spikes at s, 2s, 3s... | Seasonal period s |
| All bars near zero | White noise |
| Significant negative spike at lag 1 | Over-differenced |

| PACF Pattern | Meaning |
|--------------|---------|
| Cuts off after lag p | AR(p) process |
| Tails off slowly | MA process |
| Significant spike at lag s, near zero elsewhere | Seasonal AR |
| All bars near zero | White noise or MA(0) |

### 5.2 Seasonal ACF Patterns

```
For monthly data with yearly seasonality (s=12):
  ACF shows significant spikes at lags: 12, 24, 36 ...
  PACF shows spike at lag 12, may decay at 24, 36 ...

For daily data with weekly seasonality (s=7):
  ACF shows significant spikes at lags: 7, 14, 21 ...
```

### 5.3 Plotting Both Together

```python
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(series, lags=40, ax=axes[0])
plot_pacf(series, lags=40, method='ywm', ax=axes[1])
axes[0].set_title('ACF')
axes[1].set_title('PACF')
plt.tight_layout()
plt.show()
```

---

## 6. Model Order Identification

This is the primary **practical use** of ACF and PACF — determining the `p` and `q` orders for ARIMA.

### 6.1 Step-by-Step Process

```
Step 1: Ensure the series is STATIONARY (use ADF + KPSS from topic 03)
         If not → difference → re-check

Step 2: Plot ACF and PACF of the stationary series

Step 3: Identify the pattern using the table below

Step 4: Fit the identified model and check residuals

Step 5: If residuals show structure → adjust orders and repeat
```

### 6.2 Pattern → Model Mapping

| ACF | PACF | Model to Try |
|-----|------|-------------|
| Cuts off at lag q | Tails off | **MA(q)** |
| Tails off | Cuts off at lag p | **AR(p)** |
| Tails off | Tails off | **ARMA(p, q)** — use AIC/BIC to select orders |
| Spikes at s, 2s, 3s | — | Add seasonal terms |
| Very slow decay | — | **Non-stationary** — difference first |

### 6.3 Practical Example

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Load monthly airline passengers (classic dataset)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, index_col=0, parse_dates=True)
series = df['Passengers']

# Step 1: Check stationarity
# -> Series has trend + multiplicative seasonality -> log + diff
import numpy as np
series_transformed = np.log(series).diff().diff(12).dropna()

# Step 2: Plot ACF and PACF
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(series_transformed, lags=40, ax=axes[0])
plot_pacf(series_transformed, lags=40, method='ywm', ax=axes[1])
plt.tight_layout()
plt.show()
# Read: cuts off at lag 1 in both → ARIMA(1,1,1)(1,1,1)[12] candidate
```

---

## 7. Ljung-Box Test

### 7.1 Purpose

The **Ljung-Box test** formally tests whether a set of autocorrelations is jointly significantly different from zero. Its main use is **residual diagnostics** after fitting a model.

```
H₀: Autocorrelations at lags 1 through h are all zero (white noise)
H₁: At least one autocorrelation is non-zero

Test statistic:
  Q = n(n+2) Σₖ₌₁ʰ [ρ²(k) / (n-k)]  ~  χ²(h - p - q)

Where:
  n = number of observations
  h = number of lags tested
  p, q = AR and MA orders of the fitted model
```

### 7.2 Running the Test

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

# After fitting a model, test its residuals
lb_result = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
print(lb_result[['lb_stat', 'lb_pvalue']])

# Interpretation
# lb_pvalue > 0.05 for all lags → residuals look like white noise ✅
# lb_pvalue < 0.05 for any lag  → residuals have unexplained structure ❌
```

### 7.3 What Ljung-Box Failure Means

```
p-value < 0.05 → Residuals are NOT white noise:
  → Model is under-specified (missing AR or MA terms)
  → Try increasing p or q in ARIMA
  → Check for seasonal terms not included
  → Try a different model family
```

---

## 8. Lag Plots

A **lag plot** is a scatter plot of `Y(t)` against `Y(t-k)`. It gives a visual confirmation of autocorrelation.

```python
from pandas.plotting import lag_plot
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for i, ax in enumerate(axes.flatten()):
    lag_plot(series, lag=i+1, ax=ax)
    ax.set_title(f'Lag {i+1}')
plt.suptitle('Lag Plots')
plt.tight_layout()
plt.show()
```

| Plot Pattern | Interpretation |
|-------------|----------------|
| Tight linear cluster (upward) | Strong positive autocorrelation at this lag |
| Scattered cloud | Little to no autocorrelation |
| Tight linear cluster (downward) | Strong negative autocorrelation |
| Elliptical cluster | Moderate autocorrelation |

---

## 9. Common ACF/PACF Patterns

### Pattern 1: AR(1) Process — `Y(t) = 0.8·Y(t-1) + ε(t)`
```
ACF:  Exponential decay starting from high positive value
PACF: Single significant spike at lag 1, cuts off after
→ Fit AR(1)
```

### Pattern 2: MA(1) Process — `Y(t) = ε(t) + 0.7·ε(t-1)`
```
ACF:  Single significant spike at lag 1, cuts off after
PACF: Exponential decay (or oscillating decay)
→ Fit MA(1)
```

### Pattern 3: ARMA(1,1) Process
```
ACF:  Exponential decay starting from lag 1
PACF: Exponential decay starting from lag 1
→ Both tail off → ARMA(p,q) — use AIC/BIC to find best p, q
```

### Pattern 4: Non-Stationary (Random Walk)
```
ACF:  Very slow linear decay — significant for 20+ lags
PACF: Single dominant spike at lag 1 (near 1.0)
→ Apply first differencing, then re-examine ACF/PACF
```

### Pattern 5: Seasonal Pattern (Monthly, s=12)
```
ACF:  Large spikes at lags 12, 24, 36 (multiples of 12)
PACF: Spike at lag 12, smaller at 24, 36
→ Add seasonal MA or AR terms: SARIMA(p,d,q)(P,D,Q,12)
```

---

## 10. Cross-Correlation Function (CCF)

While ACF/PACF measure a series' relationship with its **own past**, the **CCF** measures the correlation between **two different time series** across different lags.

### 10.1 Definition

```
CCF(X, Y, k) = Corr[X(t), Y(t-k)]

  k > 0: X leads Y (past values of X predict current Y)
  k < 0: Y leads X (past values of Y predict current X)
  k = 0: Contemporaneous correlation
```

### 10.2 Use Cases

| Scenario | What CCF Reveals |
|----------|------------------|
| Does advertising spend drive sales? | CCF(ads, sales, k) — peak at k=+1 or +2 suggests ads lead sales by 1-2 weeks |
| Does temperature predict energy demand? | CCF(temp, demand, k=0) should be high |
| Does PMI predict GDP? | CCF(PMI, GDP, k=+1) — PMI is a leading indicator |

### 10.3 Python Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_ccf
from statsmodels.tsa.stattools import ccf

# Example: Does X (ads spend) lead Y (sales)?
np.random.seed(42)
n = 200
X = pd.Series(np.random.normal(0, 1, n))          # input series
Y = X.shift(2) + np.random.normal(0, 0.3, n)      # Y is X delayed by 2 lags + noise

# Compute CCF values
ccf_values = ccf(X.dropna(), Y.dropna(), unbiased=False)
lags = np.arange(len(ccf_values))

# Plot
fig, ax = plt.subplots(figsize=(12, 4))
ax.stem(lags[:30], ccf_values[:30], markerfmt='C0o', linefmt='C0-', basefmt='k-')
ax.axhline(1.96 / np.sqrt(n), color='red', linestyle='--', label='95% CI')
ax.axhline(-1.96 / np.sqrt(n), color='red', linestyle='--')
ax.set_title('Cross-Correlation Function (CCF): X → Y')
ax.set_xlabel('Lag k  (positive = X leads Y)')
ax.set_ylabel('Correlation')
ax.legend()
plt.tight_layout()
plt.show()

# Read: significant spike at lag 2 → X leads Y by 2 periods
```

### 10.4 Important: Pre-whiten Before CCF

If both series are autocorrelated (which they almost always are), the CCF will be spuriously inflated. **Pre-whiten** both series first:

```python
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Fit AR model to X (remove its own autocorrelation)
model_X = ARIMA(X, order=(1, 0, 0)).fit()
resid_X = model_X.resid

# Step 2: Filter Y with the same AR model
# Apply the same AR filter to Y
resid_Y = Y - model_X.predict()

# Step 3: CCF on the pre-whitened residuals
ccf_clean = ccf(resid_X.dropna(), resid_Y.dropna(), unbiased=False)
# Now the CCF isolates X→Y causation without autocorrelation noise
```

### 10.5 CCF vs. Granger Causality

| Method | What It Does | Limitation |
|--------|-------------|------------|
| **CCF** | Shows correlation at different lags | Correlation ≠ causation |
| **Granger Causality** | Tests if X improves forecast of Y beyond Y's own history | Still not true causality |
| **PCMCI** | Causal discovery accounting for confounders | Module 12 |

---

*← [03 — Stationarity](./03_stationarity.md) | [Module README](./README.md) | Next: [05 — Decomposition](./05_decomposition.md) →*
