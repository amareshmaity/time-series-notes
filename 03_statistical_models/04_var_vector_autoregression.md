# 04 — VAR: Vector AutoRegression

> **Module**: 03 Statistical Models | **File**: 4 of 6
>
> VAR extends the univariate AR model to handle multiple interrelated time series simultaneously — allowing each series to be a function of past values of all series in the system.

---

## Table of Contents

1. [Why VAR?](#1-why-var)
2. [VAR Model Definition](#2-var-model-definition)
3. [Stationarity for VAR](#3-stationarity-for-var)
4. [Lag Order Selection](#4-lag-order-selection)
5. [Fitting and Forecasting](#5-fitting-and-forecasting)
6. [Granger Causality](#6-granger-causality)
7. [Impulse Response Functions](#7-impulse-response-functions)
8. [Forecast Error Variance Decomposition](#8-forecast-error-variance-decomposition)
9. [VAR vs. Separate Univariate Models](#9-var-vs-separate-univariate-models)
10. [VECM — Vector Error Correction Model](#10-vecm--vector-error-correction-model)

---

## 1. Why VAR?

Univariate models (ARIMA) forecast each series independently. But in reality, many series are **interrelated**:

- GDP affects unemployment, which affects consumer spending, which affects GDP
- Temperature affects energy demand, which affects electricity prices
- Marketing spend affects sales, which affects inventory, which affects re-order decisions

**VAR captures these cross-series dynamics** — each variable is a function of past values of all variables in the system.

### When to Use VAR

| Condition | Use VAR |
|-----------|---------|
| Multiple interdependent series | ✅ |
| You want to understand causal relationships | ✅ |
| You want to model system-wide shocks | ✅ |
| Each series independently predictable | ❌ Use ARIMA per series |
| Many series (> 20) | ❌ Too many parameters — use factor models |

---

## 2. VAR Model Definition

### 2.1 VAR(p) Equation

For a system of `K` variables and `p` lags:

```
y(t) = c + A₁·y(t-1) + A₂·y(t-2) + ... + Aₚ·y(t-p) + ε(t)

Where:
  y(t) = [y₁(t), y₂(t), ..., yK(t)]ᵀ   (K×1 vector of variables at time t)
  c    = [c₁, ..., cK]ᵀ                   (K×1 constant vector)
  Aᵢ   = K×K coefficient matrix for lag i
  ε(t) = [ε₁(t), ..., εK(t)]ᵀ            (K×1 vector of white noise errors)
```

### 2.2 Number of Parameters

```
Total parameters = K² · p + K (for constants)

Example:
  K=3 variables, p=2 lags: 3² × 2 + 3 = 21 parameters
  K=5 variables, p=4 lags: 5² × 4 + 5 = 105 parameters!
```

> **⚠️ Parameter explosion**: VAR models quickly become over-parameterized with many variables. A common rule: need at least `10 × (K² · p)` observations to reliably estimate a VAR(p).

### 2.3 VAR(1) Example — Two Variables

```
[y₁(t)]   [c₁]   [a₁₁  a₁₂] [y₁(t-1)]   [ε₁(t)]
[y₂(t)] = [c₂] + [a₂₁  a₂₂] [y₂(t-1)] + [ε₂(t)]

Expanded:
y₁(t) = c₁ + a₁₁·y₁(t-1) + a₁₂·y₂(t-1) + ε₁(t)
y₂(t) = c₂ + a₂₁·y₁(t-1) + a₂₂·y₂(t-1) + ε₂(t)

→ y₁(t) is modeled using past values of BOTH y₁ and y₂
→ y₂(t) is modeled using past values of BOTH y₁ and y₂
```

---

## 3. Stationarity for VAR

All variables in the VAR system must be **stationary** before fitting.

### 3.1 Checking Stationarity

```python
from statsmodels.tsa.stattools import adfuller

variables = ["gdp_growth", "unemployment", "inflation"]

for col in variables:
    adf_stat, p_val, *_ = adfuller(df[col].dropna(), autolag="AIC")
    status = "Stationary ✅" if p_val < 0.05 else "Non-stationary ❌"
    print(f"  {col:<20} ADF p={p_val:.4f}  → {status}")
```

### 3.2 Making VAR Variables Stationary

```python
# First difference all non-stationary variables
df_stationary = df[variables].diff().dropna()

# Verify after differencing
for col in df_stationary.columns:
    _, p_val, *_ = adfuller(df_stationary[col], autolag="AIC")
    print(f"  {col:<20} ADF p={p_val:.4f}  → {'✅' if p_val < 0.05 else '❌'}")
```

> **Note**: If all variables are I(1) and cointegrated, use a **VECM (Vector Error Correction Model)** instead of VAR on differences — VAR on differences discards the long-run cointegration information.

---

## 4. Lag Order Selection

The VAR lag order `p` is selected using information criteria. Unlike ARIMA where ACF/PACF guide the order, VAR uses:

```
AIC  = log|Σ̂| + (2/T) · K² · p
BIC  = log|Σ̂| + (log T / T) · K² · p
HQC  = log|Σ̂| + (2 log log T / T) · K² · p

Where Σ̂ = estimated residual covariance matrix
```

```python
from statsmodels.tsa.api import VAR

# Fit VAR with automatic lag selection
model = VAR(df_stationary)
results_lag_selection = model.select_order(maxlags=8)
print(results_lag_selection.summary())

# Extract best lag by AIC
optimal_lag_aic = results_lag_selection.aic
optimal_lag_bic = results_lag_selection.bic
print(f"\nOptimal lag by AIC: {optimal_lag_aic}")
print(f"Optimal lag by BIC: {optimal_lag_bic}")
```

---

## 5. Fitting and Forecasting

```python
from statsmodels.tsa.api import VAR

# Fit VAR with selected lag order
model = VAR(df_stationary)
fitted = model.fit(maxlags=optimal_lag_aic, ic=None)   # fit with specific lag

print(fitted.summary())
print(f"\nAIC: {fitted.aic:.4f}")
print(f"BIC: {fitted.bic:.4f}")

# Check model residuals
print("\nLjung-Box tests on residuals:")
from statsmodels.stats.stattools import durbin_watson
print(f"Durbin-Watson: {durbin_watson(fitted.resid)}")

# Forecast (requires last p observations as input)
lag_data = df_stationary.values[-optimal_lag_aic:]
forecast_result = fitted.forecast(y=lag_data, steps=12)
forecast_df = pd.DataFrame(
    forecast_result,
    columns=df_stationary.columns,
    index=pd.date_range(
        start=df_stationary.index[-1] + df_stationary.index.freq,
        periods=12,
        freq=df_stationary.index.freq,
    )
)
print("\nVAR Forecast (differenced scale):")
print(forecast_df.head())

# Un-difference to get back to original scale
# If original series were differenced once:
last_values = df[variables].iloc[-1]
forecast_original = last_values + forecast_df.cumsum()
print("\nVAR Forecast (original scale):")
print(forecast_original.head())

# Plot forecasts
import matplotlib.pyplot as plt
fig, axes = plt.subplots(len(variables), 1, figsize=(13, 3 * len(variables)))
for i, col in enumerate(variables):
    axes[i].plot(df[col][-48:], color="#2C7BB6", linewidth=1.5, label="Historical")
    axes[i].plot(forecast_original[col], color="#D7191C",
                 linewidth=2, linestyle="--", label="VAR Forecast")
    axes[i].set_title(col)
    axes[i].legend(fontsize=9)
plt.suptitle("VAR Forecasts — All Variables", fontweight="bold")
plt.tight_layout()
plt.show()
```

---

## 6. Granger Causality

### 6.1 Definition

**Granger causality** tests whether past values of series X improve the prediction of series Y beyond what past values of Y alone can explain.

```
"X Granger-causes Y" means:
  Var[Y(t) | Y(t-1), Y(t-2), ...] > Var[Y(t) | Y(t-1), ..., X(t-1), ...]
  
  → Past X contains information about future Y that is NOT in past Y alone

H₀: X does NOT Granger-cause Y
H₁: X DOES Granger-cause Y
p-value < 0.05 → Reject H₀ → X Granger-causes Y
```

> ⚠️ **Important**: Granger causality is **predictive**, not structural causality. "X Granger-causes Y" means X helps predict Y — not that X causes Y in a physical sense.

### 6.2 Testing Granger Causality

```python
from statsmodels.tsa.stattools import grangercausalitytests

# Test: does GDP growth Granger-cause unemployment?
results = grangercausalitytests(
    df_stationary[["unemployment", "gdp_growth"]],   # [effect, cause]
    maxlag=4,
    verbose=True,
)

# Extract p-values across lags
print("\nGranger Causality: GDP → Unemployment")
for lag, result in results.items():
    f_test_p = result[0]["ssr_ftest"][1]   # F-test p-value
    chi2_p   = result[0]["ssr_chi2test"][1]
    print(f"  Lag {lag}: F-test p={f_test_p:.4f}, Chi2 p={chi2_p:.4f}")

# Within a fitted VAR — test all pairs
print("\nAll-pairs Granger causality (from fitted VAR):")
gc_result = fitted.test_causality(causing=["gdp_growth"], caused=["unemployment"])
print(gc_result.summary())
```

### 6.3 Granger Causality Matrix

```python
# Build a causality matrix for all variable pairs
variables = df_stationary.columns.tolist()
causality_matrix = pd.DataFrame(index=variables, columns=variables, dtype=float)

for causing in variables:
    for caused in variables:
        if causing == caused:
            causality_matrix.loc[causing, caused] = np.nan
            continue
        result = grangercausalitytests(
            df_stationary[[caused, causing]], maxlag=4, verbose=False
        )
        p_values = [result[lag][0]["ssr_ftest"][1] for lag in range(1, 5)]
        causality_matrix.loc[causing, caused] = min(p_values)   # min p-value across lags

# Heatmap
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(causality_matrix.astype(float), annot=True, fmt=".3f",
            cmap="RdYlGn_r", vmin=0, vmax=0.1, ax=ax)
ax.set_title("Granger Causality p-values\n(row → col; green < 0.05 = significant)")
plt.tight_layout()
plt.show()
```

---

## 7. Impulse Response Functions

**Impulse Response Functions (IRF)** show how a **one-unit shock** to one variable propagates through the entire system over time.

```python
# Generate IRFs
irf = fitted.irf(periods=20)

# Plot IRF: effect of shock to "gdp_growth" on all variables
irf.plot(impulse="gdp_growth")
plt.suptitle("Impulse Response: Shock to GDP Growth")
plt.tight_layout()
plt.show()

# Plot with confidence intervals
irf.plot(impulse="gdp_growth", response="unemployment", orth=True)
plt.title("IRF: GDP Growth → Unemployment (orthogonalized)")
plt.show()
```

---

## 8. Forecast Error Variance Decomposition

**FEVD** shows what fraction of the forecast error variance of each variable is attributable to shocks from each variable in the system.

```python
# FEVD at 1, 4, 8, 12 periods ahead
fevd = fitted.fevd(periods=12)
fevd.plot(figsize=(13, 8))
plt.suptitle("Forecast Error Variance Decomposition", fontweight="bold")
plt.tight_layout()
plt.show()

# Numerical output
print(fevd.summary())
```

---

## 9. VAR vs. Separate Univariate Models

| Aspect | VAR | Separate ARIMA per variable |
|--------|-----|----------------------------|
| Cross-series relationships | ✅ Models them | ❌ Ignores them |
| Parameter count | High (K² × p) | Low (one per series) |
| Forecast accuracy | Better when cross-dynamics exist | Better when series are independent |
| Interpretability | Granger causality, IRF, FEVD | Model coefficients per series |
| Scalability | ❌ Breaks with many variables | ✅ Each series independent |
| Cointegrated series | Use VECM (not VAR on levels) | Use ARIMA on differences |

**When VAR wins**: economic indicators, energy markets, supply chain variables with clear cross-series dependencies and sufficient data (rule: T > 10 × K² × p).

---

## 10. VECM — Vector Error Correction Model

When multiple I(1) series are **cointegrated** (share a long-run equilibrium relationship), fitting VAR on first differences discards that long-run information. The **VECM** corrects this by including an error correction term that pulls the system back toward equilibrium.

### 10.1 Cointegration Intuition

```
Two I(1) series Y₁(t) and Y₂(t) are cointegrated if:
  e(t) = Y₁(t) - β·Y₂(t)  is I(0) (stationary)

Even though both series drift (random walk), their LINEAR COMBINATION is stationary.
This means they share a long-run equilibrium: they can't drift apart indefinitely.

Examples:
  - Short-term and long-term interest rates (yield curve)
  - Spot and futures prices for the same commodity
  - Exchange rates among major currencies
  - Electricity price and fuel cost
```

### 10.2 Johansen Cointegration Test

```python
from statsmodels.tsa.johansen import coint_johansen
import pandas as pd
import numpy as np

# Test for cointegration among I(1) variables
# All variables must be I(1) before this test
result = coint_johansen(
    df_levels[variables],   # DataFrame of levels (NOT differenced)
    det_order=0,            # 0=no intercept, 1=restricted intercept, -1=no constant
    k_ar_diff=2,            # number of lagged differences (like p-1 in VAR)
)

print("Johansen Cointegration Test")
print("=" * 50)
print("\nTrace Statistic:")
print(f"  {'Rank':<10} {'Trace stat':<15} {'Crit (5%)':<15} {'Cointegrated?':<15}")
for i in range(len(variables)):
    trace   = result.lr1[i]
    crit_5  = result.cvt[i, 1]     # 5% critical value
    is_coint = trace > crit_5
    print(f"  r ≤ {i:<8} {trace:<15.4f} {crit_5:<15.4f} {'✅ Yes' if is_coint else '❌ No'}")

print("\nCointegrating vectors (beta):")
print(pd.DataFrame(result.evec.T, columns=variables).round(4))
```

### 10.3 VECM Equation

```
VECM with r cointegrating relations:

  Δy(t) = α·βᵀ·y(t-1) + Γ₁·Δy(t-1) + ... + Γₚ₋₁·Δy(t-p+1) + ε(t)

Where:
  Δy(t)   = first difference of the K×1 vector (stationary)
  β       = K×r matrix of cointegrating vectors (the long-run equilibria)
  α       = K×r matrix of adjustment speeds (how fast each variable corrects)
  βᵀ·y(t-1) = r error correction terms (deviation from long-run equilibrium at t-1)
  Γᵢ      = K×K matrices of short-run dynamics
```

### 10.4 Implementation

```python
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

# Step 1: Determine cointegration rank
coint_rank = select_coint_rank(
    df_levels[variables],
    det_order=0,
    k_ar_diff=2,
    method="trace",
    signif=0.05,
)
print(f"Selected cointegration rank: {coint_rank.rank}")

# Step 2: Fit VECM
vecm = VECM(
    df_levels[variables],
    k_ar_diff=2,                 # number of lagged differences
    coint_rank=coint_rank.rank,  # number of cointegrating relations
    deterministic="ci",          # 'n'=none, 'co'=const in CE, 'ci'=intercept in CE
)
vecm_fitted = vecm.fit()
print(vecm_fitted.summary())

# Step 3: Forecast
forecast_vecm = vecm_fitted.predict(steps=12)
forecast_df = pd.DataFrame(
    forecast_vecm,
    columns=variables,
    index=pd.date_range(
        start=df_levels.index[-1] + df_levels.index.freq,
        periods=12,
        freq=df_levels.index.freq,
    )
)
print("\nVECM Forecast (original levels scale):")
print(forecast_df.head())
```

### 10.5 VAR vs. VECM Decision

```
Are all variables I(1)?  →  YES
  │
  ├── Run Johansen test for cointegration
  │
  ├── Cointegrated (r ≥ 1)?
  │   └── YES  → Use VECM (preserves long-run equilibrium information)
  │
  └── Not cointegrated (r = 0)?
      └── YES  → First-difference all series, fit VAR on Δy

Are variables I(0) (already stationary)?  →  Use VAR on levels
```

| Scenario | Correct Model |
|----------|---------------|
| All I(1), cointegrated | **VECM** |
| All I(1), not cointegrated | VAR on first differences |
| All I(0) | VAR on levels |
| Mixed I(0) and I(1) | Transform to I(0) first, then VAR |

---

*← [03 — AR/SARIMA Family](./03_ar_ma_arma_arima_sarima.md) | [Module README](./README.md) | Next: [05 — State Space Models](./05_state_space_models.md) →*
