# 06 — Model Selection & Diagnostics

> **Module**: 03 Statistical Models | **File**: 6 of 6
>
> Fitting a model is only half the work. Model selection (choosing between candidates) and diagnostics (validating the chosen model) are what separate reliable production forecasts from overfit notebooks.

---

## Table of Contents

1. [Information Criteria](#1-information-criteria)
2. [Residual Diagnostics](#2-residual-diagnostics)
3. [Ljung-Box Test (Formal)](#3-ljung-box-test-formal)
4. [Out-of-Sample Evaluation](#4-out-of-sample-evaluation)
5. [Statistical Model Comparison Tests](#5-statistical-model-comparison-tests)
6. [Overfitting and Parsimony](#6-overfitting-and-parsimony)
7. [Complete Model Selection Workflow](#7-complete-model-selection-workflow)

---

## 1. Information Criteria

### 1.1 AIC (Akaike Information Criterion)

```
AIC = -2·log(L) + 2k

Where:
  L = maximum likelihood of the fitted model
  k = number of estimated parameters

Lower AIC = Better model (balances fit and complexity)
```

**Interpretation**: AIC penalizes each extra parameter with a constant penalty of 2. Favors **goodness of fit** slightly more than parsimony.

### 1.2 BIC (Bayesian Information Criterion)

```
BIC = -2·log(L) + k·log(n)

Where:
  n = number of observations
  k = number of parameters

Lower BIC = Better model
```

BIC's penalty `k·log(n)` grows with sample size → **BIC is stricter than AIC** and prefers simpler models, especially for large `n`.

### 1.3 AICc (Corrected AIC)

```
AICc = AIC + 2k(k+1)/(n-k-1)

Recommended when n/k < 40 (small samples relative to parameters)
```

### 1.4 Using Information Criteria

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_and_score(train, order, seasonal_order=(0,0,0,0)):
    """Fit SARIMAX and return AIC/BIC/params."""
    try:
        m = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    trend="c").fit(disp=False, maxiter=200)
        return {"order": order, "seasonal": seasonal_order,
                "AIC": m.aic, "BIC": m.bic, "model": m}
    except Exception as e:
        return None

# Grid search over candidate models
candidates = []
for p in range(3):
    for q in range(3):
        for d in [0, 1]:
            result = fit_and_score(train_stationary, order=(p, d, q))
            if result:
                candidates.append(result)

# Rank by AIC
df_results = pd.DataFrame(
    [{"order": r["order"], "AIC": r["AIC"], "BIC": r["BIC"]} for r in candidates if r]
).sort_values("AIC")
print("Top 5 models by AIC:")
print(df_results.head(5))
```

### 1.5 AIC vs. BIC Guidance

| Criterion | When to Use |
|-----------|-------------|
| **AIC** | When forecasting accuracy is primary goal (predictive focus) |
| **AICc** | AIC for small samples (n/k < 40) — always safer |
| **BIC** | When you want the most parsimonious model (fewer parameters) |

> **Rule**: If AIC and BIC disagree, use AICc as a tie-breaker. Never choose a model with more parameters just because AIC is marginally lower.

---

## 2. Residual Diagnostics

### 2.1 What Should Residuals Look Like?

After fitting a well-specified model:
```
Residuals ε(t) should satisfy:
  1. E[ε(t)] ≈ 0            (zero mean — no systematic bias)
  2. Var[ε(t)] ≈ constant   (homoskedastic)
  3. Cov[ε(t), ε(s)] = 0    (no autocorrelation — white noise)
  4. ε(t) ≈ Normal          (for valid prediction intervals)
```

### 2.2 Full Diagnostic Plot

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats

def residual_diagnostics(fitted_model, model_name: str = ""):
    """Comprehensive residual diagnostic plot."""
    residuals = fitted_model.resid.dropna()
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # 1. Residual time plot
    axes[0, 0].plot(residuals, color="#2C7BB6", linewidth=0.8)
    axes[0, 0].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[0, 0].set_title("Residuals vs. Time")
    axes[0, 0].set_xlabel("Time")
    
    # 2. Histogram
    axes[0, 1].hist(residuals, bins=30, color="#2C7BB6", edgecolor="white", density=True)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()),
                    color="#D7191C", linewidth=2, label="Normal PDF")
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].legend()
    
    # 3. Q-Q plot
    stats.probplot(residuals, plot=axes[0, 2])
    axes[0, 2].set_title("Q-Q Plot (Normal)")
    
    # 4. ACF of residuals
    plot_acf(residuals, lags=30, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title("ACF of Residuals")
    
    # 5. PACF of residuals
    plot_pacf(residuals, lags=30, ax=axes[1, 1], alpha=0.05, method="ywm")
    axes[1, 1].set_title("PACF of Residuals")
    
    # 6. Residuals² (for heteroskedasticity)
    axes[1, 2].plot(residuals**2, color="#F07D00", linewidth=0.8)
    axes[1, 2].set_title("Residuals² (Variance Stability)")
    
    plt.suptitle(f"Residual Diagnostics — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"diagnostics_{model_name.replace(' ','_')}.png", bbox_inches="tight")
    plt.show()
    
    # Numerical summary
    print(f"\n{'='*50}")
    print(f"  Residual Statistics — {model_name}")
    print(f"{'='*50}")
    print(f"  Mean:     {residuals.mean():.4f}  (should be ≈ 0)")
    print(f"  Std:      {residuals.std():.4f}")
    print(f"  Skewness: {stats.skew(residuals):.4f}  (|skew| < 0.5 = good)")
    print(f"  Kurtosis: {stats.kurtosis(residuals):.4f}  (≈ 0 = normal tails)")
    _, p_normal = stats.normaltest(residuals)
    print(f"  Normality test p-value: {p_normal:.4f}  ({'✅' if p_normal > 0.05 else '⚠️'})")


# Usage
residual_diagnostics(fitted_sarima, "SARIMA(1,1,1)(1,1,1)[12]")
```

---

## 3. Ljung-Box Test (Formal)

### 3.1 Test for Residual Autocorrelation

```
H₀: Residuals are uncorrelated up to lag h (white noise)
H₁: At least one autocorrelation ≠ 0

Test statistic:
  Q_LB = n(n+2) Σₖ₌₁ʰ [ρ²(k) / (n-k)]  ~  χ²(h - p - q)

Decision: p-value > 0.05 → Fail to reject H₀ → Residuals are white noise ✅
```

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

residuals = fitted.resid.dropna()

# Test at multiple lags
lb_results = acorr_ljungbox(
    residuals,
    lags=[5, 10, 15, 20],
    return_df=True,
    model_df=2,   # adjust degrees of freedom: subtract p+q from AR+MA model
)

print("Ljung-Box Test Results:")
print(lb_results.to_string())
print("\nAll p-values > 0.05?", (lb_results["lb_pvalue"] > 0.05).all())
```

### 3.2 Interpreting Ljung-Box

| p-value | Interpretation | Action |
|---------|---------------|--------|
| > 0.10 | ✅ Strong evidence of white noise | Model is well-specified |
| 0.05–0.10 | ⚠️ Marginal | Consider increasing p or q |
| < 0.05 | ❌ Significant autocorrelation remaining | Model is under-specified — revise orders |

---

## 4. Out-of-Sample Evaluation

### 4.1 Walk-Forward Validation

The gold standard for evaluating time series models in production:

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def walk_forward_validate(
    series: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    n_test: int,
    n_origins: int = 10,
    h: int = 1,
) -> pd.DataFrame:
    """
    Walk-forward (rolling origin) validation.
    
    At each origin:
      - Train on all data up to origin
      - Forecast h steps ahead
      - Record actual vs. predicted
    """
    results = []
    test_start = len(series) - n_test
    
    for i in range(n_origins):
        # Determine train end
        train_end = test_start + i * (n_test // n_origins)
        if train_end + h > len(series):
            break
        
        train = series.iloc[:train_end]
        actual = series.iloc[train_end : train_end + h]
        
        # Fit model
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order).fit(disp=False)
        forecast = model.get_forecast(steps=h).predicted_mean
        
        mae  = (actual.values - forecast.values[:len(actual)]).mean()
        rmse = np.sqrt(((actual.values - forecast.values[:len(actual)]) ** 2).mean())
        
        results.append({
            "origin": train.index[-1],
            "MAE":    abs(mae),
            "RMSE":   rmse,
            "horizon": h,
        })
    
    return pd.DataFrame(results)

# Run walk-forward validation
wf_results = walk_forward_validate(
    series=train_series,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    n_test=24,
    n_origins=8,
    h=6,
)
print(wf_results)
print(f"\nMean RMSE across origins: {wf_results['RMSE'].mean():.3f}")
```

### 4.2 Key Forecast Metrics

```python
def compute_forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Standard forecast accuracy metrics."""
    errors = actual - predicted
    ae = np.abs(errors)
    
    mae  = ae.mean()
    rmse = np.sqrt((errors ** 2).mean())
    mape = (ae / (np.abs(actual) + 1e-8)).mean() * 100
    
    # MASE: Scale by in-sample naive (seasonal) errors
    # (simplified: scale by mean absolute difference)
    mase = mae / np.abs(np.diff(actual)).mean()
    
    return {
        "MAE":    round(mae, 4),
        "RMSE":   round(rmse, 4),
        "MAPE":   round(mape, 4),
        "MASE":   round(mase, 4),
    }

metrics = compute_forecast_metrics(test.values, forecast_mean.values)
for k, v in metrics.items():
    print(f"  {k}: {v}")
```

---

## 5. Statistical Model Comparison Tests

### 5.1 Diebold-Mariano Test

Formally tests whether **two forecasts are significantly different** in accuracy:

```
H₀: Both forecasts have equal predictive accuracy
H₁: One forecast is significantly better than the other

Test statistic: DM = d̄ / sqrt(V̂(d̄))
Where d(t) = L(e₁(t)) - L(e₂(t))  (loss difference per period)

p-value < 0.05 → Significant difference in accuracy
```

```python
from statsmodels.stats.stattools import durbin_watson
import numpy as np
from scipy import stats

def diebold_mariano_test(
    actual: np.ndarray,
    forecast1: np.ndarray,
    forecast2: np.ndarray,
    loss: str = "mse",
) -> dict:
    """
    Diebold-Mariano test for equal predictive accuracy.
    loss: 'mse' or 'mae'
    """
    e1 = actual - forecast1
    e2 = actual - forecast2
    
    if loss == "mse":
        d = e1**2 - e2**2
    else:
        d = np.abs(e1) - np.abs(e2)
    
    n = len(d)
    d_bar = d.mean()
    d_var = ((d - d_bar) ** 2).sum() / (n * (n - 1))
    
    dm_stat = d_bar / np.sqrt(d_var)
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n-1))
    
    return {
        "DM statistic": round(dm_stat, 4),
        "p-value":      round(p_value, 4),
        "significant":  p_value < 0.05,
        "better model": "Model 1" if dm_stat > 0 else "Model 2",
    }

dm = diebold_mariano_test(test.values, forecast_arima.values, forecast_ets.values)
print("Diebold-Mariano Test (ARIMA vs. ETS):")
for k, v in dm.items():
    print(f"  {k}: {v}")
```

---

## 6. Overfitting and Parsimony

### 6.1 Signs of Overfitting in ARIMA

- High in-sample R² but poor out-of-sample RMSE
- AIC keeps decreasing but BIC levels off or increases
- Large, unstable coefficient estimates
- Near-cancelling AR and MA roots (e.g., AR root near MA root)
- Very wide prediction intervals relative to the series range

### 6.2 Near-Cancelling Roots

```python
# Check for near-cancelling AR and MA roots — sign of over-parameterization
ar_roots = fitted.arroots   # roots of AR polynomial
ma_roots = fitted.maroots   # roots of MA polynomial

print("AR roots (should be outside unit circle):", ar_roots)
print("MA roots (should be outside unit circle):", ma_roots)

# If |AR root| ≈ |MA root|, the model is over-parameterized
# Solution: reduce p or q by 1
```

### 6.3 The Parsimony Principle

> **Occam's Razor for ARIMA**: Among models with similar out-of-sample performance, prefer the one with fewer parameters.

- ARIMA(1,1,1) is usually preferred over ARIMA(3,1,3) if AIC is similar
- Check: Is each coefficient statistically significant? (`|t-stat| > 2`)

```python
# Check significance of each coefficient
print(fitted.summary().tables[1])
# t-statistics should be > |2| for each parameter to be worth keeping
```

---

## 7. Complete Model Selection Workflow

```
1. PREPARE
   ├── Check stationarity (ADF + KPSS)
   ├── Apply transformations (log, diff, seasonal diff)
   └── Split train/validation/test chronologically

2. IDENTIFY CANDIDATES
   ├── Plot ACF + PACF → suggest initial (p, d, q)
   ├── Fit candidates (p ∈ {0,1,2}, q ∈ {0,1,2}) × (P,D,Q variants)
   └── Rank by AIC/BIC → shortlist top 3 models

3. ESTIMATE (via MLE)
   ├── Fit each candidate on training data
   └── Note AIC, BIC, coefficient significance

4. DIAGNOSE RESIDUALS
   ├── Residual time plot → no patterns
   ├── ACF/PACF of residuals → all inside bands
   ├── Ljung-Box p-value > 0.05
   └── Q-Q plot approximately linear

5. OUT-OF-SAMPLE EVALUATION
   ├── Walk-forward validation on validation set
   ├── Compute MAE, RMSE, MAPE, MASE
   └── Diebold-Mariano test vs. seasonal naive

6. FINAL SELECTION
   ├── Best out-of-sample RMSE AND passes diagnostics
   └── If tied → prefer simpler model (BIC, parsimony)

7. REFIT ON FULL DATA & DEPLOY
   ├── Refit winning model on train + validation
   └── Generate production forecast with prediction intervals
```

---

*← [05 — State Space Models](./05_state_space_models.md) | [Module README](./README.md) | Next Module: [04 — ML for TS](../04_ml_for_time_series/README.md) →*
