# 01 — Multivariate Time Series Overview

> **Module**: 12 Multivariate & Advanced Topics | **File**: 1 of 6
>
> Real systems are rarely univariate. Temperature affects energy demand; traffic at one intersection propagates to the next; a tweet moves a stock. This note covers the structure of multivariate time series (MTS), dependency types, Vector Autoregression (VAR), and the foundational distinction between correlation and causality.

---

## Table of Contents

1. [What Is a Multivariate Time Series?](#1-what-is-a-multivariate-time-series)
2. [Dependency Structures](#2-dependency-structures)
3. [Cross-Correlation Analysis](#3-cross-correlation-analysis)
4. [Vector Autoregression (VAR)](#4-vector-autoregression-var)
5. [Correlation vs. Causality](#5-correlation-vs-causality)
6. [Dimensionality Challenges](#6-dimensionality-challenges)
7. [Implementation](#7-implementation)

---

## 1. What Is a Multivariate Time Series?

### 1.1 Formal Definition

```
A multivariate time series is a sequence of D-dimensional observations:
  X = {x₁, x₂, ..., xₙ}  where xₜ ∈ ℝᴰ

Each xₜ = [x₁ₜ, x₂ₜ, ..., xᴰₜ]ᵀ — a vector of D variables at time t.

Examples:
  D=3:   Weather station:    [temperature, humidity, pressure]
  D=6:   IMU sensor:         [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
  D=50:  Stock portfolio:    [price₁, price₂, ..., price₅₀]
  D=207: METR-LA traffic:   207 sensors × [speed, flow, occupancy]
```

### 1.2 Why Multivariate Matters

```
Univariate forecasting of y_t assumes:
  y_t depends only on {y_{t-1}, y_{t-2}, ..., y_{t-p}}

This IGNORES:
  - Leading indicators:  credit card spend predicts retail sales 2 weeks later
  - Contemporaneous:     temperature and cooling load move together
  - Lagged cross-effects: upstream sensor detects quality issue before downstream

Multivariate benefit:
  y_t depends on {y_{t-1}, ..., x_{t-1}, x_{t-2}, ...}
  → Exploits cross-series signal → lower forecast error
```

---

## 2. Dependency Structures

### 2.1 Types of Cross-Series Dependencies

```
1. CONTEMPORANEOUS (lag 0):
   x_t and y_t correlated at the same time step.
   Example: temperature and A/C load at the same hour.
   Detection: Pearson correlation at lag 0.

2. LEADING INDICATOR (lagged cross-correlation):
   x_{t-k} predicts y_t  (x leads y by k steps).
   Example: search query volume → product sales 2 weeks later.
   Detection: Cross-correlation function (CCF) at lag k.

3. FEEDBACK LOOP:
   x_t → y_{t+1}  AND  y_t → x_{t+1}  (bidirectional).
   Example: price and demand mutually influence each other.
   Detection: Granger test in both directions.

4. SPURIOUS CORRELATION:
   x_t and y_t both driven by hidden confounder z_t.
   Example: ice cream sales and drowning (both driven by summer).
   Cure: control for z_t or use causal discovery (PCMCI).

5. COINTEGRATION:
   Two I(1) series share a common stochastic trend.
   Example: stock and its derivative (pairs trading).
   Detection: Engle-Granger test, Johansen test.
```

### 2.2 Cross-Correlation Function

```python
import numpy as np
import pandas as pd

def compute_ccf(x: np.ndarray, y: np.ndarray, max_lag: int = 20) -> pd.DataFrame:
    """
    Cross-correlation function: x leads y if CCF(x,y, lag>0) is significant.

    Both series must be STATIONARY before computing CCF.
    Significance bounds: ±1.96/√N at 5% level.
    """
    n      = len(x)
    x_std  = (x - x.mean()) / (x.std() + 1e-12)
    y_std  = (y - y.mean()) / (y.std() + 1e-12)
    thresh = 1.96 / np.sqrt(n)

    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            corr = float(np.corrcoef(x_std[lag:], y_std[:-lag])[0, 1])
        elif lag < 0:
            corr = float(np.corrcoef(x_std[:lag], y_std[-lag:])[0, 1])
        else:
            corr = float(np.corrcoef(x_std, y_std)[0, 1])
        rows.append({"lag": lag, "ccf": round(corr, 4), "significant": abs(corr) > thresh})

    return pd.DataFrame(rows)
```

---

## 3. Cross-Correlation Analysis

### 3.1 Cointegration Test

```python
def johansen_cointegration_test(df, det_order=0, k_ar_diff=1):
    """
    Johansen test for cointegration among I(1) series.
    Rejects H₀ (r cointegration vectors) if trace_stat > cv_95%.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    result = coint_johansen(df, det_order=det_order, k_ar_diff=k_ar_diff)
    rows   = []
    for r in range(len(result.trace_stat)):
        rows.append({
            "H₀ (r≤)":   r,
            "trace_stat": round(result.trace_stat[r], 4),
            "cv_95%":     round(result.cvt[r, 1], 4),
            "reject_H₀":  result.trace_stat[r] > result.cvt[r, 1],
        })
    r_hat = sum(1 for row in rows if row["reject_H₀"])
    print(f"Johansen test: {r_hat} cointegrating relation(s)")
    return {"rank": r_hat, "table": pd.DataFrame(rows)}
```

---

## 4. Vector Autoregression (VAR)

### 4.1 Model Specification

```
VAR(p) model:
  xₜ = c + A₁·xₜ₋₁ + A₂·xₜ₋₂ + ... + Aₚ·xₜ₋ₚ + εₜ

  xₜ ∈ ℝᴰ, c ∈ ℝᴰ, Aᵢ ∈ ℝᴰˣᴰ, εₜ ~ N(0, Σ)

Parameter count: D² × p + D
  D=5, p=4: 5²×4 + 5 = 105 parameters → needs ≥ 1050 observations

Curse of dimensionality:
  D > 20 → VAR becomes intractable.
  Solutions: LASSO-VAR, Factor VAR, LightGBM with lag features.
```

### 4.2 Fitting VAR

```python
from statsmodels.tsa.vector_ar.var_model import VAR

def fit_var(df: pd.DataFrame, max_lags: int = 6) -> dict:
    """Fit VAR with BIC-selected lag order."""
    model   = VAR(df)
    sel     = model.select_order(max_lags)
    best_p  = max(1, sel.bic)
    best_p  = min(int(best_p), max_lags)
    results = model.fit(best_p, ic=None)

    print(f"VAR({best_p}) — AIC={results.aic:.2f}, BIC={results.bic:.2f}")
    print(f"Stable: {results.is_stable()}")
    return {"results": results, "lag_order": best_p}


def var_forecast(results, steps: int = 10) -> pd.DataFrame:
    """Generate VAR point forecasts."""
    fc = results.forecast(results.y, steps=steps)
    return pd.DataFrame(fc, columns=results.names)


def plot_irf(results, n_periods: int = 10, orth: bool = True) -> None:
    """Plot Impulse Response Functions."""
    import matplotlib.pyplot as plt
    irf = results.irf(n_periods)
    fig = irf.plot(orth=orth, figsize=(12, 8))
    fig.suptitle("Impulse Response Functions")
    plt.tight_layout(); plt.show()
```

---

## 5. Correlation vs. Causality

```
CORRELATION:         Cov(X,Y) ≠ 0  — symmetric, may be spurious.

GRANGER CAUSALITY:   X Granger-causes Y if past X reduces forecast error of Y.
                     Asymmetric, testable, but still ASSOCIATIONAL.

TRUE CAUSALITY:      P(Y | do(X=x)) ≠ P(Y | X=x)  (Pearl's do-calculus).
                     Requires randomized experiments or causal graph assumptions.

Ice cream sales and drowning CORRELATE (spurious — driven by summer).
Ice cream does NOT Granger-cause drowning if temperature is controlled for.
Granger causality is NOT equivalent to structural causation.
```

---

## 6. Dimensionality Challenges

```
Strategies for high-D MTS (D >> 20):

  Sparse VAR (LASSO-VAR):
    minimize ||y - Xβ||² + λ||β||₁
    → Most Aᵢ entries forced to 0 → scalable to D=50+

  Factor VAR:
    1. Reduce D → K (K=3-5) factors via PCA
    2. Fit VAR(p) on K factors
    → Interpretable, avoids parameter explosion

  Feature-based ML:
    Build lag features across all D series → LightGBM / XGBoost
    → Handles D=100s without explicit VAR structure

  Deep learning:
    TFT, PatchTST, Crossformer — attention over D series natively
    → D=100s–1000s with learned inter-series attention weights
```

---

## 7. Implementation

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR

def prepare_var_data(df: pd.DataFrame) -> pd.DataFrame:
    """ADF test each series; difference if non-stationary."""
    result = {}
    for col in df.columns:
        series = df[col].dropna()
        p      = adfuller(series)[1]
        if p < 0.05:
            result[col] = series
        else:
            diff = series.diff().dropna()
            result[f"d_{col}"] = diff
            print(f"  {col}: p={p:.4f} → differenced")
    min_len = min(len(v) for v in result.values())
    return pd.DataFrame({k: v.values[-min_len:] for k, v in result.items()})


if __name__ == "__main__":
    np.random.seed(42)
    n = 300
    x = np.cumsum(np.random.normal(0, 1, n))
    y = 0.5 * x + np.random.normal(0, 0.5, n)
    z = 0.3 * np.roll(y, 2) + np.random.normal(0, 0.3, n)

    df = pd.DataFrame({"x": x, "y": y, "z": z})
    df_stat = prepare_var_data(df)
    res = fit_var(df_stat, max_lags=5)
    fc  = var_forecast(res["results"], steps=5)
    print("\nForecast (5 steps):\n", fc.round(4).to_string())
```

---

*← [Module README](./README.md) | Next: [02 — Granger Causality](./02_granger_causality.md) →*
