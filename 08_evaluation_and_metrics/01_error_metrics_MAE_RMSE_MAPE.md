# 01 — Error Metrics: MAE, RMSE, MAPE & Variants

> **Module**: 08 Evaluation & Metrics | **File**: 1 of 5
>
> Choosing the wrong metric leads to selecting the wrong model. This note covers every standard point-forecast metric — their formulas, statistical properties, edge cases, and failure modes — so you can pick the right one for your domain.

---

## Table of Contents

1. [Why Metric Choice Matters](#1-why-metric-choice-matters)
2. [MAE — Mean Absolute Error](#2-mae--mean-absolute-error)
3. [MSE and RMSE](#3-mse-and-rmse)
4. [MAPE — Mean Absolute Percentage Error](#4-mape--mean-absolute-percentage-error)
5. [SMAPE — Symmetric MAPE](#5-smape--symmetric-mape)
6. [MdAE and MdAPE — Median Variants](#6-mdae-and-mdape--median-variants)
7. [Metric Comparison and Selection Guide](#7-metric-comparison-and-selection-guide)
8. [Implementation from Scratch](#8-implementation-from-scratch)

---

## 1. Why Metric Choice Matters

### 1.1 The Same Model Can Look Very Different Across Metrics

```
Model A vs. Model B on the same test set:

  Point errors: [2, 2, 2, 2, 100]

  Model A: MAE  = 21.6  ← heavily influenced by the single large error
           RMSE = 45.1  ← even more influenced (squares the 100)
           MAPE = 10%   ← might be fine if actuals are large

  Model B: MAE  = 5     ← 5 small errors of 5 each
           RMSE = 5
           MAPE = 100%  ← terrible if actuals are near zero

→ The model you pick depends entirely on which metric you optimize.
```

### 1.2 The Optimization–Metric Alignment Principle

```
If you train with MSE loss  → model targets the conditional MEAN
If you train with MAE loss  → model targets the conditional MEDIAN
If you train with pinball   → model targets a specific QUANTILE

Always align your training loss with your evaluation metric.
Misalignment = suboptimal model for your actual objective.
```

---

## 2. MAE — Mean Absolute Error

### 2.1 Formula

```
MAE = (1/n) · Σᵢ |yᵢ - ŷᵢ|

Where:
  yᵢ  = actual value at time i
  ŷᵢ  = predicted value at time i
  n   = number of forecast periods
```

### 2.2 Properties

| Property               | Detail                                                     |
|------------------------|------------------------------------------------------------|
| **Unit**               | Same as the target variable (interpretable)                |
| **Outlier sensitivity**| Low — all errors weighted equally (L1 norm)                |
| **Optimal predictor**  | Conditional median of y given x                            |
| **Training loss**      | `mae` / `regression_l1` in LightGBM; `loss='mae'` in sklearn |
| **Scale dependence**   | ✅ — cannot compare across series with different scales     |
| **Zero actuals**       | ✅ Safe — no division by actual values                      |

### 2.3 Interpretation

```
MAE = 50 means: on average, the forecast is off by 50 units.

Naive benchmark comparison:
  If the naïve forecast (ŷ = yₜ₋₁) gives MAE = 100,
  and your model gives MAE = 50 → model is 2× better than naïve.
  (This comparison is formalized as MASE — see Note 02)
```

### 2.4 When to Use MAE

- ✅ When large errors are **not disproportionately costly** (symmetric loss)
- ✅ When the data has **outliers** you don't want to dominate the metric
- ✅ When the target **can include zeros**
- ✅ As the primary training loss for robust models
- ❌ When error scale varies widely across series (use MASE instead)

---

## 3. MSE and RMSE

### 3.1 Formulas

```
MSE  = (1/n) · Σᵢ (yᵢ - ŷᵢ)²

RMSE = √MSE = √[ (1/n) · Σᵢ (yᵢ - ŷᵢ)² ]
```

### 3.2 Properties

| Property               | Detail                                                      |
|------------------------|-------------------------------------------------------------|
| **Unit**               | RMSE: same as target; MSE: squared units (less interpretable)|
| **Outlier sensitivity**| High — squares large errors (L2 norm)                       |
| **Optimal predictor**  | Conditional mean of y given x                               |
| **Training loss**      | Default in most frameworks (`mse`, `reg:squarederror`)      |
| **Scale dependence**   | ✅ — scale-dependent                                         |
| **Decomposition**      | `MSE = Bias² + Variance` (bias-variance tradeoff visible)   |

### 3.3 RMSE vs. MAE

```
RMSE ≥ MAE   always (by Cauchy-Schwarz inequality)

If RMSE >> MAE:
  → A few very large errors dominate RMSE
  → Model has intermittent large mistakes (worth investigating)

If RMSE ≈ MAE:
  → Errors are roughly uniform in magnitude (no outlier events)

Ratio diagnostic:
  RMSE / MAE  → 1.0:  uniform errors
  RMSE / MAE  → 1.4:  normally distributed errors
  RMSE / MAE  >> 2.0: large outlier errors present
```

### 3.4 MSE Decomposition

```
MSE = Bias² + Variance + Irreducible Noise

Where:
  Bias²     = (E[ŷ] - E[y])²       — systematic over/under-prediction
  Variance  = E[(ŷ - E[ŷ])²]       — prediction spread
  Noise     = E[(y - E[y])²]        — irreducible (aleatoric uncertainty)

Implication:
  A model that always predicts the mean has:
    Bias = 0, Variance = 0 → MSE = Noise (lower bound)
  But it's useless — MASE or skill score needed to verify improvement
```

### 3.5 When to Use RMSE

- ✅ When **large errors are particularly harmful** (e.g., energy demand → grid failure)
- ✅ As training loss when the conditional mean is the target
- ✅ In competitions (M4, Kaggle) where RMSE/MSE is the scoring criterion
- ❌ When data has outliers that dominate — use MAE instead
- ❌ For scale comparison across series

---

## 4. MAPE — Mean Absolute Percentage Error

### 4.1 Formula

```
MAPE = (100/n) · Σᵢ |yᵢ - ŷᵢ| / |yᵢ|

Expressed as a percentage (multiply by 100 for % display).
```

### 4.2 Properties

| Property               | Detail                                                         |
|------------------------|----------------------------------------------------------------|
| **Unit**               | Percentage — scale-free, comparable across series              |
| **Range**              | [0%, +∞)                                                       |
| **Optimal predictor**  | Conditional median of log(y) (not mean of y)                   |
| **Zero actuals**       | ❌ **Undefined** — division by zero when yᵢ = 0               |
| **Near-zero actuals**  | ❌ **Explodes** — 1% actual with 2% forecast → 100% error     |
| **Asymmetry**          | ❌ Penalizes over-forecasting more than under-forecasting       |

### 4.3 The Asymmetry Problem

```
Actual y = 100:
  Forecast ŷ = 150  → error = |100-150|/100 = 50%
  Forecast ŷ = 50   → error = |100-50|/100  = 50%
  → Symmetric at 50%

Actual y = 100:
  Forecast ŷ = 200  → error = 100%   (over-forecast by 100%)
  Forecast ŷ = 0    → error = 100%   (under-forecast by 100%)
  → Still symmetric

BUT at the bounded extreme:
  Forecast ŷ = 0    → max error = 100%  (cannot under-predict more than 100%)
  Forecast ŷ = ∞   → error = ∞         (over-prediction is unbounded)

→ MAPE systematically FAVORS under-forecasting models.
→ A model trained to minimize MAPE will learn to predict too low.
```

### 4.4 When to Use MAPE

- ✅ When all actual values are **strictly positive** and well away from zero
- ✅ For communicating forecast accuracy to business stakeholders (intuitive %)
- ✅ When relative errors matter equally regardless of scale
- ❌ Retail demand with intermittent zeros
- ❌ Any domain with near-zero actuals (financial returns, energy generation at night)

---

## 5. SMAPE — Symmetric MAPE

### 5.1 Formula

```
SMAPE = (100/n) · Σᵢ |yᵢ - ŷᵢ| / (|yᵢ| + |ŷᵢ|) / 2

       = (200/n) · Σᵢ |yᵢ - ŷᵢ| / (|yᵢ| + |ŷᵢ|)
```

### 5.2 Properties

| Property               | Detail                                                         |
|------------------------|----------------------------------------------------------------|
| **Range**              | [0%, 200%]                                                     |
| **Symmetry**           | Partially symmetric — penalizes over/under more evenly        |
| **Zero actuals**       | ⚠️ When yᵢ = 0 AND ŷᵢ = 0: undefined (0/0); when only one = 0: 200% |
| **Used in**            | M4 competition, standard benchmark comparison                  |

### 5.3 SMAPE vs. MAPE Behaviour

```
Actual y = 100, Forecast ŷ = 150:
  MAPE  = |100-150|/100           = 50%
  SMAPE = 2·|100-150|/(100+150)   = 40%  ← lower

Actual y = 100, Forecast ŷ = 50:
  MAPE  = |100-50|/100            = 50%
  SMAPE = 2·|100-50|/(100+50)     = 66.7%  ← higher

→ SMAPE penalizes under-forecasting more than MAPE
→ Neither is truly symmetric — MASE is preferred
```

---

## 6. MdAE and MdAPE — Median Variants

### 6.1 Formulas

```
MdAE  = median(|yᵢ - ŷᵢ|)

MdAPE = median(|yᵢ - ŷᵢ| / |yᵢ|) × 100
```

### 6.2 Use Case

Median-based metrics are **maximally robust to outliers** — a single catastrophically bad forecast step cannot inflate the metric:

```
Errors: [1, 2, 1, 3, 1000]

MAE   = 201.4  ← dominated by the 1000
MdAE  = 2.0    ← unaffected by the 1000

→ MdAE is preferred for evaluating models on noisy data
   where occasional extreme errors are expected but not representative
```

---

## 7. Metric Comparison and Selection Guide

### 7.1 Full Comparison Table

| Metric | Unit | Outlier Robust | Zero-Safe | Scale-Free | Symmetric | Recommended For |
|--------|------|---------------|-----------|-----------|-----------|-----------------|
| MAE    | Target | ✅ | ✅ | ❌ | ✅ | General purpose, single series |
| MSE    | Target² | ❌ | ✅ | ❌ | ✅ | Training loss, penalize large errors |
| RMSE   | Target | ❌ | ✅ | ❌ | ✅ | Same unit as MAE, harsher on outliers |
| MAPE   | %    | ❌ | ❌ | ✅ | ❌ | Positive targets, business reporting |
| SMAPE  | %    | ❌ | ⚠️ | ✅ | Partial | Competition benchmarking (M4) |
| MdAE   | Target | ✅✅ | ✅ | ❌ | ✅ | Noisy data, robust evaluation |
| MASE   | Ratio | ✅ | ✅ | ✅ | ✅ | **Multi-series comparison (recommended)** |

### 7.2 Domain-Specific Recommendations

| Domain                   | Recommended Metric | Reason                                  |
|--------------------------|--------------------|-----------------------------------------|
| Retail demand (no zeros) | MAPE or MASE      | Relative errors matter across SKUs      |
| Retail demand (with zeros)| MASE or RMSE     | MAPE breaks on zero actuals             |
| Energy (MWh)             | RMSE or MAE       | Large errors are physically costly      |
| Financial returns        | RMSE (annualized) | Near-zero actuals — MAPE unusable       |
| Macroeconomics           | MASE              | Short series, varied scales             |
| Medical / safety-critical| RMSE              | Under-prediction risk is asymmetric     |

---

## 8. Implementation from Scratch

```python
import numpy as np
import pandas as pd
from typing import Union

Array = Union[np.ndarray, pd.Series]


def mae(y_true: Array, y_pred: Array) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: Array, y_pred: Array) -> float:
    """Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: Array, y_pred: Array) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: Array, y_pred: Array, eps: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.

    Parameters
    ----------
    eps : small constant added to denominator to avoid division by zero.
          Set to 0 for strict MAPE (will raise on zero actuals).

    Returns
    -------
    MAPE as a decimal (0.15 = 15%). Multiply by 100 for percentage.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if eps == 0 and np.any(y_true == 0):
        raise ValueError("MAPE undefined for zero actuals. Use eps > 0 or SMAPE/MASE.")
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)))


def smape(y_true: Array, y_pred: Array, eps: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    Returns decimal [0, 2]. Multiply by 100 for percentage [0%, 200%].
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def mdae(y_true: Array, y_pred: Array) -> float:
    """Median Absolute Error — robust to outliers."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.median(np.abs(y_true - y_pred)))


def evaluate_all(
    y_true: Array,
    y_pred: Array,
    label: str = "Model",
    eps: float = 1e-8,
) -> pd.Series:
    """
    Compute all standard point forecast metrics in one call.

    Parameters
    ----------
    y_true : actual values
    y_pred : predicted values
    label  : model name (used as Series name)
    eps    : epsilon for MAPE/SMAPE denominator

    Returns
    -------
    pd.Series with metric names as index
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors      = y_true - y_pred
    abs_errors  = np.abs(errors)

    results = {
        "MAE":        float(abs_errors.mean()),
        "MdAE":       float(np.median(abs_errors)),
        "MSE":        float((errors**2).mean()),
        "RMSE":       float(np.sqrt((errors**2).mean())),
        "MAPE (%)":   float(100 * np.mean(abs_errors / (np.abs(y_true) + eps))),
        "SMAPE (%)":  float(100 * np.mean(abs_errors / ((np.abs(y_true) + np.abs(y_pred)) / 2 + eps))),
        "Max Error":  float(abs_errors.max()),
        "RMSE/MAE":   float(np.sqrt((errors**2).mean()) / (abs_errors.mean() + 1e-12)),
    }
    return pd.Series(results, name=label)


# ─── Example Usage ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    n      = 200
    y_true = 100 + 10 * np.sin(2 * np.pi * np.arange(n) / 12) + np.random.normal(0, 2, n)
    y_naive = np.roll(y_true, 1); y_naive[0] = y_true[0]   # naïve: y_t = y_{t-1}
    y_model = y_true + np.random.normal(0, 1.5, n)          # better model

    metrics_naive = evaluate_all(y_true, y_naive, label="Naïve")
    metrics_model = evaluate_all(y_true, y_model, label="Model")

    comparison = pd.DataFrame([metrics_naive, metrics_model]).T
    print(comparison.round(4))
    print(f"\nModel RMSE/MAE ratio: {metrics_model['RMSE/MAE']:.2f}")
    print(" → Ratio near 1.4 suggests approximately Gaussian errors")
```

---

*← [Module README](./README.md) | Next: [02 — Skill Scores & Relative Metrics](./02_skill_scores_and_relative_metrics.md) →*
