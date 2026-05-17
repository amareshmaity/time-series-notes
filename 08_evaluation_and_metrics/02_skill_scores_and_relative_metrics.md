# 02 — Skill Scores and Relative Metrics

> **Module**: 08 Evaluation & Metrics | **File**: 2 of 5
>
> A model that achieves MAE = 50 might be excellent or terrible — it depends on how hard the problem is. Skill scores and relative metrics answer the key question: **is this model actually better than a naïve baseline?** MASE is the industry standard for cross-series comparison.

---

## Table of Contents

1. [The Baseline Problem](#1-the-baseline-problem)
2. [MASE — Mean Absolute Scaled Error](#2-mase--mean-absolute-scaled-error)
3. [wMAPE — Weighted MAPE](#3-wmape--weighted-mape)
4. [Skill Scores](#4-skill-scores)
5. [OWA — Overall Weighted Average (M4 Standard)](#5-owa--overall-weighted-average-m4-standard)
6. [Relative RMSE and Relative MAE](#6-relative-rmse-and-relative-mae)
7. [Multi-Series Aggregation](#7-multi-series-aggregation)
8. [Implementation](#8-implementation)

---

## 1. The Baseline Problem

### 1.1 Why Raw Metrics Are Not Enough

```
Series A (high-frequency, stable):  MAE = 50
Series B (noisy, volatile):         MAE = 200

Is Series B's model worse? Not necessarily.
If the naïve baseline gives MAE = 500 on Series B and MAE = 48 on Series A,
then Series B's model is actually much better (200/500 vs 50/48).

→ We need metrics that account for how hard the forecasting problem is.
```

### 1.2 Naïve Baselines

| Baseline                 | Forecast        | Best For                         |
|--------------------------|-----------------|----------------------------------|
| **Naïve (random walk)**  | ŷₜ = yₜ₋₁      | Non-seasonal, non-stationary     |
| **Seasonal naïve**       | ŷₜ = yₜ₋ₛ      | Seasonal series (s = season length)|
| **Mean**                 | ŷₜ = ȳ_train   | Stationary series                |
| **Drift**                | ŷₜ = yₜ₋₁ + slope | Trending series              |

---

## 2. MASE — Mean Absolute Scaled Error

### 2.1 Formula

```
MASE = MAE_model / MAE_naïve_in_sample

Where:
  MAE_model   = (1/h) · Σₜ₌T+1^{T+h} |yₜ - ŷₜ|          (test set errors)

  MAE_naïve   = 1/(T-1) · Σₜ₌2^T |yₜ - yₜ₋₁|            (non-seasonal)
  or
  MAE_naïve   = 1/(T-s) · Σₜ₌s+1^T |yₜ - yₜ₋ₛ|           (seasonal, period s)

Where:
  T = training set length
  h = forecast horizon
  s = seasonal period
```

**Hyndman & Koehler (2006)** — MASE is recommended as the primary metric for forecasting competitions and multi-series benchmarks.

### 2.2 Interpretation

```
MASE < 1.0  → Model outperforms naïve baseline  ✅
MASE = 1.0  → Model is equivalent to naïve      ⚠️
MASE > 1.0  → Model is WORSE than naïve         ❌

MASE = 0.65 means: model's MAE is 35% lower than the naïve MAE.
MASE = 1.20 means: model's MAE is 20% higher than the naïve MAE — discard model!
```

### 2.3 Why MASE Is Preferred

| Property                    | MAE | MAPE | MASE |
|-----------------------------|-----|------|------|
| Scale-free                  | ❌  | ✅   | ✅   |
| Zero-safe                   | ✅  | ❌   | ✅   |
| Comparable across series    | ❌  | ✅   | ✅   |
| Defined for all y           | ✅  | ❌   | ✅   |
| Symmetric                   | ✅  | ❌   | ✅   |
| Benchmarks against naïve    | ❌  | ❌   | ✅   |
| Used in M competitions      | —   | M3   | M4 ✅|

### 2.4 MASE for Seasonal Series

```python
import numpy as np

def mase(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    seasonality: int = 1,
) -> float:
    """
    Mean Absolute Scaled Error.

    Parameters
    ----------
    y_train    : training series (used to compute naïve MAE scaling factor)
    y_test     : actual test values
    y_pred     : model forecasts
    seasonality: seasonal period s (1 = non-seasonal naïve, 7 = weekly, 12 = monthly)

    Returns
    -------
    MASE score (< 1.0 = better than naïve)
    """
    # In-sample naïve MAE (denominator)
    if seasonality == 1:
        naive_errors = np.abs(np.diff(y_train))          # yₜ - yₜ₋₁
    else:
        naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])

    if len(naive_errors) == 0 or naive_errors.mean() == 0:
        raise ValueError("Naïve MAE is zero — cannot compute MASE (constant training series).")

    mae_naive = naive_errors.mean()
    mae_model = np.abs(y_test - y_pred).mean()

    return float(mae_model / mae_naive)
```

### 2.5 Multi-Horizon MASE

```python
def mase_per_horizon(
    y_train: np.ndarray,
    y_test_matrix: np.ndarray,
    y_pred_matrix: np.ndarray,
    seasonality: int = 1,
) -> np.ndarray:
    """
    Compute MASE for each forecast horizon step independently.

    Parameters
    ----------
    y_train       : training series
    y_test_matrix : (n_origins, H) matrix of actual test values
    y_pred_matrix : (n_origins, H) matrix of predictions
    seasonality   : seasonal period

    Returns
    -------
    mase_per_step : (H,) array of MASE values
    """
    if seasonality == 1:
        naive_mae = np.abs(np.diff(y_train)).mean()
    else:
        naive_mae = np.abs(y_train[seasonality:] - y_train[:-seasonality]).mean()

    H = y_test_matrix.shape[1]
    return np.array([
        np.abs(y_test_matrix[:, h] - y_pred_matrix[:, h]).mean() / naive_mae
        for h in range(H)
    ])
```

---

## 3. wMAPE — Weighted MAPE

### 3.1 Formula

```
wMAPE = Σᵢ |yᵢ - ŷᵢ| / Σᵢ |yᵢ|

       = Total absolute error / Total actual volume
```

### 3.2 Motivation

Standard MAPE gives equal weight to each time period. In retail, a forecast error during peak season (Christmas, Diwali) matters far more than a mid-season error. wMAPE naturally down-weights low-volume periods:

```
If yᵢ = 1 (low season):   contributes 1/Σyᵢ weight  → tiny
If yᵢ = 1000 (peak):      contributes 1000/Σyᵢ weight → dominant

→ wMAPE is volume-weighted: errors in high-volume periods dominate
→ A 10% error during peak costs 10× more than a 10% error at low volume
```

### 3.3 Properties

| Property               | Detail                                              |
|------------------------|-----------------------------------------------------|
| **Range**              | [0%, ∞)                                             |
| **Zero actuals**       | ✅ Safe — zeros contribute 0 to both numerator and denominator |
| **Asymmetry**          | Same as MAPE — over-forecasting slightly penalized  |
| **Industry use**       | Supply chain, retail (Walmart, Amazon internal KPI) |
| **Comparison**         | Lower than MAPE when errors cluster at high volumes  |

```python
def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Mean Absolute Percentage Error.

    Returns decimal (0.12 = 12%). Multiply by 100 for percentage.
    Zero actuals contribute zero weight — safe for intermittent series.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    total_actual = np.abs(y_true).sum()
    if total_actual == 0:
        raise ValueError("wMAPE undefined when all actuals are zero.")
    return float(np.abs(y_true - y_pred).sum() / total_actual)
```

---

## 4. Skill Scores

### 4.1 General Skill Score Formula

A **skill score** normalizes model performance relative to a reference (baseline):

```
Skill = 1 - (Score_model / Score_reference)

Where Score is any error metric (MAE, RMSE, etc.) — lower = better.

Interpretation:
  Skill = 0.30 → model is 30% better than baseline
  Skill = 0.0  → model equals baseline (no skill)
  Skill < 0    → model is WORSE than baseline
  Skill = 1.0  → perfect forecast
```

### 4.2 Implementations

```python
import numpy as np

def skill_score(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    metric_fn=None,
) -> float:
    """
    General skill score: how much better is model vs. baseline?

    Parameters
    ----------
    y_true          : actual values
    y_pred_model    : model predictions
    y_pred_baseline : baseline predictions (naïve, climatology, etc.)
    metric_fn       : error metric function (default: MAE)

    Returns
    -------
    skill : float in (-inf, 1.0]
            0 = no skill, 1 = perfect, < 0 = worse than baseline
    """
    if metric_fn is None:
        metric_fn = lambda a, b: float(np.mean(np.abs(a - b)))

    score_model    = metric_fn(y_true, y_pred_model)
    score_baseline = metric_fn(y_true, y_pred_baseline)

    if score_baseline == 0:
        return 1.0 if score_model == 0 else float("-inf")

    return float(1 - score_model / score_baseline)


def naive_forecast(y_train: np.ndarray, h: int, seasonality: int = 1) -> np.ndarray:
    """
    Generate naïve baseline forecasts.

    seasonality=1 : random walk (ŷ = last observed value, repeated h times)
    seasonality>1 : seasonal naïve (ŷ = same period from last season)
    """
    if seasonality == 1:
        return np.full(h, y_train[-1])
    else:
        # Repeat last full season
        tail = y_train[-seasonality:]
        reps = (h // seasonality) + 1
        return np.tile(tail, reps)[:h]
```

### 4.3 Percent Better Statistic

```python
def percent_better(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
) -> float:
    """
    Fraction of time steps where model error < baseline error.

    A simple non-parametric skill indicator.
    Complements the Diebold-Mariano test (see Note 04).
    """
    model_abs    = np.abs(y_true - y_pred_model)
    baseline_abs = np.abs(y_true - y_pred_baseline)
    return float((model_abs < baseline_abs).mean())
```

---

## 5. OWA — Overall Weighted Average (M4 Standard)

### 5.1 Formula

Used by the **M4 Competition** as the primary ranking metric:

```
OWA = 0.5 · (MASE / MASE_naïve2) + 0.5 · (SMAPE / SMAPE_naïve2)

Where naïve2 = seasonal naïve forecast (the competition's reference model)

OWA < 1 → better than seasonal naïve
OWA = 1 → equal to seasonal naïve
```

### 5.2 Implementation

```python
def owa(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_naive2: np.ndarray,
    seasonality: int = 1,
    eps: float = 1e-8,
) -> float:
    """
    Overall Weighted Average metric from the M4 Competition.

    Parameters
    ----------
    y_train    : training series (for MASE scaling)
    y_test     : actual test values
    y_pred     : model predictions
    y_naive2   : seasonal naïve predictions (the benchmark)
    seasonality: seasonal period
    eps        : epsilon for SMAPE denominator

    Returns
    -------
    OWA score (< 1.0 = better than seasonal naïve)
    """
    def _mase(pred):
        if seasonality == 1:
            scale = np.abs(np.diff(y_train)).mean()
        else:
            scale = np.abs(y_train[seasonality:] - y_train[:-seasonality]).mean()
        return np.abs(y_test - pred).mean() / (scale + eps)

    def _smape(pred):
        denom = (np.abs(y_test) + np.abs(pred)) / 2 + eps
        return np.mean(np.abs(y_test - pred) / denom)

    mase_model  = _mase(y_pred)
    mase_naive2 = _mase(y_naive2)
    smape_model  = _smape(y_pred)
    smape_naive2 = _smape(y_naive2)

    return float(0.5 * (mase_model / (mase_naive2 + eps)) +
                 0.5 * (smape_model / (smape_naive2 + eps)))
```

---

## 6. Relative RMSE and Relative MAE

### 6.1 Formulas

```
rMAE  = MAE_model / MAE_baseline

rRMSE = RMSE_model / RMSE_baseline

Interpretation:
  rMAE = 0.75  → model MAE is 75% of baseline MAE (25% improvement)
  rMAE = 1.10  → model is 10% WORSE than baseline
```

### 6.2 Relationship to MASE

When the baseline is the in-sample naïve forecast and seasonality=1:

```
rMAE ≈ MASE    (they use the same baseline structure)
```

The difference is subtle — MASE uses in-sample naïve errors (T-1 terms), while rMAE uses out-of-sample baseline errors. MASE is preferred because it uses the training set for the denominator (reproducible), while rMAE computes the baseline on the test set.

---

## 7. Multi-Series Aggregation

### 7.1 The Problem

When evaluating a model across hundreds or thousands of series, how should per-series metrics be aggregated?

```
Option 1: Unweighted mean
  Mean MASE = average MASE across all series
  → Each series contributes equally regardless of volume

Option 2: Volume-weighted mean
  Weighted MASE = Σᵢ (volumeᵢ × MASEᵢ) / Σᵢ volumeᵢ
  → High-volume series dominate the overall score

Option 3: Pooled metric (wMAPE-style)
  Pooled MAE = Σᵢ MAEᵢ × nᵢ / Σᵢ nᵢ
  → Each test period contributes equally regardless of which series
```

### 7.2 Implementation

```python
import pandas as pd
import numpy as np

def aggregate_metrics_across_series(
    results_df: pd.DataFrame,
    metric_col: str = "mase",
    volume_col: str = "total_volume",
    method: str = "unweighted",
) -> dict:
    """
    Aggregate per-series metrics into a single summary figure.

    Parameters
    ----------
    results_df : DataFrame with one row per series, columns = metrics + volume
    metric_col : metric column to aggregate
    volume_col : column for volume-weighted aggregation
    method     : 'unweighted', 'volume_weighted', or 'median'

    Returns
    -------
    dict with mean, median, std, and percentile breakdown
    """
    scores = results_df[metric_col].dropna().values

    stats = {
        "mean":   float(scores.mean()),
        "median": float(np.median(scores)),
        "std":    float(scores.std()),
        "p25":    float(np.percentile(scores, 25)),
        "p75":    float(np.percentile(scores, 75)),
        "p90":    float(np.percentile(scores, 90)),
        "n_series": len(scores),
        "n_better_than_naive": int((scores < 1.0).sum()) if metric_col == "mase" else None,
    }

    if method == "volume_weighted" and volume_col in results_df.columns:
        volumes = results_df[volume_col].values
        stats["volume_weighted_mean"] = float(np.average(scores, weights=volumes))
    elif method == "median":
        stats["aggregate"] = stats["median"]
    else:
        stats["aggregate"] = stats["mean"]

    return stats
```

---

## 8. Implementation

```python
import numpy as np
import pandas as pd


def full_skill_report(
    y_train: np.ndarray,
    y_test: np.ndarray,
    forecasts: dict,
    seasonality: int = 1,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Generate a comprehensive skill report comparing multiple models to naïve.

    Parameters
    ----------
    y_train    : training series
    y_test     : test actuals
    forecasts  : dict of {model_name: prediction_array}
    seasonality: seasonal period for MASE and naïve baseline
    eps        : denominator epsilon for MAPE/SMAPE

    Returns
    -------
    DataFrame with rows = models, columns = metrics
    """
    # Generate naïve baseline
    if seasonality == 1:
        y_naive = np.full(len(y_test), y_train[-1])
        scale   = np.abs(np.diff(y_train)).mean()
    else:
        tail    = y_train[-seasonality:]
        reps    = (len(y_test) // seasonality) + 1
        y_naive = np.tile(tail, reps)[:len(y_test)]
        scale   = np.abs(y_train[seasonality:] - y_train[:-seasonality]).mean()

    forecasts = {"Naïve": y_naive, **forecasts}

    rows = []
    for name, preds in forecasts.items():
        preds = np.asarray(preds, dtype=float)
        err   = y_test - preds
        abs_e = np.abs(err)

        denom_mape  = np.abs(y_test) + eps
        denom_smape = (np.abs(y_test) + np.abs(preds)) / 2 + eps

        row = {
            "Model":      name,
            "MAE":        round(abs_e.mean(), 4),
            "RMSE":       round(np.sqrt((err**2).mean()), 4),
            "MAPE (%)":   round(100 * (abs_e / denom_mape).mean(), 2),
            "SMAPE (%)":  round(100 * (abs_e / denom_smape).mean(), 2),
            "MASE":       round(abs_e.mean() / (scale + eps), 4),
            "wMAPE (%)":  round(100 * abs_e.sum() / (np.abs(y_test).sum() + eps), 2),
            "Skill (MAE)": round(1 - abs_e.mean() / (np.abs(y_test - y_naive).mean() + eps), 4)
                           if name != "Naïve" else 0.0,
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    return df


# ─── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    N  = 300
    t  = np.arange(N)
    y  = 100 + 0.1 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 3, N)

    SPLIT     = 250
    y_train   = y[:SPLIT]
    y_test    = y[SPLIT:]

    preds_good   = y_test + np.random.normal(0, 1.5, len(y_test))
    preds_medium = y_test + np.random.normal(0, 4.0, len(y_test))
    preds_bad    = y_test + np.random.normal(5, 6.0, len(y_test))

    report = full_skill_report(
        y_train, y_test,
        forecasts={"Good Model": preds_good,
                   "Medium Model": preds_medium,
                   "Bad Model": preds_bad},
        seasonality=12,
    )
    print(report.to_string())
    print(f"\nModels with MASE < 1.0 (better than naïve):")
    print(report[report["MASE"] < 1.0].index.tolist())
```

---

*← [01 — Error Metrics](./01_error_metrics_MAE_RMSE_MAPE.md) | [Module README](./README.md) | Next: [03 — Backtesting Design](./03_backtesting_design.md) →*
