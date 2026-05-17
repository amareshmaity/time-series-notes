# 03 — Backtesting Design

> **Module**: 08 Evaluation & Metrics | **File**: 3 of 5
>
> A train/test split is not a backtest. Rigorous backtesting simulates the real deployment conditions of a forecasting model — using only information available at each forecast origin, across multiple evaluation windows. This note covers every design principle, pitfall, and implementation pattern.

---

## Table of Contents

1. [Why Simple Train/Test Splits Fail](#1-why-simple-traintest-splits-fail)
2. [Walk-Forward (Rolling Origin) Validation](#2-walk-forward-rolling-origin-validation)
3. [Expanding vs. Fixed-Width Windows](#3-expanding-vs-fixed-width-windows)
4. [Gap Strategy — Preventing Leakage](#4-gap-strategy--preventing-leakage)
5. [Blocked Cross-Validation](#5-blocked-cross-validation)
6. [Purged and Embargoed CV](#6-purged-and-embargoed-cv)
7. [Backtesting Design Checklist](#7-backtesting-design-checklist)
8. [Implementation](#8-implementation)

---

## 1. Why Simple Train/Test Splits Fail

### 1.1 The Single Split Problem

```
Data: 1000 observations
Train: first 800
Test: last 200

Problems:
  1. Only ONE evaluation point → results are highly variable
     (test period could be unusually easy or hard)

  2. Test performance depends heavily on WHERE you split
     → Seasonal effects, trend shifts, or anomalies in test period
       can make a good model look bad or vice versa

  3. Production gap ignored — in reality, models are retrained
     periodically (weekly, monthly), not just once

Standard minimum: 5–10 rolling evaluation windows
```

### 1.2 Data Leakage in Time Series Evaluation

```
Forbidden patterns:

  ❌ Using future data in features:
     Rolling mean computed over future observations

  ❌ Using test labels for hyperparameter search:
     Grid search that includes test set in parameter selection

  ❌ Fitting scalers/encoders on test data:
     StandardScaler.fit(train + test) then transform train

  ❌ Cross-validation with shuffled folds (k-fold):
     Fold 3 (training) contains data AFTER fold 1 (validation)
     → Future info leaks into past predictions

Time series CV must ALWAYS respect temporal ordering.
```

---

## 2. Walk-Forward (Rolling Origin) Validation

### 2.1 Concept

At each **evaluation origin** `t`, the model is trained on all data up to `t` and tested on the next `H` steps. The origin then advances by a step size `s`:

```
Origins: t₁, t₂, t₃, ..., tₙ   (equally spaced by step s)

Origin t₁:  ──────train─────| test_1 |
Origin t₂:  ──────────train────| test_2 |
Origin t₃:  ────────────────train──| test_3 |
...

Final metric = average across all test windows
```

### 2.2 Key Design Parameters

| Parameter      | Description                              | Typical Value              |
|----------------|------------------------------------------|----------------------------|
| `min_train`    | Minimum training window before evaluation | 1–2× seasonal period        |
| `H`            | Forecast horizon (test window size)       | Business requirement        |
| `step`         | Spacing between origins                   | H (non-overlapping) or 1   |
| `n_origins`    | Number of evaluation windows              | ≥ 5, ideally 10–20          |
| `refit`        | Retrain model at each origin?             | Depends on cost/stationarity|

### 2.3 Overlapping vs. Non-Overlapping Windows

```
Non-overlapping (step = H):
  Origin 1: train[1..100], test[101..112]
  Origin 2: train[1..112], test[113..124]
  → Test windows are independent — clean evaluation
  → Fewer total windows for the same data size

Overlapping (step = 1):
  Origin 1: train[1..100], test[101..112]
  Origin 2: train[1..101], test[102..113]
  → More windows → more stable metric estimates
  → Test windows overlap → errors are correlated → adjust p-values
```

---

## 3. Expanding vs. Fixed-Width Windows

### 3.1 Expanding (Cumulative) Window

Training set grows with each origin. All historical data used at every step:

```
Origin 1: |──────── 200 obs ────────|
Origin 2: |─────────── 212 obs ─────────|
Origin 3: |──────────────── 224 obs ──────────────|

✅ Uses all available history → more data → better models
✅ Reflects real production (models retrained on all data)
❌ Early origins have less training data → harder forecast problem
❌ Computational cost grows with each origin
```

### 3.2 Fixed (Rolling/Sliding) Window

Training set has a fixed size and moves forward:

```
Origin 1: |── 200 obs ──|
Origin 2:       |── 200 obs ──|
Origin 3:              |── 200 obs ──|

✅ Each origin has identical training conditions → comparable results
✅ Faster computation (constant training size)
✅ Better for non-stationary data (older data may harm performance)
❌ Discards potentially useful older data
❌ Does not reflect production retraining (if model uses all history)
```

### 3.3 Choosing Between Them

```
Use Expanding Window when:
  - Data is approximately stationary
  - Older data is still relevant (stable patterns)
  - Reflecting real production conditions is priority

Use Fixed Window when:
  - Data is non-stationary or has structural breaks
  - You suspect older data hurts (distribution shift)
  - Series is very long (expanding becomes slow)
  - You need comparable evaluation conditions across origins
```

---

## 4. Gap Strategy — Preventing Leakage

### 4.1 The Production Gap

In production, there is always a lag between when data is collected and when forecasts are needed:

```
Monday:   Data for last week arrives (with some delay)
Tuesday:  Model is retrained
Wednesday: Forecasts delivered for next week

→ Minimum 1–2 day gap between last training observation and first forecast step

Ignoring this gap in backtesting → optimistic results that won't reproduce in production
```

### 4.2 Gap Implementation

```python
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BacktestOrigin:
    train_end:  int   # last index in training set
    test_start: int   # first index in test set (train_end + gap + 1)
    test_end:   int   # last index in test set


def generate_backtest_origins(
    n: int,
    min_train: int,
    horizon: int,
    n_origins: int,
    step: Optional[int] = None,
    gap: int = 0,
    window_type: str = "expanding",
    fixed_window_size: Optional[int] = None,
) -> list[BacktestOrigin]:
    """
    Generate backtest origins with optional gap between train and test.

    Parameters
    ----------
    n                 : total number of observations
    min_train         : minimum training observations before first evaluation
    horizon           : forecast horizon H (test window length)
    n_origins         : number of evaluation origins
    step              : spacing between origins (default = horizon, non-overlapping)
    gap               : number of observations between train end and test start
                        (simulates production data lag)
    window_type       : 'expanding' or 'fixed'
    fixed_window_size : training window size for fixed window (required if fixed)

    Returns
    -------
    list of BacktestOrigin objects
    """
    if step is None:
        step = horizon

    # Determine the range of valid train_end indices
    # test must end at most at index n-1
    max_train_end = n - 1 - gap - horizon
    min_train_end = min_train - 1

    if min_train_end >= max_train_end:
        raise ValueError(f"Not enough data: need at least {min_train + gap + horizon} observations")

    train_ends = np.linspace(min_train_end, max_train_end, n_origins, dtype=int)

    origins = []
    for train_end in train_ends:
        test_start = train_end + gap + 1
        test_end   = test_start + horizon - 1

        if test_end >= n:
            break

        if window_type == "fixed":
            assert fixed_window_size is not None
            train_start = max(0, train_end - fixed_window_size + 1)
        else:
            train_start = 0

        origins.append(BacktestOrigin(
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        ))

    return origins
```

---

## 5. Blocked Cross-Validation

### 5.1 When to Use

Blocked CV is a compromise between k-fold CV (fast hyperparameter search) and walk-forward validation (correct evaluation). It splits data into **contiguous blocks** and uses them as folds without shuffling:

```
Blocked CV with 5 folds:

Fold 1: train=[2,3,4,5]  validate=[1]   ← temporal order WRONG
         ↑ Never do this — future data in training!

Correct: Only train on PAST folds, validate on NEXT fold

Fold 1: train=[1]         validate=[2]
Fold 2: train=[1,2]       validate=[3]
Fold 3: train=[1,2,3]     validate=[4]
Fold 4: train=[1,2,3,4]   validate=[5]
```

### 5.2 Implementation with sklearn

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def demonstrate_ts_split():
    """
    TimeSeriesSplit from sklearn — correct implementation of expanding-window CV.

    Key parameter: test_size fixes the validation window to exactly H steps.
    gap parameter adds a gap between train and test.
    """
    n       = 300
    horizon = 24

    tscv = TimeSeriesSplit(
        n_splits=5,
        test_size=horizon,   # each test fold = H steps
        gap=0,               # set gap > 0 for production lag simulation
    )

    X = np.arange(n).reshape(-1, 1)
    y = np.random.randn(n)

    print("TimeSeriesSplit — fold sizes:")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"  Fold {fold+1}: train[{train_idx[0]}..{train_idx[-1]}] "
              f"({len(train_idx)} obs) | "
              f"test[{test_idx[0]}..{test_idx[-1]}] ({len(test_idx)} obs)")
```

---

## 6. Purged and Embargoed CV

### 6.1 Motivation

When features contain **lagged values**, training rows near the validation boundary can "see" the future through their feature construction:

```
Example: model uses lag_7 (7-day lag feature)

Training row at t=100:
  feature lag_7 = y[93]
  ← if y[93] is in the validation period, information from future leaks into training

Solution: PURGE rows near the train/validation boundary
EMBARGO: skip an additional buffer after the validation period
```

### 6.2 Implementation

```python
import numpy as np

def purged_cv_indices(
    n: int,
    n_splits: int,
    horizon: int,
    purge_size: int,
    embargo_size: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Purged walk-forward CV for time series with lag features.

    Parameters
    ----------
    n           : total number of observations
    n_splits    : number of CV folds
    horizon     : validation window size
    purge_size  : number of training rows to remove near the boundary
                  (set to max_lag used in feature engineering)
    embargo_size: additional rows to skip after validation (optional)

    Returns
    -------
    list of (train_indices, val_indices) tuples
    """
    fold_size = n // (n_splits + 1)
    splits    = []

    for fold in range(n_splits):
        val_start = (fold + 1) * fold_size
        val_end   = val_start + horizon
        if val_end > n:
            break

        # Training: all rows before val_start, minus purge window
        train_end_purged = val_start - purge_size
        train_idx        = np.arange(0, max(0, train_end_purged))

        # Validation
        val_idx = np.arange(val_start, val_end)

        # Optional embargo: skip rows after validation before next fold
        # (handled automatically by the fold spacing)

        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))

    return splits
```

---

## 7. Backtesting Design Checklist

```
Before running any backtest, verify:

  ✅ Temporal ordering preserved — no shuffling
  ✅ Features use only past data (shift(1) before rolling windows)
  ✅ Scalers/encoders fitted on training fold only
  ✅ At least 5 evaluation origins (10+ recommended)
  ✅ Horizon matches production forecast requirement
  ✅ Gap reflects real data latency (if applicable)
  ✅ Purge window ≥ max lag used in features (if lag features present)
  ✅ Hyperparameter search uses only training folds (no test leakage)
  ✅ Model selected based on CV metric, then evaluated once on final holdout
  ✅ Multiple metrics reported (MAE, MASE, RMSE) — not just one
  ✅ Baseline (naïve) computed and compared at every origin
```

---

## 8. Implementation

### 8.1 Production Backtesting Framework

```python
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class BacktestResult:
    """Results from a single backtesting run."""
    metric_per_origin: np.ndarray   # (n_origins,) overall metric
    mae_per_step:      np.ndarray   # (H,) MAE per horizon step
    rmse_per_step:     np.ndarray   # (H,) RMSE per horizon step
    mase_per_origin:   np.ndarray   # (n_origins,) MASE
    predictions:       np.ndarray   # (n_origins, H) all predictions
    actuals:           np.ndarray   # (n_origins, H) all actuals
    origins:           list         # list of BacktestOrigin objects

    @property
    def mean_mae(self) -> float:
        return float(np.mean(np.abs(self.actuals - self.predictions)))

    @property
    def mean_mase(self) -> float:
        return float(self.mase_per_origin.mean())

    def summary(self) -> pd.Series:
        mae  = np.abs(self.actuals - self.predictions)
        rmse = np.sqrt((self.actuals - self.predictions) ** 2)
        return pd.Series({
            "Mean MAE":       mae.mean(),
            "Median MAE":     np.median(mae),
            "Mean RMSE":      rmse.mean(),
            "Mean MASE":      self.mase_per_origin.mean(),
            "MAE Std":        np.abs(self.actuals - self.predictions).mean(axis=1).std(),
            "N Origins":      len(self.origins),
        }).round(4)


def run_backtest(
    model_fn: Callable[[np.ndarray], np.ndarray],
    series: np.ndarray,
    min_train: int,
    horizon: int,
    n_origins: int = 10,
    step: Optional[int] = None,
    gap: int = 0,
    window_type: str = "expanding",
    fixed_window_size: Optional[int] = None,
    seasonality: int = 1,
) -> BacktestResult:
    """
    Run walk-forward backtesting for a forecasting model.

    Parameters
    ----------
    model_fn    : callable that takes training series (1D array) and returns
                  forecast array of length `horizon`
    series      : full time series (1D array)
    min_train   : minimum training length before first evaluation
    horizon     : forecast horizon H
    n_origins   : number of evaluation origins
    step        : spacing between origins (default = horizon)
    gap         : gap between train end and test start
    window_type : 'expanding' or 'fixed'
    fixed_window_size : training window size (required if fixed)
    seasonality : for MASE computation

    Returns
    -------
    BacktestResult with all predictions, actuals, and metrics
    """
    if step is None:
        step = horizon

    origins = generate_backtest_origins(
        n=len(series),
        min_train=min_train,
        horizon=horizon,
        n_origins=n_origins,
        step=step,
        gap=gap,
        window_type=window_type,
        fixed_window_size=fixed_window_size,
    )

    if not origins:
        raise ValueError("No valid backtest origins — check min_train, horizon, and data length.")

    all_preds  = []
    all_actuals = []
    all_mase   = []

    for orig in origins:
        if window_type == "expanding":
            train = series[:orig.train_end + 1]
        else:
            start = max(0, orig.train_end + 1 - (fixed_window_size or orig.train_end + 1))
            train = series[start:orig.train_end + 1]

        test    = series[orig.test_start:orig.test_end + 1]
        preds   = model_fn(train)[:horizon]

        # MASE: use training series for scale
        if seasonality == 1:
            scale = np.abs(np.diff(train)).mean() + 1e-12
        else:
            scale = np.abs(train[seasonality:] - train[:-seasonality]).mean() + 1e-12

        h = min(len(preds), len(test))
        all_preds.append(preds[:h])
        all_actuals.append(test[:h])
        all_mase.append(np.abs(test[:h] - preds[:h]).mean() / scale)

    preds_mat   = np.array(all_preds)
    actuals_mat = np.array(all_actuals)

    return BacktestResult(
        metric_per_origin=np.abs(actuals_mat - preds_mat).mean(axis=1),
        mae_per_step     =np.abs(actuals_mat - preds_mat).mean(axis=0),
        rmse_per_step    =np.sqrt((actuals_mat - preds_mat)**2).mean(axis=0),
        mase_per_origin  =np.array(all_mase),
        predictions      =preds_mat,
        actuals          =actuals_mat,
        origins          =origins,
    )


# ─── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    N      = 500
    t      = np.arange(N)
    series = 100 + 0.05*t + 8*np.sin(2*np.pi*t/12) + np.random.normal(0, 2, N)

    # Model function: simple exponential smoothing forecast
    def ses_model(train, alpha=0.3, h=12):
        level = train[0]
        for y in train:
            level = alpha * y + (1 - alpha) * level
        return np.full(h, level)

    model_h12 = lambda train: ses_model(train, h=12)

    result = run_backtest(
        model_fn=model_h12,
        series=series,
        min_train=60,
        horizon=12,
        n_origins=15,
        gap=0,
        seasonality=12,
    )

    print("Backtest Summary:")
    print(result.summary())
    print(f"\nMAE per horizon step:")
    for h, m in enumerate(result.mae_per_step, 1):
        print(f"  h={h:02d}: MAE={m:.3f}")
```

---

*← [02 — Skill Scores](./02_skill_scores_and_relative_metrics.md) | [Module README](./README.md) | Next: [04 — Model Comparison & Statistical Tests](./04_model_comparison_and_statistical_tests.md) →*
