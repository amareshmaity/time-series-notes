# 02 — Time Series Cross-Validation

> **Module**: 04 ML for Time Series | **File**: 2 of 6
>
> Standard K-Fold CV is wrong for time series — it leaks future data into training. Time series requires forward-only validation strategies that respect temporal ordering. This is the most important concept in ML for time series.

---

## Table of Contents

1. [Why Standard K-Fold is Wrong](#1-why-standard-k-fold-is-wrong)
2. [Walk-Forward Validation](#2-walk-forward-validation)
3. [Expanding Window CV](#3-expanding-window-cv)
4. [Sliding Window CV](#4-sliding-window-cv)
5. [sklearn TimeSeriesSplit](#5-sklearn-timeseriessplit)
6. [Purged Cross-Validation](#6-purged-cross-validation)
7. [Choosing CV Strategy](#7-choosing-cv-strategy)
8. [Hyperparameter Tuning with Time CV](#8-hyperparameter-tuning-with-time-cv)

---

## 1. Why Standard K-Fold is Wrong

### 1.1 The Problem

Standard K-Fold randomly shuffles data into `k` folds. This means the validation fold can contain data from **before the training fold** — the model "sees the future" during training.

```
K-Fold (WRONG for time series):

Data: [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec]

Fold 1: Train=[Feb,Apr,Jun,Aug,Oct,Dec]  Val=[Jan,Mar,May,Jul,Sep,Nov]
         → Model trained on June is validated on March ← LEAKAGE!

Fold 2: Train=[Jan,Mar,May,Jul,Sep,Nov]  Val=[Feb,Apr,Jun,Aug,Oct,Dec]
         → Model trained on November is validated on February ← LEAKAGE!
```

### 1.2 Why This Inflates Performance

With K-Fold on time series:
- Lag features computed from shuffled data look into the future
- Autocorrelation creates artificially high correlations between "train" and "val"
- Reported CV scores are overoptimistic by 10–30% in practice

### 1.3 The Golden Rule

> **Time must always move forward.** The validation set must always be strictly later in time than the training set.

---

## 2. Walk-Forward Validation

### 2.1 Concept

At each **origin** (rolling evaluation point):
1. Train on all data up to origin `t`
2. Forecast `h` steps ahead
3. Compare forecast to actual values at `t+1` to `t+h`
4. Advance the origin by one step (or one window)

```
Walk-Forward (1-step ahead, rolling origin):

Origin 1: Train=[t1..t8]   Forecast=[t9]    Actual=[t9]
Origin 2: Train=[t1..t9]   Forecast=[t10]   Actual=[t10]
Origin 3: Train=[t1..t10]  Forecast=[t11]   Actual=[t11]
...

Key: validation data is ALWAYS in the future relative to training data
```

### 2.2 Implementation

```python
import numpy as np
import pandas as pd
from typing import Iterator

def walk_forward_splits(
    series: pd.Series,
    n_splits: int = 5,
    horizon: int = 1,
    gap: int = 0,
) -> Iterator[tuple[pd.Series, pd.Series]]:
    """
    Generate walk-forward train/test splits.

    Parameters:
        series   : full time series
        n_splits : number of validation origins
        horizon  : forecast horizon per origin
        gap      : gap between train end and test start (to avoid leakage in some datasets)

    Yields:
        (train, test) tuples
    """
    n = len(series)
    test_size = n_splits * horizon
    initial_train_size = n - test_size

    for i in range(n_splits):
        train_end = initial_train_size + i * horizon
        test_start = train_end + gap
        test_end   = test_start + horizon

        if test_end > n:
            break

        train = series.iloc[:train_end]
        test  = series.iloc[test_start:test_end]
        yield train, test


# Usage
series = pd.Series(...)   # your time series
scores = []

for fold, (train, test) in enumerate(walk_forward_splits(series, n_splits=5, horizon=12)):
    # Build features
    X_train, y_train = build_features(train)
    X_test,  y_test  = build_features_for_test(train, test)

    # Fit model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(((y_test.values - y_pred) ** 2).mean())
    scores.append(rmse)
    print(f"Fold {fold+1}: RMSE={rmse:.4f} | Train={len(train)} | Test={len(test)}")

print(f"\nMean RMSE: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

---

## 3. Expanding Window CV

### 3.1 Concept

The training window **grows** with each split — all historical data is always included:

```
Expanding Window:

Fold 1: Train=[t1..t6]       Test=[t7..t9]
Fold 2: Train=[t1..t9]       Test=[t10..t12]
Fold 3: Train=[t1..t12]      Test=[t13..t15]
Fold 4: Train=[t1..t15]      Test=[t16..t18]

→ Training set grows with each fold
→ Later folds have more training data → potentially better performance
→ Closer to production conditions (where more data accumulates over time)
```

### 3.2 Implementation

```python
def expanding_window_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    horizon: int = 12,
    initial_size: int = None,
) -> list[dict]:
    """Expanding window CV — returns per-fold metrics."""
    n = len(X)
    if initial_size is None:
        initial_size = n - n_splits * horizon

    results = []
    for fold in range(n_splits):
        train_end  = initial_size + fold * horizon
        test_start = train_end
        test_end   = test_start + horizon

        if test_end > n:
            break

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test  = X.iloc[test_start:test_end]
        y_test  = y.iloc[test_start:test_end]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        errors = y_test.values - y_pred
        results.append({
            "fold":        fold + 1,
            "train_size":  train_end,
            "train_end":   X_train.index[-1],
            "test_start":  X_test.index[0],
            "MAE":         np.abs(errors).mean(),
            "RMSE":        np.sqrt((errors**2).mean()),
        })

    return results
```

---

## 4. Sliding Window CV

### 4.1 Concept

The training window has a **fixed size** and slides forward:

```
Sliding Window (window=6):

Fold 1: Train=[t1..t6]   Test=[t7..t9]
Fold 2: Train=[t3..t9]   Test=[t10..t12]
Fold 3: Train=[t6..t12]  Test=[t13..t15]

→ Training set is always the same size
→ Old data is discarded (good if series has structural breaks)
→ May underfit if series has long-range dependencies
```

### 4.2 When Sliding Window is Better

| Situation | Prefer Expanding | Prefer Sliding |
|-----------|----------------|----------------|
| Series is stable (no regime change) | ✅ | |
| Series has structural breaks | | ✅ |
| Old data is irrelevant or misleading | | ✅ |
| Long-range seasonality matters | ✅ | |
| Short-term patterns dominate | | ✅ |

---

## 5. sklearn TimeSeriesSplit

### 5.1 Basic Usage

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(
    n_splits=5,
    gap=0,            # gap between train end and test start
    max_train_size=None,   # None = expanding; int = sliding
    test_size=None,        # None = auto; int = fixed test window
)

# Visualize splits
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(13, 4))
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    ax.scatter(train_idx, [fold] * len(train_idx), c="#2C7BB6",
               marker="_", linewidth=3, alpha=0.7, label="Train" if fold == 0 else "")
    ax.scatter(test_idx, [fold] * len(test_idx), c="#D7191C",
               marker="_", linewidth=3, alpha=0.7, label="Test" if fold == 0 else "")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Fold")
ax.set_title("TimeSeriesSplit — Expanding Window CV")
ax.legend()
plt.tight_layout()
plt.show()
```

### 5.2 Fixed Test Window (Recommended)

```python
# Fixed test size = mimics real production forecast window
tscv = TimeSeriesSplit(
    n_splits=5,
    test_size=12,    # always forecast 12 periods ahead
    gap=0,
)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(
    model, X, y,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
)
print(f"CV RMSE: {-scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 6. Purged Cross-Validation

### 6.1 The Embargo Problem

Even with chronological splits, lag features can create **indirect leakage** between train and test:

```
If test starts at t=100 and you have lag_1 as a feature:
  test row at t=100 uses y(t=99) as lag_1
  y(t=99) is the LAST training observation → fine

But if you have a rolling_30_mean:
  test row at t=100 uses mean(y[70..99]) → includes y[99], y[98] from train boundary
  → The "test" row's feature depends heavily on recent train values
  → This is technically not leakage, but the train/test boundary rows are "contaminated"
```

### 6.2 Gap-Based Purging

Add a **gap** between training end and test start to reduce boundary contamination:

```python
tscv = TimeSeriesSplit(
    n_splits=5,
    gap=7,      # skip 7 periods between train end and test start
    test_size=30,
)
```

The gap size should equal the maximum feature lookback window that uses target values:

```python
gap = max(rolling_window_size, max_lag)  # conservative approach
```

### 6.3 Combinatorial Purged CV (Advanced)

Used in financial ML where overlapping returns create strong contamination:

```python
# pip install mlfinlab
from mlfinlab.cross_validation import CombinatorialPurgedKFold

cpkf = CombinatorialPurgedKFold(
    n_splits=5,
    n_test_splits=2,
    pct_embargo=0.01,   # 1% embargo period
)
```

---

## 7. Choosing CV Strategy

```
Is your series < 200 observations?
├── YES → Expanding window (maximize training data)
└── NO ↓

Has the series experienced a structural break (policy change, COVID, etc.)?
├── YES → Sliding window (discard irrelevant old data)
└── NO → Expanding window (use all history)

Do you have overlapping lag features from target variable?
├── YES → Add a gap equal to max lag window
└── NO → No gap needed

Do you need to evaluate multiple forecast horizons?
├── YES → Use multi-horizon walk-forward (horizon=1,3,6,12)
└── NO → Single-horizon walk-forward
```

---

## 8. Hyperparameter Tuning with Time CV

The key principle: **the objective function for hyperparameter search must use time-safe CV**:

```python
import optuna
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import numpy as np

tscv = TimeSeriesSplit(n_splits=5, test_size=12)

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "verbose": -1,
    }

    fold_rmses = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        y_pred = model.predict(X_val)
        rmse = np.sqrt(((y_val.values - y_pred) ** 2).mean())
        fold_rmses.append(rmse)

    return np.mean(fold_rmses)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, timeout=300)
print(f"Best CV RMSE: {study.best_value:.4f}")
print(f"Best params:  {study.best_params}")
```

---

*← [01 — ML vs Statistical](./01_ml_vs_statistical_models.md) | [Module README](./README.md) | Next: [03 — XGBoost & LightGBM](./03_gradient_boosting_xgboost_lgbm.md) →*
