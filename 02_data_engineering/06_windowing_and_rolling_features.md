# 06 — Windowing & Rolling Features

> **Module**: 02 Data Engineering | **File**: 6 of 6
>
> Windowing transforms a time series into structured feature matrices. Rolling statistics capture local behavior over time. Together they are the backbone of ML-ready time series datasets.

---

## Table of Contents

1. [Sliding Windows for Supervised Learning](#1-sliding-windows-for-supervised-learning)
2. [Rolling Statistics](#2-rolling-statistics)
3. [Expanding Window Statistics](#3-expanding-window-statistics)
4. [Exponentially Weighted Moving Average (EWM)](#4-exponentially-weighted-moving-average-ewm)
5. [Multi-Step Output Windows (MIMO)](#5-multi-step-output-windows-mimo)
6. [Windowing for Deep Learning](#6-windowing-for-deep-learning)
7. [Choosing Window Size](#7-choosing-window-size)

---

## 1. Sliding Windows for Supervised Learning

A **sliding window** (lookback window) converts a time series into a tabular dataset: each row contains the last `w` observations as features, and the next `h` observations as targets.

```
Original series: [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10]
Window size w=3, horizon h=1:

Row 1: X=[y1, y2, y3]  →  y=y4
Row 2: X=[y2, y3, y4]  →  y=y5
Row 3: X=[y3, y4, y5]  →  y=y6
...
Row 7: X=[y7, y8, y9]  →  y=y10
```

```python
import numpy as np
import pandas as pd

def create_sliding_window_dataset(
    series: np.ndarray,
    lookback: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1D time series into sliding window (X, y) pairs.

    Parameters:
        series   : 1D numpy array of time series values
        lookback : number of past time steps used as features
        horizon  : number of future time steps to predict

    Returns:
        X : shape (n_samples, lookback)
        y : shape (n_samples, horizon)
    """
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback : i + lookback + horizon])
    return np.array(X), np.array(y)

# Example
values = series.values
X, y = create_sliding_window_dataset(values, lookback=30, horizon=7)
print(f"X shape: {X.shape}   (n_samples, lookback)")
print(f"y shape: {y.shape}   (n_samples, horizon)")
```

---

## 2. Rolling Statistics

Rolling statistics summarize recent behavior in a single number — capturing trend, volatility, and range.

### 2.1 Rolling Mean (Moving Average)

```python
# Simple rolling mean — always shift(1) before rolling to avoid leakage
df["roll7_mean"]  = df["sales"].shift(1).rolling(window=7).mean()
df["roll14_mean"] = df["sales"].shift(1).rolling(window=14).mean()
df["roll30_mean"] = df["sales"].shift(1).rolling(window=30).mean()

# min_periods: compute even if window is not full (for start of series)
df["roll7_mean_minp"] = df["sales"].shift(1).rolling(7, min_periods=1).mean()
```

### 2.2 Rolling Standard Deviation (Volatility)

```python
df["roll7_std"]  = df["sales"].shift(1).rolling(7).std()
df["roll30_std"] = df["sales"].shift(1).rolling(30).std()
```

### 2.3 Rolling Min, Max, Range

```python
df["roll7_min"]   = df["sales"].shift(1).rolling(7).min()
df["roll7_max"]   = df["sales"].shift(1).rolling(7).max()
df["roll7_range"] = df["roll7_max"] - df["roll7_min"]
```

### 2.4 Rolling Quantiles

```python
df["roll30_q25"] = df["sales"].shift(1).rolling(30).quantile(0.25)
df["roll30_q75"] = df["sales"].shift(1).rolling(30).quantile(0.75)
df["roll30_iqr"] = df["roll30_q75"] - df["roll30_q25"]
```

### 2.5 Rolling Correlation (Multivariate)

```python
# Rolling correlation between sales and temperature
df["roll30_corr_temp"] = (
    df["sales"].shift(1)
    .rolling(30)
    .corr(df["temperature"].shift(1))
)
```

### 2.6 Multiple Windows at Once

```python
def add_rolling_features(
    df: pd.DataFrame,
    col: str,
    windows: list[int],
    funcs: list[str] = None,
) -> pd.DataFrame:
    """Add rolling statistics for multiple windows."""
    df = df.copy()
    funcs = funcs or ["mean", "std", "min", "max"]
    shifted = df[col].shift(1)

    for w in windows:
        rolled = shifted.rolling(w, min_periods=1)
        for fn in funcs:
            df[f"roll{w}_{fn}"] = getattr(rolled, fn)()

    return df

df = add_rolling_features(df, col="sales", windows=[7, 14, 30, 90], funcs=["mean", "std"])
```

---

## 3. Expanding Window Statistics

An **expanding window** grows from the start of the series — always uses ALL past data up to time `t`.

```python
# Cumulative statistics (expanding window)
df["expanding_mean"] = df["sales"].shift(1).expanding().mean()
df["expanding_std"]  = df["sales"].shift(1).expanding().std()
df["expanding_max"]  = df["sales"].shift(1).expanding().max()
df["expanding_min"]  = df["sales"].shift(1).expanding().min()
```

**When to use expanding windows:**
- Capturing long-run trends without choosing a window size
- Cumulative revenue, cumulative units sold
- Historical maximum / minimum (useful for normalization context)

**Rolling vs. Expanding:**

| | Rolling | Expanding |
|--|---------|-----------|
| Window size | Fixed | Grows with time |
| Adapts to recent data | ✅ Yes | ❌ No — affected by early history |
| Captures long-run history | ❌ Limited | ✅ Yes |
| Useful for | Local behavior, volatility | Cumulative context |

---

## 4. Exponentially Weighted Moving Average (EWM)

EWM gives **exponentially decreasing weights** to older observations — recent values matter more.

```
EWM(t) = α · y(t) + (1-α) · EWM(t-1)

Where:
  α = smoothing factor (0 < α < 1)
  α close to 1 → recent values dominate (fast decay of history)
  α close to 0 → history matters more (slow decay, smoother)
```

```python
# EWM mean — span parameter: span ≈ 2/(α+1)
df["ewm7_mean"]  = df["sales"].shift(1).ewm(span=7,  adjust=False).mean()
df["ewm30_mean"] = df["sales"].shift(1).ewm(span=30, adjust=False).mean()

# EWM std
df["ewm7_std"]   = df["sales"].shift(1).ewm(span=7).std()

# Using half-life instead of span
# half_life: number of periods for weight to decay to 50%
df["ewm_hl7_mean"] = df["sales"].shift(1).ewm(halflife=7).mean()
```

**EWM vs. Rolling mean:**

| | Simple Moving Average | EWM |
|--|----------------------|-----|
| Weights | Equal | Exponentially decaying |
| Adapts to change | Slow (lagged) | Faster (recent points upweighted) |
| Jumpiness | Sensitive to old values leaving the window | Smooth |
| Parameters | `window` | `span` or `halflife` or `alpha` |

---

## 5. Multi-Step Output Windows (MIMO)

For **multi-step forecasting**, create multiple output targets simultaneously:

```python
def create_mimo_dataset(
    series: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    MIMO: Multiple-Input Multiple-Output sliding window.
    X: lookback steps → y: horizon steps
    """
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback : i + lookback + horizon])
    return np.array(X), np.array(y)

X, y = create_mimo_dataset(values, lookback=60, horizon=14)
print(f"X shape: {X.shape}   →  y shape: {y.shape}")
# X: (n_samples, 60)   y: (n_samples, 14)
```

---

## 6. Windowing for Deep Learning

Deep learning models (LSTM, TCN, TFT) consume 3D tensors:

```
Shape: (batch_size, sequence_length, n_features)
```

```python
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for sliding window time series.
    Supports univariate and multivariate inputs.
    """

    def __init__(
        self,
        features: np.ndarray,   # shape: (T, n_features)
        targets: np.ndarray,    # shape: (T,) or (T, n_outputs)
        lookback: int,
        horizon: int = 1,
    ):
        self.features = features
        self.targets  = targets
        self.lookback = lookback
        self.horizon  = horizon

    def __len__(self) -> int:
        return len(self.features) - self.lookback - self.horizon + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.lookback]
        y = self.targets[idx + self.lookback : idx + self.lookback + self.horizon]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# Build dataset and dataloader
feature_matrix = df[feature_cols].values   # (T, n_features)
target_array   = df["sales"].values         # (T,)

dataset = TimeSeriesDataset(feature_matrix, target_array, lookback=30, horizon=7)
loader  = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=True)

# Verify shapes
x_batch, y_batch = next(iter(loader))
print(f"X batch shape: {x_batch.shape}")   # (64, 30, n_features)
print(f"y batch shape: {y_batch.shape}")   # (64, 7)
```

---

## 7. Choosing Window Size

The lookback window size `w` is a critical hyperparameter:

### 7.1 Rules of Thumb

| Data Frequency | Recommended Lookback | Reasoning |
|---------------|---------------------|-----------|
| Daily | 30–90 days | Captures monthly + some seasonal pattern |
| Hourly | 24–168 hours | Captures 1 day to 1 week of context |
| Weekly | 52–104 weeks | Captures 1–2 full years |
| Monthly | 24–36 months | Captures 2–3 seasonal cycles |

### 7.2 Data-Driven Window Selection

```python
from statsmodels.tsa.stattools import acf

def optimal_lookback(series: pd.Series, max_lags: int = 100) -> int:
    """
    Estimate optimal lookback window using ACF:
    Use the lag at which ACF drops below the significance threshold.
    """
    acf_vals = acf(series.dropna(), nlags=max_lags, fft=True)
    conf_bound = 1.96 / np.sqrt(len(series))
    # Find the last lag with significant autocorrelation
    significant = np.where(np.abs(acf_vals) > conf_bound)[0]
    if len(significant) > 1:
        return int(significant[-1])
    return 14   # fallback default

window = optimal_lookback(series)
print(f"Recommended lookback window: {window} steps")
```

### 7.3 Hyperparameter Tuning

Always validate window size using time series cross-validation (not random search):

```python
import optuna
from sklearn.metrics import mean_absolute_error

def objective(trial):
    lookback = trial.suggest_int("lookback", 7, 90)
    # ... build features, train model, evaluate on hold-out
    return mae_score

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)
print(f"Best lookback: {study.best_params['lookback']}")
```

---

*← [05 — Feature Engineering](./05_feature_engineering_for_ts.md) | [Module README](./README.md) | Next Module: [03 — Statistical Models](../03_statistical_models/README.md) →*
