# 05 — Feature Engineering for Time Series

> **Module**: 02 Data Engineering | **File**: 5 of 6
>
> Feature engineering is the most impactful step in time series ML. Tree-based models like XGBoost and LightGBM have no inherent temporal awareness — it all comes from the features you create.

---

## Table of Contents

1. [Why Feature Engineering Matters](#1-why-feature-engineering-matters)
2. [Lag Features](#2-lag-features)
3. [Calendar Features](#3-calendar-features)
4. [Fourier Terms for Seasonality](#4-fourier-terms-for-seasonality)
5. [Target Encoding](#5-target-encoding)
6. [Interaction Features](#6-interaction-features)
7. [Normalization and Scaling](#7-normalization-and-scaling)
8. [Automated Feature Extraction (tsfresh)](#8-automated-feature-extraction-tsfresh)
9. [The Leakage Rules](#9-the-leakage-rules)
10. [Reusable Feature Pipeline](#10-reusable-feature-pipeline)

---

## 1. Why Feature Engineering Matters

Tree-based models (XGBoost, LightGBM, Random Forest) are **inherently non-temporal** — they treat every row independently. To make them time-aware, you must embed temporal context into features.

```
Without features:  X = []       → Model has no knowledge of the past
With lag features: X = [y(t-1), y(t-7), rolling_mean_7, month_sin, ...]
                   → Model learns: "yesterday's value + same-day-last-week
                      + recent trend → today's forecast"
```

**Empirical hierarchy of feature importance** (from Kaggle TS competitions):

```
1. Lag features         ← Most important
2. Rolling statistics
3. Calendar features (day of week, month, holiday)
4. External covariates (price, weather, promotions)
5. Fourier terms
6. Interaction features
```

---

## 2. Lag Features

### 2.1 Definition

A **lag feature** is the value of the target variable `k` time steps in the past:

```
lag_1   = y(t-1)    → previous period value
lag_7   = y(t-7)    → same day last week
lag_28  = y(t-28)   → same day ~4 weeks ago
lag_365 = y(t-365)  → same day last year
```

### 2.2 Creating Lag Features

```python
import pandas as pd
import numpy as np

def create_lag_features(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df

# Example: daily sales
lags = [1, 2, 3, 7, 14, 21, 28, 365]
df = create_lag_features(df, target_col="sales", lags=lags)
df = df.dropna()   # drop first max(lags) rows where lags are NaN
```

### 2.3 ACF-Guided Lag Selection

```python
from statsmodels.tsa.stattools import acf

acf_vals = acf(series, nlags=60, fft=True)
conf = 1.96 / np.sqrt(len(series))
significant_lags = [k for k in range(1, 61) if abs(acf_vals[k]) > conf]
print(f"Significant lags: {significant_lags}")
```

### 2.4 Lag Selection Guide

| Strategy | Lags | When |
|----------|------|------|
| ACF-guided | Only significant lags | Best — data-driven |
| Domain knowledge | 1, 7, 14, 28, 365 | When seasonality is known |
| All short lags | 1, 2, ..., 14 | Let the model select via regularization |
| Sparse multi-scale | 1, 7, 28, 91, 182, 365 | Covers key time scales compactly |

---

## 3. Calendar Features

Calendar features encode **time structure** that is always known in advance — making them completely leakage-safe.

### 3.1 Basic Calendar Features

```python
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index
    df["month"]          = idx.month           # 1–12
    df["day"]            = idx.day             # 1–31
    df["day_of_week"]    = idx.dayofweek       # 0=Monday, 6=Sunday
    df["day_of_year"]    = idx.dayofyear       # 1–366
    df["week_of_year"]   = idx.isocalendar().week.astype(int)
    df["quarter"]        = idx.quarter         # 1–4
    df["hour"]           = idx.hour            # 0–23 (hourly data)
    df["is_weekend"]     = (idx.dayofweek >= 5).astype(int)
    df["is_month_start"] = idx.is_month_start.astype(int)
    df["is_month_end"]   = idx.is_month_end.astype(int)
    df["is_quarter_end"] = idx.is_quarter_end.astype(int)
    return df
```

### 3.2 Holiday Features

```python
import holidays

def add_holiday_features(df: pd.DataFrame, country: str = "IN") -> pd.DataFrame:
    df = df.copy()
    h = holidays.country_holidays(country)
    df["is_holiday"] = df.index.map(lambda d: 1 if d in h else 0)
    return df
```

### 3.3 Cyclical Encoding

Raw integers (month=1..12) imply December and January are 11 apart, but they're adjacent. Cyclical encoding fixes this:

```python
def cyclical_encode(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    df = df.copy()
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    return df

df = cyclical_encode(df, col="month",       period=12)
df = cyclical_encode(df, col="day_of_week", period=7)
df = cyclical_encode(df, col="hour",        period=24)
```

---

## 4. Fourier Terms for Seasonality

**Fourier terms** approximate seasonal patterns using sine and cosine waves — more compact and flexible than one-hot encoding.

### 4.1 Mathematical Basis

```
S(t) ≈ Σₖ₌₁ᴷ [aₖ · sin(2πkt/T) + bₖ · cos(2πkt/T)]

T = seasonal period
K = number of Fourier pairs (higher K = more complex seasonal shape)
```

### 4.2 Implementation

```python
def add_fourier_terms(df: pd.DataFrame, period: float, n_terms: int) -> pd.DataFrame:
    df = df.copy()
    t = np.arange(len(df))
    for k in range(1, n_terms + 1):
        df[f"fourier_sin_{period:.0f}_{k}"] = np.sin(2 * np.pi * k * t / period)
        df[f"fourier_cos_{period:.0f}_{k}"] = np.cos(2 * np.pi * k * t / period)
    return df

# Daily data: weekly + yearly Fourier features
df = add_fourier_terms(df, period=7,      n_terms=3)   # weekly
df = add_fourier_terms(df, period=365.25, n_terms=6)   # yearly
```

### 4.3 How Many Terms (K)?

| Period | Seasonal Shape | Recommended K |
|--------|---------------|--------------|
| 7 (weekly) | Simple | 2–3 |
| 365.25 (yearly) | Moderate | 5–8 |
| 365.25 (complex + holidays) | Complex | 10–15 |
| 24 (hourly) | Moderate | 3–5 |

---

## 5. Target Encoding

### 5.1 Smooth Target Encoding

Encodes a categorical variable as the **mean target value** for that category, with smoothing to avoid overfitting small groups:

```python
def target_encode(train: pd.DataFrame, test: pd.DataFrame,
                  cat_col: str, target_col: str, smoothing: int = 10):
    global_mean = train[target_col].mean()
    stats = train.groupby(cat_col)[target_col].agg(["mean", "count"])
    stats["encoded"] = (
        (stats["count"] * stats["mean"] + smoothing * global_mean)
        / (stats["count"] + smoothing)
    )
    train = train.copy()
    test  = test.copy()
    train[f"{cat_col}_te"] = train[cat_col].map(stats["encoded"])
    test[f"{cat_col}_te"]  = test[cat_col].map(stats["encoded"]).fillna(global_mean)
    return train, test

# CRITICAL: Always fit encoding on TRAINING SET ONLY
train, test = target_encode(train_df, test_df, cat_col="store_id", target_col="sales")
```

---

## 6. Interaction Features

```python
# Promotion × weekend interaction
df["promo_weekend"] = df["is_promo"] * df["is_weekend"]

# Year-over-year growth ratio
df["yoy_ratio"] = df["sales"] / (df["lag_365"] + 1e-8)

# Week-over-week growth
df["wow_ratio"] = df["sales"] / (df["lag_7"] + 1e-8)

# Price × promotion
df["price_x_promo"] = df["price"] * df["is_promo"]
```

---

## 7. Normalization and Scaling

Scaling is **optional for tree-based models** (XGBoost, LightGBM) but **mandatory for deep learning** (LSTM, TFT, Transformer). Wrong scaling is one of the most common silent bugs in TS pipelines.

### 7.1 Methods

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np

# StandardScaler: zero mean, unit variance
# Best for: normally distributed series, ARIMA residuals
scaler_std = StandardScaler()
train_scaled = scaler_std.fit_transform(train.values.reshape(-1, 1))
test_scaled  = scaler_std.transform(test.values.reshape(-1, 1))

# MinMaxScaler: scale to [0, 1]
# Best for: bounded data (e.g., percentages, prices)
scaler_mm = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler_mm.fit_transform(train.values.reshape(-1, 1))

# RobustScaler: uses median and IQR — robust to outliers
# Best for: financial data, IoT sensor data with spikes
scaler_rb = RobustScaler()
train_scaled = scaler_rb.fit_transform(train.values.reshape(-1, 1))
```

### 7.2 The Golden Rule of Scaling

```python
# ✅ CORRECT: Fit on train, transform both
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
test_scaled  = scaler.transform(test.reshape(-1, 1)).flatten()

# ❌ WRONG: Fit on all data (leaks test statistics into training)
scaler.fit_transform(all_data.reshape(-1, 1))   # DO NOT DO THIS!
```

### 7.3 Per-Series Scaling (Global Models)

When training a **global model** on many time series (e.g., thousands of retail SKUs), scale each series independently:

```python
import pandas as pd
import numpy as np

def scale_series_globally(
    df: pd.DataFrame,      # long format: columns = [unique_id, ds, y]
    target_col: str = "y",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize each unique series by its own mean and std.
    Returns scaled DataFrame and a scale_params DataFrame for inverse transform.
    """
    df = df.copy()
    scale_params = (
        df.groupby("unique_id")[target_col]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    scale_params["std"] = scale_params["std"].replace(0, 1)   # avoid division by zero

    df = df.merge(scale_params, on="unique_id")
    df[target_col] = (df[target_col] - df["mean"]) / df["std"]
    df = df.drop(columns=["mean", "std"])
    return df, scale_params

def inverse_scale(forecast_df, scale_params, target_col="y"):
    """Reverse the per-series scaling to get original units."""
    forecast_df = forecast_df.merge(scale_params, on="unique_id")
    forecast_df[target_col] = forecast_df[target_col] * forecast_df["std"] + forecast_df["mean"]
    return forecast_df.drop(columns=["mean", "std"])
```

### 7.4 Log Transform as Scaling

For **right-skewed, positive** series (revenue, sales, prices):

```python
# Log1p is safer than log (handles zeros)
import numpy as np
train_log = np.log1p(train)
test_log  = np.log1p(test)

# Inverse: expm1
forecast_original_scale = np.expm1(forecast_log)
```

---

## 8. Automated Feature Extraction (tsfresh)

**`tsfresh`** automatically extracts hundreds of statistical features from time series — useful for classification and regression tasks where you don't want to hand-craft features.

### 8.1 Installation

```bash
pip install tsfresh
```

### 8.2 Basic Usage

```python
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd

# tsfresh expects a long-format DataFrame with columns:
#   id (series identifier), time (time index), value
df_long = pd.DataFrame({
    "id":    [0]*100 + [1]*100,
    "time":  list(range(100)) * 2,
    "value": series_1.tolist() + series_2.tolist(),
})

# Extract all features (>700 features by default)
features = extract_features(
    df_long,
    column_id="id",
    column_sort="time",
    column_value="value",
)
impute(features)   # fill any NaN from failed calculations
print(f"Extracted {features.shape[1]} features for {features.shape[0]} series")
```

### 8.3 Efficient Extraction (EfficientFCParameters)

```python
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters

# EfficientFCParameters: ~800 features, good balance of speed vs. coverage
features_efficient = extract_features(
    df_long,
    column_id="id",
    column_sort="time",
    default_fc_parameters=EfficientFCParameters(),
)

# MinimalFCParameters: ~10 features, very fast — good for prototyping
features_minimal = extract_features(
    df_long,
    column_id="id",
    column_sort="time",
    default_fc_parameters=MinimalFCParameters(),
)
```

### 8.4 Feature Selection

```python
# Select only statistically relevant features for a classification/regression target
y_labels = pd.Series([0, 1], index=[0, 1])   # label for each series

features_filtered = select_features(
    features,
    y_labels,
    fdr_level=0.05,   # false discovery rate threshold
)
print(f"Selected {features_filtered.shape[1]} relevant features")
```

### 8.5 When to Use tsfresh

| Task | Use tsfresh? |
|------|--------------|
| TS Classification (e.g., ECG, activity recognition) | ✅ Yes — primary use case |
| Regression on many series characteristics | ✅ Yes |
| Forecasting (future values) | ❌ No — use lag/rolling features instead |
| Anomaly detection features | ✅ Yes — can characterize windows |
| Production real-time forecasting | ❌ No — too slow for row-by-row computation |

---

## 9. The Leakage Rules

> **Core Rule**: At prediction time for period `t`, you may only use data from time `t-1` and earlier. Any feature that uses information from `t` or later is leaking.

### 7.1 Safe vs. Leaking Patterns

```python
# ✅ SAFE — lag shifts the series back in time
df["lag_1"] = df["sales"].shift(1)

# ✅ SAFE — shift BEFORE rolling (roll over t-1, t-2, ..., t-7)
df["roll7_mean"] = df["sales"].shift(1).rolling(7).mean()

# ❌ LEAKAGE — rolling WITHOUT shift includes y(t) in the window
df["roll7_mean"] = df["sales"].rolling(7).mean()   # WRONG!

# ✅ SAFE — calendar features are always known in advance
df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

# ❌ LEAKAGE — target encoding computed on train+test together
# Always fit encoding only on train set!

# ❌ LEAKAGE — using future price or promotions
df["next_price"] = df["price"].shift(-1)   # WRONG!
```

### 7.2 The Shift-Then-Roll Pattern

```python
# Correct rolling: shift(1) ensures we never include y(t) in the window
df["roll7_mean"] = df["sales"].shift(1).rolling(7).mean()
# At time t, this equals: mean(y[t-7], y[t-6], ..., y[t-1]) ✅
```

---

## 10. Reusable Feature Pipeline

```python
class TimeSeriesFeatureBuilder:
    def __init__(self, target_col: str, lags=None, rolling_windows=None,
                 fourier_periods=None, calendar=True):
        self.target_col      = target_col
        self.lags            = lags or [1, 7, 14, 28]
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.fourier_periods = fourier_periods or [(7, 3), (365.25, 6)]
        self.calendar        = calendar

    def fit(self, df):
        return self   # no train-time statistics needed for most features

    def transform(self, df):
        df = df.copy()
        t = self.target_col
        for lag in self.lags:
            df[f"lag_{lag}"] = df[t].shift(lag)
        for w in self.rolling_windows:
            s = df[t].shift(1)
            df[f"roll{w}_mean"] = s.rolling(w, min_periods=1).mean()
            df[f"roll{w}_std"]  = s.rolling(w, min_periods=1).std()
        if self.calendar:
            idx = df.index
            df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
            df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
            df["dow_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 7)
            df["dow_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 7)
            df["is_weekend"]= (idx.dayofweek >= 5).astype(int)
        t_num = np.arange(len(df))
        for period, n_terms in self.fourier_periods:
            for k in range(1, n_terms + 1):
                df[f"f_s{period:.0f}_{k}"] = np.sin(2 * np.pi * k * t_num / period)
                df[f"f_c{period:.0f}_{k}"] = np.cos(2 * np.pi * k * t_num / period)
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)


# Usage
builder = TimeSeriesFeatureBuilder(
    target_col="sales", lags=[1, 7, 14, 28, 365],
    rolling_windows=[7, 30], fourier_periods=[(7, 3), (365.25, 6)]
)
train_features = builder.fit_transform(train_df).dropna()
test_features  = builder.transform(test_df).dropna()
```

---

*← [04 — Outlier Detection](./04_outlier_detection_and_treatment.md) | [Module README](./README.md) | Next: [06 — Windowing & Rolling](./06_windowing_and_rolling_features.md) →*
