# 04 — Outlier Detection & Treatment in Time Series

> **Module**: 02 Data Engineering | **File**: 4 of 6
>
> Outliers in time series are not always errors — sometimes they are the signal (anomalies in sensor data). The goal is to distinguish between data errors and genuine extreme events, and handle each correctly.

---

## Table of Contents

1. [Types of Outliers in Time Series](#1-types-of-outliers-in-time-series)
2. [Statistical Detection Methods](#2-statistical-detection-methods)
3. [STL Residual Method](#3-stl-residual-method)
4. [Isolation Forest](#4-isolation-forest)
5. [Treatment Strategies](#5-treatment-strategies)
6. [Outlier vs. Anomaly — Key Distinction](#6-outlier-vs-anomaly--key-distinction)
7. [Decision Framework](#7-decision-framework)

---

## 1. Types of Outliers in Time Series

| Type | Description | Example |
|------|-------------|---------|
| **Additive outlier (AO)** | Single point spike — affects only one observation | Sensor glitch, data entry error |
| **Innovative outlier (IO)** | Affects the point AND future observations through autocorrelation | Real event that alters the series level |
| **Level shift (LS)** | Sudden permanent change in mean level | Policy change, new equipment installed |
| **Transient change (TC)** | Temporary level shift that decays back | System overload that self-corrects |
| **Seasonal outlier (SO)** | Abnormal value in one specific seasonal period | One exceptional Christmas season |

```
Normal:            ───────────────────────────────
Additive Outlier:  ──────────────|spike|──────────  (one point)
Level Shift:       ──────────────|═════════════════  (permanent)
Transient Change:  ──────────────|████▓▒░──────────  (decays)
```

---

## 2. Statistical Detection Methods

### 2.1 Z-Score Method

Flags observations more than `z_thresh` standard deviations from the mean.

```python
import pandas as pd
import numpy as np

def zscore_outliers(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Return boolean mask: True where outlier."""
    z_scores = (series - series.mean()) / series.std()
    return z_scores.abs() > threshold

outlier_mask = zscore_outliers(series, threshold=3.0)
print(f"Outliers detected: {outlier_mask.sum()} ({outlier_mask.mean()*100:.2f}%)")
```

**Limitation**: Assumes a normal distribution. Non-robust — a single extreme outlier inflates the mean and std, making other outliers harder to detect.

### 2.2 Modified Z-Score (Median-Based, Robust)

Uses **median** and **MAD** (Median Absolute Deviation) — robust to the outlier itself:

```python
def modified_zscore_outliers(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """Robust Z-score using MAD — less sensitive to the outliers themselves."""
    median = series.median()
    mad = (series - median).abs().median()
    modified_z = 0.6745 * (series - median) / (mad + 1e-8)
    return modified_z.abs() > threshold

outlier_mask_robust = modified_zscore_outliers(series)
```

### 2.3 IQR (Interquartile Range) Method

```python
def iqr_outliers(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Flag values outside [Q1 - factor*IQR, Q3 + factor*IQR].
    factor=1.5 → mild outliers, factor=3.0 → extreme outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return (series < lower) | (series > upper)

outlier_mask_iqr = iqr_outliers(series, factor=1.5)
```

### 2.4 Rolling Z-Score (Time-Aware)

The standard Z-score uses the global mean/std. A **rolling Z-score** adapts to local behavior — better for trended or seasonal series:

```python
def rolling_zscore_outliers(series: pd.Series, window: int = 30, threshold: float = 3.0) -> pd.Series:
    """Rolling window Z-score — detects outliers relative to local behavior."""
    rolling_mean = series.rolling(window=window, center=True, min_periods=window // 2).mean()
    rolling_std  = series.rolling(window=window, center=True, min_periods=window // 2).std()
    z_scores = (series - rolling_mean) / (rolling_std + 1e-8)
    return z_scores.abs() > threshold

outlier_mask_rolling = rolling_zscore_outliers(series, window=30, threshold=3.0)
```

> **Rolling Z-score is preferred** over global Z-score for time series because it respects trend and seasonal context.

---

## 3. STL Residual Method

The most **domain-aware** approach for time series: decompose the series with STL, then flag outliers in the remainder (residual) component.

**Why this works**: The residual component has already had trend and seasonality removed. A value that looks normal in the raw series (e.g., a high December temperature) might be a true outlier — but the STL residual correctly contextualizes it.

```python
from statsmodels.tsa.seasonal import STL
import numpy as np
import pandas as pd

def stl_outliers(
    series: pd.Series,
    period: int = 12,
    threshold: float = 3.0,
    robust: bool = True,
) -> pd.Series:
    """
    Detect outliers using STL decomposition residuals.
    Returns boolean mask: True where outlier.
    """
    stl = STL(series, period=period, robust=robust)
    result = stl.fit()
    residuals = result.resid
    
    # Modified Z-score on residuals (robust)
    median_resid = residuals.median()
    mad = (residuals - median_resid).abs().median()
    modified_z = 0.6745 * (residuals - median_resid) / (mad + 1e-8)
    
    return modified_z.abs() > threshold

outlier_mask_stl = stl_outliers(series, period=12, threshold=3.0)
print(f"STL-based outliers detected: {outlier_mask_stl.sum()}")
```

---

## 4. Isolation Forest

A machine learning approach — effective for **multivariate outlier detection** or when temporal features are included:

```python
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

def isolation_forest_outliers(
    series: pd.Series,
    contamination: float = 0.02,
    window: int = 5,
) -> pd.Series:
    """
    Detect outliers using Isolation Forest with temporal features.
    contamination = expected proportion of outliers (0.01–0.1 typically)
    """
    # Create feature matrix: [value, lag1, lag2, rolling_mean, rolling_std]
    df = pd.DataFrame({"value": series})
    for lag in range(1, window + 1):
        df[f"lag_{lag}"] = series.shift(lag)
    df["rolling_mean"] = series.rolling(window).mean()
    df["rolling_std"]  = series.rolling(window).std()
    df = df.dropna()
    
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    predictions = clf.fit_predict(df)   # -1 = outlier, 1 = normal
    
    # Align index with original series
    outlier_mask = pd.Series(predictions == -1, index=df.index)
    return outlier_mask.reindex(series.index, fill_value=False)

outlier_mask_if = isolation_forest_outliers(series, contamination=0.02)
```

---

## 5. Treatment Strategies

Once outliers are detected, you have three choices:

### 5.1 Remove (Set to NaN) then Impute

The safest approach — removes the bad value and fills with the appropriate method:

```python
series_treated = series.copy()
series_treated[outlier_mask] = np.nan
series_treated = series_treated.interpolate(method="linear")   # or STL impute
```

### 5.2 Winsorizing (Clip to Percentile Bounds)

Caps extreme values at a percentile threshold — preserves the observation but limits its influence:

```python
from scipy.stats import mstats

# Clip to 1st–99th percentile
series_winsorized = pd.Series(
    mstats.winsorize(series.values, limits=[0.01, 0.01]),
    index=series.index,
    name=series.name,
)

# Or manually with pandas:
lower = series.quantile(0.01)
upper = series.quantile(0.99)
series_clipped = series.clip(lower=lower, upper=upper)
```

### 5.3 Replace with Rolling Median

Replace the outlier with the local rolling median — preserves temporal locality:

```python
rolling_median = series.rolling(window=7, center=True, min_periods=1).median()
series_treated = series.copy()
series_treated[outlier_mask] = rolling_median[outlier_mask]
```

### 5.4 Keep + Flag (Recommended for Analysis)

Sometimes the outlier is real and important (e.g., Black Friday sales spike). Keep the value but add a binary feature:

```python
series_with_flag = pd.DataFrame({
    "value":       series,
    "is_outlier":  outlier_mask.astype(int),
})
# Models can learn that the flagged periods have different behavior
```

### 5.5 Side-by-Side Comparison

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

axes[0].plot(series, color="blue", linewidth=1, label="Original")
axes[0].scatter(series.index[outlier_mask], series[outlier_mask],
                color="red", s=50, zorder=5, label="Detected Outliers")
axes[0].set_title("Original Series with Detected Outliers")
axes[0].legend()

axes[1].plot(series_winsorized, color="green", linewidth=1, label="Winsorized")
axes[1].set_title("After Winsorizing (clip to 1st–99th percentile)")
axes[1].legend()

axes[2].plot(series_treated, color="orange", linewidth=1, label="Interpolated")
axes[2].set_title("After Remove + Interpolate")
axes[2].legend()

plt.suptitle("Outlier Treatment Comparison", fontweight="bold")
plt.tight_layout()
plt.show()
```

---

## 6. Outlier vs. Anomaly — Key Distinction

| | Outlier (Preprocessing) | Anomaly (Modeling Task) |
|--|------------------------|------------------------|
| **Goal** | Clean data for modeling | Detect unusual events |
| **Action** | Fix or flag before training | Report and alert in production |
| **Example** | Sensor reading stuck at 0 (malfunction) | Energy consumption 3× normal (theft?) |
| **When handled** | During data engineering | As a separate modeling task (Module 09) |

> **Rule**: In data engineering, treat outliers that are likely **data quality issues**. Preserve outliers that are likely **real events** — they may be the most important signal.

---

## 7. Decision Framework

```
Is the outlier likely a DATA ERROR (sensor glitch, entry error)?
│
├── YES → Remove (set NaN) + interpolate to fill
│
├── NO — Is it a real but extreme event?
│   ├── Will it recur? → Keep + add binary flag feature
│   └── Won't recur? → Winsorize or remove
│
└── UNSURE → Keep + flag, discuss with domain expert
```

**Method selection by series characteristics:**

| Characteristic | Best Detection Method |
|---------------|----------------------|
| Stationary, no seasonality | Z-score or IQR |
| Trended series | Rolling Z-score |
| Seasonal series | STL residual method |
| Multivariate data | Isolation Forest with feature matrix |
| High noise / fat tails | Modified Z-score (MAD-based) |
| Production pipeline | Isolation Forest (handles all cases) |

---

*← [03 — Missing Values](./03_handling_missing_values.md) | [Module README](./README.md) | Next: [05 — Feature Engineering](./05_feature_engineering_for_ts.md) →*
