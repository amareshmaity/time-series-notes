# 03 — Handling Missing Values in Time Series

> **Module**: 02 Data Engineering | **File**: 3 of 6
>
> Missing values in time series are not random — they have temporal patterns. The wrong imputation strategy can destroy the autocorrelation structure that your model depends on.

---

## Table of Contents

1. [Types of Missingness](#1-types-of-missingness)
2. [Diagnosing Missing Values](#2-diagnosing-missing-values)
3. [Simple Imputation Methods](#3-simple-imputation-methods)
4. [Interpolation Methods](#4-interpolation-methods)
5. [Advanced Imputation](#5-advanced-imputation)
6. [Seasonal and Trend-Aware Imputation](#6-seasonal-and-trend-aware-imputation)
7. [Choosing the Right Strategy](#7-choosing-the-right-strategy)
8. [After Imputation — Validation](#8-after-imputation--validation)

---

## 1. Types of Missingness

Understanding **why** data is missing determines **how** to fill it.

### 1.1 Missing Completely at Random (MCAR)

The probability of missing is unrelated to any variable — pure random gaps.

```
Example: Server downtime at random intervals drops sensor readings.
Fix:     Forward fill, interpolation, or model-based imputation work well.
```

### 1.2 Missing at Random (MAR)

Missingness is related to **other observed variables**, not to the missing value itself.

```
Example: A thermometer fails during high temperatures (temperature logged elsewhere).
Fix:     Use other features to impute (KNN, regression imputation).
```

### 1.3 Missing Not at Random (MNAR)

Missingness is related to the **value of the missing variable itself**.

```
Example: Patients with extreme lab values are too sick to attend check-ups.
         The data is missing BECAUSE of the health state — systematically biased.
Fix:     Hardest case — may require domain knowledge, sensitivity analysis,
         or flagging missingness as a feature.
```

### 1.4 Temporal Patterns of Missingness

| Pattern | Description | Example |
|---------|-------------|---------|
| **Single gap** | One isolated missing point | Sensor spike filtered out |
| **Short block** | 2–5 consecutive NaNs | Short network outage |
| **Long block** | Many consecutive NaNs | Device offline for days |
| **Periodic** | Regular pattern of NaNs | Night shift with no readings |
| **End censoring** | Missing at the start or end | New sensor not yet deployed |

---

## 2. Diagnosing Missing Values

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno   # pip install missingno

# Basic missing value count
print(f"Total missing   : {series.isna().sum()}")
print(f"Pct missing     : {series.isna().mean() * 100:.2f}%")
print(f"Longest gap     : {series.isna().astype(int).groupby(series.notna().cumsum()).sum().max()} steps")

# Identify gap locations and lengths
def find_gaps(series: pd.Series) -> pd.DataFrame:
    """Return a DataFrame describing each contiguous gap."""
    is_null = series.isna()
    gap_id = (~is_null).cumsum()
    gaps = []
    for g_id, group in series[is_null].groupby(gap_id[is_null]):
        gaps.append({
            "start":  group.index[0],
            "end":    group.index[-1],
            "length": len(group),
        })
    return pd.DataFrame(gaps)

gaps_df = find_gaps(series)
print(f"\nNumber of gaps : {len(gaps_df)}")
print(gaps_df.head(10))

# Visual: missingness matrix
msno.matrix(df)
plt.show()

# Plot missing as black bars on timeline
fig, ax = plt.subplots(figsize=(13, 3))
ax.plot(series.index, series.isna().astype(int), color="red", linewidth=0.5)
ax.fill_between(series.index, series.isna().astype(int), alpha=0.4, color="red")
ax.set_title("Missing Value Timeline (red = missing)")
ax.set_ylabel("Is Missing")
plt.tight_layout()
plt.show()
```

---

## 3. Simple Imputation Methods

### 3.1 Forward Fill (ffill / pad)

Propagate the **last valid observation forward**. Creates a step-function effect.

```python
series_ffill = series.ffill()

# Limit how far to fill
series_ffill_limited = series.ffill(limit=3)   # fill max 3 consecutive NaNs
```

**When to use:**
- State or level variables (price, temperature, inventory level)
- Short gaps (1–3 steps)
- When the measurement is believed to not change until the next reading

**Drawback**: Underestimates variance; creates flat segments that inflate lag correlations.

### 3.2 Backward Fill (bfill)

Fill NaN with the **next valid observation** going backward.

```python
series_bfill = series.bfill()
series_bfill_limited = series.bfill(limit=2)
```

**When to use:** When the next observation is a better estimate than the previous one (e.g., scheduled meter readings that apply to the period before).

### 3.3 Constant / Mean Fill

```python
# Fill with series mean (destroys autocorrelation structure)
series_mean_fill = series.fillna(series.mean())

# Fill with a domain constant (e.g., 0 for missing sales)
series_zero_fill = series.fillna(0)
```

> ⚠️ **Avoid mean fill for time series.** It destroys temporal autocorrelation by inserting flat values, and is inappropriate for trended or seasonal data.

### 3.4 Seasonal Mean Fill

Fill with the **average value from the same period** (e.g., same month, same day-of-week):

```python
def seasonal_fill(series: pd.Series, period: str = "month") -> pd.Series:
    """Fill NaN with the mean from the same seasonal period."""
    s = series.copy()
    if period == "month":
        group_key = s.index.month
    elif period == "dayofweek":
        group_key = s.index.dayofweek
    elif period == "hour":
        group_key = s.index.hour
    else:
        raise ValueError(f"Unknown period: {period}")

    seasonal_means = s.groupby(group_key).transform("mean")
    s = s.fillna(seasonal_means)
    return s

series_seasonal = seasonal_fill(series, period="month")
```

---

## 4. Interpolation Methods

Interpolation estimates missing values **between known observations**. Better than forward fill for smooth, continuously changing quantities.

### 4.1 Linear Interpolation

Draws a straight line between the two surrounding valid values.

```python
series_linear = series.interpolate(method="linear")

# Limit the number of consecutive NaNs to fill
series_linear_limit = series.interpolate(method="linear", limit=5, limit_direction="both")
```

**Mathematical formula:**
```
Y_fill(t) = Y(t₁) + (Y(t₂) - Y(t₁)) × (t - t₁) / (t₂ - t₁)
```

### 4.2 Polynomial Interpolation

Fits a polynomial of order `k` through surrounding points:

```python
series_poly = series.interpolate(method="polynomial", order=2)  # quadratic
series_poly3 = series.interpolate(method="polynomial", order=3) # cubic
```

**Caution**: High-order polynomials can oscillate wildly (Runge's phenomenon) — keep order ≤ 3.

### 4.3 Spline Interpolation

Piecewise polynomial — smooth and stable:

```python
series_spline = series.interpolate(method="spline", order=3)  # cubic spline
```

### 4.4 Time-Based Interpolation

Accounts for **unequal time spacing** between observations:

```python
# Interpolate proportional to actual time gaps (important for irregular series)
series_time = series.interpolate(method="time")
```

### 4.5 Comparison of Interpolation Methods

```python
methods = ["linear", "polynomial", "spline", "time"]
fig, axes = plt.subplots(len(methods), 1, figsize=(13, 10), sharex=True)

for i, method in enumerate(methods):
    if method in ["polynomial", "spline"]:
        filled = series.interpolate(method=method, order=3)
    else:
        filled = series.interpolate(method=method)
    
    axes[i].plot(series.index, series.values, "o", markersize=4, color="blue", label="Observed")
    axes[i].plot(filled.index, filled.values, linewidth=1.5, color="red", alpha=0.7, label=f"{method} interp")
    axes[i].set_title(f"Method: {method}")
    axes[i].legend(fontsize=8)

plt.suptitle("Interpolation Method Comparison", fontweight="bold")
plt.tight_layout()
plt.show()
```

---

## 5. Advanced Imputation

### 5.1 KNN Imputation

Finds the `k` most similar time windows and uses their values to impute:

```python
from sklearn.impute import KNNImputer
import numpy as np

# Reshape for sklearn (needs 2D input)
values = series.values.reshape(-1, 1)

imputer = KNNImputer(n_neighbors=5, weights="distance")
values_filled = imputer.fit_transform(values)
series_knn = pd.Series(values_filled.flatten(), index=series.index)
```

**Note**: KNN imputation ignores temporal ordering — it finds k-nearest by value similarity, not temporal proximity. Works better when used with lag features as additional columns.

### 5.2 Multivariate Imputation (MICE / IterativeImputer)

Uses relationships between multiple columns to impute each column iteratively:

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Works on a DataFrame with multiple columns
df_with_nans = df[["temp", "humidity", "pressure"]].copy()

imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
    max_iter=10,
    random_state=42,
)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df_with_nans),
    columns=df_with_nans.columns,
    index=df_with_nans.index,
)
```

**When to use MICE**: Multivariate sensor data where columns are strongly correlated.

---

## 6. Seasonal and Trend-Aware Imputation

The most sophisticated approach for time series — uses decomposition to impute coherently.

```python
from statsmodels.tsa.seasonal import STL
import numpy as np

def stl_impute(series: pd.Series, period: int = 12, max_iter: int = 5) -> pd.Series:
    """
    Iterative STL imputation:
    1. Interpolate NaN linearly (temporary)
    2. Decompose with STL
    3. Replace NaN positions with trend + seasonal estimate
    4. Repeat until convergence
    """
    s = series.copy()
    nan_mask = s.isna()
    
    if nan_mask.sum() == 0:
        return s
    
    # Initial fill for STL to work (STL can't handle NaN)
    s_temp = s.interpolate(method="linear").ffill().bfill()
    
    for iteration in range(max_iter):
        stl = STL(s_temp, period=period, robust=True)
        result = stl.fit()
        # Imputed value = trend + seasonal (ignore remainder for smoother fill)
        imputed_values = result.trend + result.seasonal
        # Only update originally missing positions
        s_temp[nan_mask] = imputed_values[nan_mask]
    
    return s_temp

series_stl_imputed = stl_impute(series, period=12)
```

---

## 7. Choosing the Right Strategy

```
Decision Tree:
│
├── Is the gap very short (1–3 steps)?
│   └── YES → Forward fill (for state variables) or linear interpolation (for smooth data)
│
├── Is the series strongly seasonal?
│   └── YES → Seasonal mean fill or STL-based imputation
│
├── Is the series strongly trended?
│   └── YES → Linear or polynomial interpolation
│
├── Are multiple correlated columns available?
│   └── YES → MICE / IterativeImputer
│
├── Is the gap very long (> 20% of a seasonal cycle)?
│   └── YES → Flag as missing, create indicator variable, or exclude period
│
└── General default → Linear interpolation with `limit` parameter
```

| Method | Short Gaps | Long Gaps | Seasonal | Multivariate | Speed |
|--------|-----------|-----------|----------|-------------|-------|
| Forward fill | ✅ | ❌ | ❌ | ❌ | Instant |
| Linear interpolation | ✅ | OK | ❌ | ❌ | Instant |
| Seasonal mean fill | OK | OK | ✅ | ❌ | Fast |
| KNN imputation | OK | OK | OK | ✅ | Moderate |
| MICE | OK | ✅ | OK | ✅ | Slow |
| STL imputation | ✅ | ✅ | ✅ | ❌ | Moderate |

---

## 8. After Imputation — Validation

Always validate that imputation:
1. Did not introduce negative values (if domain-constrained)
2. Did not create spikes at gap boundaries
3. Did not significantly alter ACF/PACF structure
4. Did not inflate or deflate variance

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(13, 9))

# Original vs imputed
axes[0].plot(series, color="blue", label="Original (with NaNs)", linewidth=1)
axes[0].plot(series_imputed, color="red", alpha=0.7, linestyle="--", label="Imputed", linewidth=1)
axes[0].set_title("Original vs. Imputed Series")
axes[0].legend()

# Difference at imputed positions
diff = series_imputed - series
axes[1].plot(diff[series.isna()], "ro", markersize=4, label="Imputed values")
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_title("Imputed Values (difference from original where observed)")
axes[1].legend()

# ACF comparison
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series.dropna(), lags=30, ax=axes[2], alpha=0.05)
axes[2].set_title("ACF of Original (observed only)")

plt.tight_layout()
plt.show()

# Statistical comparison
print("Before imputation:")
print(series.dropna().describe())
print("\nAfter imputation:")
print(series_imputed.describe())
```

---

*← [02 — Resampling](./02_resampling_and_frequency.md) | [Module README](./README.md) | Next: [04 — Outlier Detection](./04_outlier_detection_and_treatment.md) →*
