# 02 — Resampling & Frequency Conversion

> **Module**: 02 Data Engineering | **File**: 2 of 6
>
> Real-world time series rarely come at the frequency you need. Resampling converts between frequencies correctly — and choosing the wrong aggregation method silently corrupts your data.

---

## Table of Contents

1. [What is Resampling?](#1-what-is-resampling)
2. [Downsampling](#2-downsampling)
3. [Upsampling](#3-upsampling)
4. [Aggregation Rules by Domain](#4-aggregation-rules-by-domain)
5. [OHLC Aggregation](#5-ohlc-aggregation)
6. [Handling Timezone-Aware Resampling](#6-handling-timezone-aware-resampling)
7. [Filling Gaps vs. Resampling](#7-filling-gaps-vs-resampling)
8. [Practical Patterns](#8-practical-patterns)

---

## 1. What is Resampling?

**Resampling** is the process of converting a time series from one frequency to another.

```
Downsampling: High frequency → Low frequency
  e.g., Hourly → Daily → Weekly → Monthly

Upsampling:   Low frequency → High frequency
  e.g., Monthly → Daily → Hourly
```

### Why Resampling?

| Reason | Example |
|--------|---------|
| Model requires a specific frequency | SARIMA needs regular monthly data |
| Reduce noise | Smooth out minute-level spikes into hourly averages |
| Align multiple series | Join daily series with weekly series |
| Reduce dataset size | Convert tick data to 1-minute bars for storage |
| Business aggregation | Monthly revenue report from daily transactions |

---

## 2. Downsampling

### 2.1 Basic Syntax

```python
import pandas as pd

# Core pattern: df.resample(rule).agg_function()
daily_to_monthly_sum  = df["sales"].resample("MS").sum()
daily_to_weekly_mean  = df["temp"].resample("W").mean()
hourly_to_daily_max   = df["load"].resample("D").max()
```

### 2.2 Common Aggregation Functions

```python
series = df["value"]

# Standard aggregations
monthly_sum    = series.resample("MS").sum()      # revenue, sales, count
monthly_mean   = series.resample("MS").mean()     # temperature, rate
monthly_median = series.resample("MS").median()   # robust to outliers
monthly_max    = series.resample("MS").max()      # peak load, high price
monthly_min    = series.resample("MS").min()      # minimum inventory
monthly_std    = series.resample("MS").std()      # volatility
monthly_count  = series.resample("MS").count()    # number of observations
monthly_first  = series.resample("MS").first()    # opening price
monthly_last   = series.resample("MS").last()     # closing price

# Custom aggregation
monthly_range  = series.resample("MS").agg(lambda x: x.max() - x.min())
```

### 2.3 Multiple Columns with Different Aggregations

```python
df_daily = pd.DataFrame({
    "sales":    [...],   # sum when downsampling
    "price":    [...],   # mean when downsampling
    "units":    [...],   # sum when downsampling
    "in_stock": [...],   # last value (state at end of period)
})
df_daily.index = pd.DatetimeIndex(df_daily.index)

# Apply different aggregations per column
df_monthly = df_daily.resample("MS").agg({
    "sales":    "sum",
    "price":    "mean",
    "units":    "sum",
    "in_stock": "last",
})
```

### 2.4 Closed and Label Parameters

These control **which end of the period** is closed/labeled:

```python
# Default for most frequencies: left-closed, left-labeled
# Example: weekly resample — week starts Monday (left)
weekly = series.resample("W-MON", closed="left", label="left").sum()

# For right-closed (e.g., end of day = 24:00 belongs to that day):
daily = series.resample("D", closed="right", label="right").sum()
```

---

## 3. Upsampling

### 3.1 What Happens During Upsampling

Upsampling creates **new timestamps** between existing ones. The new values are `NaN` by default — they must be filled using a strategy.

```python
# Monthly to daily (creates NaN for all new daily rows)
monthly_series = pd.Series([100, 120, 110], index=pd.date_range("2023-01", periods=3, freq="MS"))
daily_series = monthly_series.resample("D").asfreq()
print(daily_series.head(10))
# Only Jan 1, Feb 1, Mar 1 have values. All other days are NaN.
```

### 3.2 Filling Upsampled Values

```python
# Forward fill — last known value persists (step function)
daily_ffill = monthly_series.resample("D").ffill()

# Backward fill — next known value fills backwards
daily_bfill = monthly_series.resample("D").bfill()

# Linear interpolation — proportional fill between points
daily_interp = monthly_series.resample("D").interpolate(method="linear")

# Cubic spline — smooth interpolation
daily_cubic = monthly_series.resample("D").interpolate(method="cubic")
```

**When to use each:**

| Method | When Appropriate |
|--------|-----------------|
| `ffill()` | State variables (price, temperature) — last value carries forward |
| `bfill()` | When you know next period's value is the right fill |
| `interpolate("linear")` | Gradually changing quantities (population, cumulative sales) |
| `interpolate("cubic")` | Smooth physical measurements |
| Custom distribution | Allocate monthly total to days by known pattern |

### 3.3 Distributing Totals (Monthly → Daily)

A common need: given monthly totals, distribute them to daily values proportionally.

```python
# Distribute monthly total equally across all days in the month
def distribute_monthly_to_daily(monthly_series: pd.Series) -> pd.Series:
    """Convert monthly totals to equal daily values."""
    daily = monthly_series.resample("D").ffill()   # forward fill first
    # Divide each day's value by number of days in that month
    days_in_month = daily.index.days_in_month
    return daily / days_in_month

daily_distributed = distribute_monthly_to_daily(monthly_series)
```

---

## 4. Aggregation Rules by Domain

**Choosing the wrong aggregation silently corrupts your data.** Use this table:

| Domain | Variable | Correct Aggregation | Wrong Aggregation |
|--------|----------|--------------------|--------------------|
| **Retail** | Daily sales revenue | `sum` | `mean` |
| **Retail** | Item price | `mean` or `last` | `sum` |
| **Energy** | Power consumption (kWh) | `sum` | `mean` |
| **Energy** | Instantaneous load (kW) | `mean` | `sum` |
| **Finance** | Trading volume | `sum` | `mean` |
| **Finance** | Stock price | `last` (close) | `sum` |
| **Weather** | Daily rainfall (mm) | `sum` | `mean` |
| **Weather** | Temperature | `mean` (or min/max) | `sum` |
| **IoT Sensor** | Count of events | `sum` | `mean` |
| **IoT Sensor** | Sensor reading (temperature, pressure) | `mean` | `sum` |
| **Inventory** | Stock level (state) | `last` | `sum` |
| **Traffic** | Vehicle count | `sum` | `mean` |
| **Traffic** | Speed | `mean` | `sum` |

> **Rule of thumb**: Flows (things that accumulate) → `sum`. Stocks/States (snapshots) → `mean` or `last`.

---

## 5. OHLC Aggregation

**OHLC** (Open, High, Low, Close) is standard in financial time series:

```python
# Resample tick/minute data to hourly OHLC bars
ohlc = df["price"].resample("h").ohlc()
print(ohlc.head())
# Columns: open, high, low, close

# Add volume
ohlc["volume"] = df["volume"].resample("h").sum()

# Daily OHLC from minute data
ohlc_daily = df["price"].resample("D").ohlc()
ohlc_daily["volume"] = df["volume"].resample("D").sum()
```

---

## 6. Handling Timezone-Aware Resampling

```python
# Convert to timezone-aware before resampling
df.index = df.index.tz_localize("UTC")
df_ist = df.copy()
df_ist.index = df_ist.index.tz_convert("Asia/Kolkata")

# Resample in local time (respects DST boundaries)
daily_ist = df_ist["value"].resample("D").sum()

# Resample in UTC, then convert result
daily_utc = df["value"].resample("D").sum()
daily_utc.index = daily_utc.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
```

> **Warning**: Resampling across DST boundaries can create days with 23 or 25 hours. Always resample in UTC and convert after, or explicitly handle DST.

---

## 7. Filling Gaps vs. Resampling

**Gaps** are missing timestamps in an otherwise regular series. They are different from upsampling.

```python
# Check for gaps (missing timestamps)
expected_idx = pd.date_range(start=series.index.min(), end=series.index.max(), freq="D")
missing_dates = expected_idx.difference(series.index)
print(f"Missing timestamps: {len(missing_dates)}")
print(missing_dates[:10])

# Fill gaps by reindexing
series_complete = series.reindex(expected_idx)
# NaN introduced at missing timestamps → apply imputation (see topic 03)

# Fill gaps with forward fill only (quick fix)
series_filled = series.reindex(expected_idx).ffill()
```

---

## 8. Practical Patterns

### Pattern 1: Align Two Series at Different Frequencies

```python
# Series at daily and monthly frequency
daily = df_daily["sales"]            # Daily
monthly_budget = df_budget["budget"] # Monthly

# Upsample budget to daily (forward fill)
budget_daily = monthly_budget.resample("D").ffill()

# Now both are daily — can join
df_joined = pd.DataFrame({"sales": daily, "budget": budget_daily})
```

### Pattern 2: Multi-Level Aggregation

```python
# Group by category, then resample each group
result = (
    df.groupby("category")["sales"]
    .resample("MS")
    .sum()
    .reset_index()
)
```

### Pattern 3: Rolling Resample (Custom Windows)

```python
# Trailing 4-week sum (not calendar monthly)
trailing_4w = series.resample("W").sum().rolling(window=4).sum()
```

### Pattern 4: Validate Resample Counts

```python
# Check how many observations fall in each period
count_per_period = series.resample("MS").count()
expected_days = series.resample("MS").apply(lambda x: len(x.index))
# If count < expected_days → gaps exist within that period
```

---

*← [01 — Data Collection](./01_data_collection_sources.md) | [Module README](./README.md) | Next: [03 — Missing Values](./03_handling_missing_values.md) →*
