# 📘 Module 02 — Data Engineering for Time Series

> **Level**: 🟢 Beginner–Intermediate | **Prerequisites**: [Module 01 — Foundations](../01_foundations/README.md), pandas & NumPy proficiency
>
> Time series data is messy in unique ways — irregular timestamps, missing values, outliers, and leakage traps that don't exist in tabular data. This module covers every preprocessing and feature engineering technique you need before any modeling begins.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Load and parse time series data from CSVs, APIs, databases, and Parquet files
- Resample time series to any target frequency (upsample and downsample)
- Detect and handle missing values using multiple imputation strategies
- Identify and treat outliers without introducing data leakage
- Engineer lag features, rolling statistics, calendar features, and Fourier terms
- Build sliding window datasets ready for ML and deep learning models

---

## 🔗 Prerequisites

- [Module 01 — Foundations](../01_foundations/README.md)
- Pandas and NumPy proficiency

---

## 📂 Module Contents

### 📒 Theory Notes

| File | Topic | Description |
|------|-------|-------------|
| [`01_data_collection_sources.md`](./01_data_collection_sources.md) | Data Collection & Sources | CSV/Parquet loading, datetime parsing, APIs, databases, DatetimeIndex |
| [`02_resampling_and_frequency.md`](./02_resampling_and_frequency.md) | Resampling & Frequency | Downsampling, upsampling, aggregation rules, OHLC, offset aliases |
| [`03_handling_missing_values.md`](./03_handling_missing_values.md) | Handling Missing Values | Forward/backward fill, interpolation, KNN, MICE, missingness patterns |
| [`04_outlier_detection_and_treatment.md`](./04_outlier_detection_and_treatment.md) | Outlier Detection & Treatment | IQR, Z-score, STL residuals, Isolation Forest, Winsorizing, treatment strategies |
| [`05_feature_engineering_for_ts.md`](./05_feature_engineering_for_ts.md) | Feature Engineering | Lag features, calendar features, Fourier terms, target encoding, leakage rules |
| [`06_windowing_and_rolling_features.md`](./06_windowing_and_rolling_features.md) | Windowing & Rolling Features | Sliding windows, rolling mean/std/min/max, expanding windows, EWM |

### 💻 Code Practicals

| File | What It Demonstrates |
|------|----------------------|
| [`code/01_resampling_demo.py`](./code/01_resampling_demo.py) | Frequency conversion, OHLC aggregation, forward/back fill on gaps, date range |
| [`code/02_missing_values.py`](./code/02_missing_values.py) | Multiple imputation strategies with side-by-side comparison and diagnostics |
| [`code/03_outlier_handling.py`](./code/03_outlier_handling.py) | Detection pipeline (IQR + STL + Isolation Forest) and treatment (Winsorize, impute) |
| [`code/04_feature_engineering.py`](./code/04_feature_engineering.py) | Full ML-ready feature engineering pipeline — lags, rolling, calendar, Fourier |

---

## 🗺️ Learning Path (Recommended Order)

```
01_data_collection_sources.md
        ↓
02_resampling_and_frequency.md
        ↓
03_handling_missing_values.md
        ↓
04_outlier_detection_and_treatment.md
        ↓
05_feature_engineering_for_ts.md
        ↓
06_windowing_and_rolling_features.md
        ↓
code/01_resampling_demo.py
code/02_missing_values.py
code/03_outlier_handling.py
code/04_feature_engineering.py
```

---

## ⚠️ The Golden Rule of Time Series Data Engineering

> **Never use future information when engineering features for the past.**
>
> All rolling windows, lag computations, and imputation must be computed **strictly from data available at or before time `t`**. Violating this creates **data leakage** — the silent killer of seemingly good models.

---

## 📌 Key Takeaways

1. Always parse datetimes and set a **DatetimeIndex** as the first step
2. Choose aggregation method (sum vs. mean) based on **domain semantics**, not convenience
3. Lag/rolling features must be created **after the train/test split** or with strict shift alignment
4. STL residuals are the most reliable way to detect **contextual outliers** in time series
5. **Fourier terms** are the most compact way to capture multiple seasonalities for ML models
6. A well-structured feature engineering pipeline is **reproducible** — always wrap it in a function or class

---

## 📖 Further Reading

- [Pandas Time Series Documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [Feature Engineering for Time Series (Skforecast)](https://skforecast.org/latest/user_guides/feature-engineering.html)
- [Forecasting: Principles and Practice — Ch. 13 (Useful Predictors)](https://otexts.com/fpp3/useful-predictors.html)

---

*← [Module 01 — Foundations](../01_foundations/README.md) | Back to [Master README](../README.md) | Next: [Module 03 — Statistical Models](../03_statistical_models/README.md) →*
