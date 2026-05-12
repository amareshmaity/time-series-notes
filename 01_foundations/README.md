# 📘 Module 01 — Foundations of Time Series

> **Level**: 🟢 Beginner | **Prerequisites**: Python basics (pandas, numpy, matplotlib)
>
> This module builds the theoretical bedrock for all time series work. Every concept here is referenced in every module that follows. Master this before touching any model.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Define time series data and distinguish it from cross-sectional data
- Identify trend, seasonality, cyclicality, and noise components
- Understand and test for stationarity using ADF, KPSS, Phillips-Perron, and Zivot-Andrews tests
- Interpret ACF and PACF plots to identify model order
- Use the Cross-Correlation Function (CCF) to find leading/lagging relationships between two series
- Apply STL decomposition and interpret each component
- Recognize different types of time series patterns in real data

---

## 🔗 Prerequisites

- Basic Python — pandas, numpy, matplotlib
- No prior time series knowledge required

---

## 📂 Module Contents

### 📒 Theory Notes

| File | Topic | Description |
|------|-------|-------------|
| [`01_what_is_time_series.md`](./01_what_is_time_series.md) | What is Time Series? | Definition, examples, types, TS task landscape |
| [`02_components_trend_seasonality.md`](./02_components_trend_seasonality.md) | TS Components | Trend, seasonality, cyclicality, noise, additive vs multiplicative |
| [`03_stationarity.md`](./03_stationarity.md) | Stationarity | Weak stationarity, ADF, KPSS, Phillips-Perron, Zivot-Andrews, differencing |
| [`04_autocorrelation_acf_pacf.md`](./04_autocorrelation_acf_pacf.md) | ACF, PACF & CCF | Autocorrelation, ACF, PACF, CCF, Ljung-Box, model order selection |
| [`05_decomposition.md`](./05_decomposition.md) | Decomposition | Classical, STL, MSTL, trend/seasonal strength, deseasonalization |

### 💻 Code Practicals

| File | What It Demonstrates |
|------|----------------------|
| [`code/01_basics_exploration.py`](./code/01_basics_exploration.py) | Loading TS, datetime indexing, plotting, rolling stats, ADF/KPSS, ACF/PACF |
| [`code/02_decomposition_demo.py`](./code/02_decomposition_demo.py) | Classical decomposition, STL, MSTL, trend/seasonal strength, deseasonalization |

---

## 🗺️ Learning Path (Recommended Order)

```
01_what_is_time_series.md
        ↓
02_components_trend_seasonality.md
        ↓
03_stationarity.md
        ↓
04_autocorrelation_acf_pacf.md
        ↓
05_decomposition.md
        ↓
code/01_basics_exploration.py   →   code/02_decomposition_demo.py
```

---

## 📌 Key Takeaways

1. Time series rows are **not independent** — temporal order is fundamental
2. Most models require **stationarity** — always test with ADF + KPSS before modeling
3. **ACF** cuts off → MA order; **PACF** cuts off → AR order
4. Use **CCF** to detect leading/lagging relationships between two variables
5. Use **STL** over classical decomposition for all practical work
6. Check residuals with **Ljung-Box** — white noise residuals = well-specified model
7. For financial series or structural breaks: add **PP** and **Zivot-Andrews** tests

---

## 📖 Further Reading

- [Forecasting: Principles and Practice — Ch. 2, 3, 4](https://otexts.com/fpp3/)
- [statsmodels Time Series Docs](https://www.statsmodels.org/stable/tsa.html)
- [pandas Time Series Docs](https://pandas.pydata.org/docs/user_guide/timeseries.html)

---

*← Back to [Master README](../README.md) | Next: [Module 02 — Data Engineering](../02_data_engineering/README.md) →*
