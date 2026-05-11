# 📘 Module 03 — Statistical Models for Time Series

> **Level**: 🟡 Intermediate | **Prerequisites**: [Module 01](../01_foundations/README.md), [Module 02](../02_data_engineering/README.md), basic statistics
>
> Classical statistical models remain the most widely used tools in production time series systems. This module covers the full model family — from naive baselines through SARIMA, ETS, VAR, and State Space models — with hands-on implementation for each.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Build and justify naive baseline models before trying anything complex
- Fit ETS (Exponential Smoothing) models for trend and seasonal series
- Understand the AR → MA → ARMA → ARIMA → SARIMA model family from first principles
- Use the Box-Jenkins methodology to identify, estimate, and diagnose ARIMA models
- Fit Vector Autoregression (VAR) models for multivariate time series
- Understand Kalman filter-based State Space models
- Select the best model using AIC/BIC and residual diagnostics

---

## 🔗 Prerequisites

- [Module 01 — Foundations](../01_foundations/README.md) (stationarity, ACF/PACF)
- [Module 02 — Data Engineering](../02_data_engineering/README.md) (preprocessing)
- Basic statistics: probability distributions, least squares regression

---

## 📂 Module Contents

### 📒 Theory Notes

| File | Topic | Description |
|------|-------|-------------|
| [`01_naive_baseline_models.md`](./01_naive_baseline_models.md) | Naive Baselines | Mean, drift, naïve, seasonal naïve — always the first benchmark |
| [`02_exponential_smoothing_ETS.md`](./02_exponential_smoothing_ETS.md) | ETS Models | SES, Holt's linear, Holt-Winters, full ETS state space framework |
| [`03_ar_ma_arma_arima_sarima.md`](./03_ar_ma_arma_arima_sarima.md) | AR → SARIMA Family | Full model family with math, intuition, Box-Jenkins methodology |
| [`04_var_vector_autoregression.md`](./04_var_vector_autoregression.md) | VAR Models | Multivariate AR, lag selection, Granger causality, IRF |
| [`05_state_space_models.md`](./05_state_space_models.md) | State Space Models | Local level, Kalman filter, structural TS models |
| [`06_model_selection_and_diagnostics.md`](./06_model_selection_and_diagnostics.md) | Model Selection | AIC/BIC, residual analysis, Ljung-Box, forecast diagnostics |

### 💻 Code Practicals

| File | What It Demonstrates |
|------|----------------------|
| [`code/01_naive_models.py`](./code/01_naive_models.py) | All four baseline forecasts with comparison plots |
| [`code/02_ets_models.py`](./code/02_ets_models.py) | SES, Holt, Holt-Winters fitting and forecasting |
| [`code/03_ar_ma_arma_arima_sarima.py`](./code/03_ar_ma_arma_arima_sarima.py) | Full model family: AR → MA → ARMA → ARIMA → SARIMA with diagnostics |
| [`code/04_var_models.py`](./code/04_var_models.py) | VAR fitting, lag selection, Granger causality, impulse response |
| [`code/05_diagnostics.py`](./code/05_diagnostics.py) | Residual plots, Ljung-Box, AIC/BIC selection, model comparison |

---

## 🗺️ Learning Path (Recommended Order)

```
01_naive_baseline_models.md           ← Always start here
        ↓
02_exponential_smoothing_ETS.md
        ↓
03_ar_ma_arma_arima_sarima.md         ← Core of this module
        ↓
04_var_vector_autoregression.md
        ↓
05_state_space_models.md
        ↓
06_model_selection_and_diagnostics.md ← Always end here
        ↓
code/ (run in order 01 → 05)
```

---

## 🧠 Model Family Overview

| Model | Data Type | Seasonal | Multivariate | Key Assumption |
|-------|-----------|----------|-------------|---------------|
| Naïve / Seasonal Naïve | Any | ✅ (seasonal) | ❌ | Last value = best forecast |
| SES / Holt / Holt-Winters | Univariate | ✅ | ❌ | Exponential decay of past influence |
| AR(p) | Univariate | ❌ | ❌ | Stationary, linear AR structure |
| MA(q) | Univariate | ❌ | ❌ | Stationary, finite MA structure |
| ARMA(p,q) | Univariate | ❌ | ❌ | Stationary, AR + MA combined |
| ARIMA(p,d,q) | Univariate | ❌ | ❌ | Non-stationary (d differences) |
| SARIMA(p,d,q)(P,D,Q,s) | Univariate | ✅ | ❌ | Non-stationary + seasonal |
| VAR(p) | Multivariate | ❌ | ✅ | Stationary, linear cross-series |
| State Space | Univariate | ✅ | ✅ | Latent state evolution |

---

## 📌 Key Takeaways

1. **Always start with naive baselines** — they are surprisingly hard to beat
2. **ETS often outperforms ARIMA** in practice due to robust automatic parameter selection
3. **AR order p** → read from PACF; **MA order q** → read from ACF
4. **SARIMA** is the industry-standard model for seasonal univariate production forecasting
5. **Residuals must be white noise** — always validate with Ljung-Box before trusting a model
6. **AIC selects model complexity**; use BIC when parsimony is preferred

---

## 📖 Further Reading

- [Forecasting: Principles and Practice (Hyndman) — Ch. 7 (ETS), Ch. 9 (ARIMA)](https://otexts.com/fpp3/)
- [statsmodels SARIMAX Documentation](https://www.statsmodels.org/stable/statespace.html)
- [pmdarima — auto_arima](http://alkaline-ml.com/pmdarima/)
- [Box, Jenkins, Reinsel, Ljung — Time Series Analysis (5th Ed.)](https://www.wiley.com/en-us/Time+Series+Analysis)

---

*← [Module 02 — Data Engineering](../02_data_engineering/README.md) | Back to [Master README](../README.md) | Next: [Module 04 — ML for TS](../04_ml_for_time_series/README.md) →*
