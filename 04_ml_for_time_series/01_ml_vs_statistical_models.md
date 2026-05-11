# 01 — ML vs. Statistical Models for Time Series

> **Module**: 04 ML for Time Series | **File**: 1 of 6
>
> Knowing *when* to use ML and when to use classical statistical models is the most important judgment call in a time series project. Neither approach dominates — the right choice depends on data size, seasonality complexity, external regressors, and forecast horizon.

---

## Table of Contents

1. [How ML Models Learn Time Structure](#1-how-ml-models-learn-time-structure)
2. [Strengths of ML Models](#2-strengths-of-ml-models)
3. [Strengths of Statistical Models](#3-strengths-of-statistical-models)
4. [Decision Framework](#4-decision-framework)
5. [The Global vs. Local Model Distinction](#5-the-global-vs-local-model-distinction)
6. [Hybrid Strategies](#6-hybrid-strategies)

---

## 1. How ML Models Learn Time Structure

Statistical models (ARIMA, ETS) **explicitly model time dynamics** — their equations are built around lags, differences, and autocorrelations.

ML models (XGBoost, LightGBM) **have no inherent temporal awareness** — they treat every row as an independent sample. The temporal structure must be encoded into features.

```
Statistical model:
  y(t) = φ₁·y(t-1) + φ₂·y(t-2) + ... + ε(t)
  → Temporal dynamics are built INTO the model equation

ML model:
  y(t) = f(lag_1, lag_7, roll7_mean, month_sin, is_weekend, ...)
  → Temporal dynamics are encoded AS FEATURES
  → f is a tree ensemble — any nonlinear function of those features
```

This means: **the quality of an ML forecast is almost entirely determined by the quality of feature engineering.**

---

## 2. Strengths of ML Models

### 2.1 Non-linear Relationships

ML models capture complex, non-linear interactions that statistical models miss:

```
Example: Sales spike when:
  - It's a weekend AND there's a promotion AND it's December
  
ARIMA: Cannot capture this 3-way interaction
XGBoost: Naturally learns it via tree splits
```

### 2.2 Many External Regressors

ML models scale gracefully with hundreds of features — price, weather, promotions, competitor data, macroeconomic indicators:

```python
features = [
    "lag_1", "lag_7", "lag_28",          # past values
    "roll7_mean", "roll30_mean",          # rolling stats
    "price", "promo_flag", "temperature", # external
    "month_sin", "month_cos",             # calendar
    "fourier_weekly_sin_1", ...,          # Fourier
    "store_id_encoded",                   # entity embedding
]
```

ARIMAX handles regressors but is limited to linear relationships with small numbers of covariates.

### 2.3 Multiple Related Series (Global Models)

A single XGBoost model can be trained across **thousands of related series** simultaneously (global forecasting):

```python
# One model for all 1000 stores
# Features include: store_id_encoded, category_encoded, + all time features
# The model learns: "store A in category X in December → sale pattern Y"
```

SARIMA must be fitted independently for each series — no information sharing.

### 2.4 Robustness to Outliers

With appropriate hyperparameters (e.g., Huber loss, MAE loss), gradient boosting is more robust to outliers than ARIMA models which are sensitive to model specification errors.

### 2.5 No Stationarity Requirement

ML models work on non-stationary series as long as the **non-stationarity is captured in features** (e.g., trend feature = `rolling_mean_365`, yoy features, expanding mean).

---

## 3. Strengths of Statistical Models

### 3.1 Small Sample Sizes

ARIMA/ETS work well with as few as 30–50 observations. XGBoost typically needs hundreds of training examples per unique pattern to generalize.

```
Rule of thumb:
  < 100 observations → Statistical models
  100–500           → Both are viable, compare on CV
  > 500             → ML models generally win
```

### 3.2 Exact Prediction Intervals

ARIMA provides **exact analytical prediction intervals** based on the model's probability distribution. ML prediction intervals require additional machinery (conformal prediction, quantile regression).

### 3.3 Interpretability and Trust

In regulated industries (finance, healthcare), ARIMA/ETS models are more easily audited and explained:

```
"Sales are forecast to grow because the exponential smoothing detects a 
0.8% monthly upward trend and the seasonal factor for December is 1.35"

vs.

"The XGBoost model's 523rd tree splits on lag_7 < 450..."
```

### 3.4 Short Series, Many Series

For thousands of independent short series (e.g., retail SKU forecasting), the **M4/M5 competition results** show ETS and ARIMA competitive with or superior to ML when series are short.

### 3.5 Clear Seasonal Patterns

When seasonality is stable, periodic, and well-behaved, SARIMA/ETS often models it as accurately as ML — with far fewer hyperparameters.

---

## 4. Decision Framework

```
Is the series length < 100 observations?
├── YES → Use statistical models (ETS, SARIMA)
└── NO ↓

Do you have many external regressors (>5)?
├── YES → Use ML (XGBoost, LightGBM)
└── NO ↓

Do you need to forecast thousands of related series?
├── YES → Use global ML model
└── NO ↓

Is the relationship between features and target non-linear?
├── YES → Use ML
└── NO ↓ (linear relationships)

Are you confident in model stationarity?
├── YES → Use SARIMA
└── NO → Use ML (more robust to misspecification)
```

### Summary Table

| Scenario | Best Approach |
|----------|-------------|
| < 100 observations, single series | ETS or SARIMA |
| Many external regressors | XGBoost / LightGBM |
| Thousands of related series | Global ML model |
| Stable, clear seasonality, univariate | SARIMA / ETS |
| Competition / kaggle forecasting | LightGBM + features |
| Need exact prediction intervals | SARIMA / ETS |
| Production with complex feature space | LightGBM + SHAP |
| Short horizon (1-3 steps) | Statistical often wins |
| Long horizon (12+ steps) | ML (MIMO forecasting) |

---

## 5. The Global vs. Local Model Distinction

### 5.1 Local Models

Fit **one model per series**. Every series is treated independently:

```python
# Local ARIMA: one fit per series
for store_id in stores:
    model = SARIMA(df[df.store == store_id]["sales"], ...)
    models[store_id] = model.fit()
```

**Pros**: Adapts precisely to each series' history  
**Cons**: No information sharing between series; fails with short/new series

### 5.2 Global Models

Fit **one model across all series**. The model learns shared patterns:

```python
# Global LightGBM: all stores in one dataset
df["store_id_encoded"] = label_encode(df["store_id"])
# Features: lag_1, lag_7, roll30_mean, store_encoded, category_encoded, ...
model = LGBMRegressor()
model.fit(X_train, y_train)   # X_train spans all stores
```

**Pros**: Information sharing across series; handles new/short series; one model to maintain  
**Cons**: May underfit peculiarities of individual series

### 5.3 When Global Models Win

- M5 Forecasting Competition (Walmart sales): Global LightGBM dominated
- Many stores/products with shared seasonal patterns
- New series with insufficient history for local models

---

## 6. Hybrid Strategies

### 6.1 Statistical Deseasonalization + ML Residuals

```
Step 1: Fit STL to remove trend + seasonality
Step 2: Fit ML model on the deseasonalized residuals
Step 3: Forecast = ML forecast (deseasonalized) + STL seasonal + linear trend

→ ML handles the non-seasonal non-linear patterns
→ Statistical handles the seasonal structure cleanly
```

### 6.2 Statistical Forecasts as ML Features

```python
# Use ARIMA, ETS, seasonal naive forecasts as features for the ML model
df["arima_forecast"]  = arima_fitted.fittedvalues.shift(1)
df["ets_forecast"]    = ets_fitted.fittedvalues.shift(1)
df["snaive_forecast"] = series.shift(12)

# ML model learns to weight these statistical forecasts
model = LGBMRegressor()
model.fit(X_with_stat_features, y)
```

### 6.3 Blending

Simple weighted average of statistical and ML forecasts — often surprisingly effective:

```python
alpha = 0.4   # tune on validation set
final_forecast = alpha * sarima_forecast + (1 - alpha) * lgbm_forecast
```

---

*← [Module README](./README.md) | Next: [02 — Time Series CV](./02_timeseries_crossvalidation.md) →*
