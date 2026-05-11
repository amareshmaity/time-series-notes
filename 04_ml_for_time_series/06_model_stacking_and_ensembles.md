# 06 — Model Stacking & Ensembles

> **Module**: 04 ML for Time Series | **File**: 6 of 6
>
> Ensembling is the single most reliable way to improve forecast accuracy beyond any individual model. This file covers weighted blending, stacking, and ML + statistical hybrid strategies.

---

## Table of Contents

1. [Why Ensembles Work](#1-why-ensembles-work)
2. [Simple Averaging and Weighted Blending](#2-simple-averaging-and-weighted-blending)
3. [Optimal Blend Weight Optimization](#3-optimal-blend-weight-optimization)
4. [Stacking (Meta-Learning)](#4-stacking-meta-learning)
5. [ML + Statistical Hybrid Ensembles](#5-ml--statistical-hybrid-ensembles)
6. [Quantile Ensembles for Prediction Intervals](#6-quantile-ensembles-for-prediction-intervals)
7. [Practical Ensemble Guidelines](#7-practical-ensemble-guidelines)

---

## 1. Why Ensembles Work

### 1.1 The Bias-Variance Tradeoff

```
Individual model error = Bias² + Variance + Irreducible Noise

Ensemble of M uncorrelated models:
  Bias²    ≈ same as individual models
  Variance ≈ Var_individual / M   ← shrinks inversely with ensemble size

→ Ensembles reduce variance without increasing bias
→ The more *diverse* (uncorrelated) the models, the more variance reduction
```

### 1.2 The Diversity Principle

```
Ensemble gain is maximized when models make DIFFERENT errors.

Example:
  Model A: Excellent at trend prediction, poor at holiday effects
  Model B: Excellent at holiday effects, poor at trend
  Ensemble: Captures both → lower overall error than either

Two correlated copies of the same model → almost no ensemble gain
Two fundamentally different models (SARIMA + LightGBM) → high ensemble gain
```

### 1.3 Empirical Evidence

- **M4 Competition (2018)**: 50/50 average of ML + statistical models outperformed either alone
- **M5 Competition (2020)**: Ensemble of LightGBM + statistical baselines won top 5 spots
- **Kaggle TS competitions**: Virtually every winning solution uses ensembles

---

## 2. Simple Averaging and Weighted Blending

### 2.1 Simple Average (Equal Weights)

```python
import numpy as np

# Forecasts from different models (numpy arrays or pandas Series)
fc_lgbm    = model_lgbm.predict(X_test)
fc_rf      = model_rf.predict(X_test)
fc_ridge   = pipe_ridge.predict(X_test)
fc_sarima  = sarima_fit.get_forecast(len(X_test)).predicted_mean.values
fc_ets     = ets_fit.forecast(len(X_test)).values

# Simple average
fc_ensemble_avg = np.mean([fc_lgbm, fc_rf, fc_ridge, fc_sarima, fc_ets], axis=0)

rmse = lambda a, p: np.sqrt(((a - p)**2).mean())
print(f"LightGBM:  RMSE={rmse(y_test, fc_lgbm):.4f}")
print(f"RF:        RMSE={rmse(y_test, fc_rf):.4f}")
print(f"Ridge:     RMSE={rmse(y_test, fc_ridge):.4f}")
print(f"SARIMA:    RMSE={rmse(y_test, fc_sarima):.4f}")
print(f"ETS:       RMSE={rmse(y_test, fc_ets):.4f}")
print(f"Ensemble:  RMSE={rmse(y_test, fc_ensemble_avg):.4f}")
```

### 2.2 When Simple Averaging Works

- Models have similar individual performance
- When you cannot overfit the blending weights (small validation set)
- As a quick, robust baseline ensemble

---

## 3. Optimal Blend Weight Optimization

### 3.1 Non-Negative Least Squares (NNLS)

Find weights `w` that minimize RMSE of the blend on the **validation set**, with non-negativity constraint:

```python
import numpy as np
from scipy.optimize import nnls

# Stack model predictions as columns: shape (n_val, n_models)
# CRITICAL: optimize weights on VALIDATION SET ONLY — not test set
val_preds = np.column_stack([
    fc_lgbm_val,
    fc_rf_val,
    fc_ridge_val,
    fc_sarima_val,
    fc_ets_val,
])
y_val_arr = y_val.values

# Solve: argmin ||w||₂ subject to w ≥ 0
# (NNLS adds non-negativity: no negative weights allowed)
weights, residual = nnls(val_preds, y_val_arr)
weights = weights / weights.sum()   # normalize to sum to 1

print("Optimal blend weights:")
for name, w in zip(["LightGBM", "RF", "Ridge", "SARIMA", "ETS"], weights):
    print(f"  {name:<12}: {w:.4f}")

# Apply to test set
test_preds = np.column_stack([fc_lgbm, fc_rf, fc_ridge, fc_sarima, fc_ets])
fc_nnls = test_preds @ weights
print(f"\nNNLS Blend RMSE: {rmse(y_test, fc_nnls):.4f}")
```

### 3.2 Optuna Weight Optimization

```python
import optuna

def blend_objective(trial):
    # Weight suggestions (Dirichlet-like)
    w_lgbm   = trial.suggest_float("w_lgbm",  0.0, 1.0)
    w_rf     = trial.suggest_float("w_rf",    0.0, 1.0)
    w_ridge  = trial.suggest_float("w_ridge", 0.0, 1.0)
    w_sarima = trial.suggest_float("w_sarima",0.0, 1.0)
    w_ets    = trial.suggest_float("w_ets",   0.0, 1.0)

    total = w_lgbm + w_rf + w_ridge + w_sarima + w_ets
    if total < 1e-8:
        return 1e9

    blend = (w_lgbm*fc_lgbm_val + w_rf*fc_rf_val + w_ridge*fc_ridge_val +
             w_sarima*fc_sarima_val + w_ets*fc_ets_val) / total
    return rmse(y_val, blend)

study = optuna.create_study(direction="minimize")
study.optimize(blend_objective, n_trials=200, show_progress_bar=True)

best_w = study.best_params
total  = sum(best_w.values())
print("\nOptimized weights:")
for k, v in best_w.items():
    print(f"  {k}: {v/total:.4f}")
```

---

## 4. Stacking (Meta-Learning)

### 4.1 How Stacking Works

```
Level 0 (Base Models): Train diverse models on the data
Level 1 (Meta-Model):  Train a model to learn the optimal combination of base model predictions

Step 1: Split train into k folds (time-safe!)
        For each fold, train base models on training folds, predict on validation fold
        → Out-of-fold predictions (same size as train set, no leakage)

Step 2: Train meta-model on out-of-fold predictions → learn to weight base models

Step 3: For test set:
        Re-train base models on full training set → predict on test set
        Meta-model takes base predictions as input → final prediction
```

### 4.2 Time-Safe Stacking Implementation

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

def time_safe_stacking(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    base_models: list,
    meta_model,
    n_splits: int = 5,
    test_size: int = 12,
) -> np.ndarray:
    """
    Time-safe stacking ensemble.
    Returns final predictions on X_test.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    n_base = len(base_models)
    n_train = len(X)

    # Out-of-fold predictions: shape (n_train, n_base)
    oof_preds = np.zeros((n_train, n_base))

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        for model_idx, model in enumerate(base_models):
            model.fit(X_tr, y_tr)
            oof_preds[val_idx, model_idx] = model.predict(X_val)

        print(f"Fold {fold_idx+1}/{n_splits} done")

    # Train meta-model on OOF predictions
    meta_model.fit(oof_preds, y)

    # Generate test predictions from base models (refit on full train)
    test_preds = np.zeros((len(X_test), n_base))
    for model_idx, model in enumerate(base_models):
        model.fit(X, y)
        test_preds[:, model_idx] = model.predict(X_test)

    # Final prediction
    final_pred = meta_model.predict(test_preds)
    return final_pred


# Define base models
base_models = [
    lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1),
    RandomForestRegressor(n_estimators=200, min_samples_leaf=10, n_jobs=-1, random_state=42),
]

# Meta-model: Ridge (prevents overfitting on limited OOF data)
meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10], cv=TimeSeriesSplit(n_splits=3))

# Run stacking
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

meta_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=[0.01, 0.1, 1.0, 10])),
])

y_stack = time_safe_stacking(X_train, y_train, X_test, base_models, meta_pipeline)
print(f"Stacking RMSE: {rmse(y_test, y_stack):.4f}")
```

---

## 5. ML + Statistical Hybrid Ensembles

### 5.1 Strategy 1: Statistical Deseasonalization + ML

```python
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import GradientBoostingRegressor

# Step 1: Decompose with STL
stl = STL(train_series, period=12, robust=True).fit()
seasonal_component = stl.seasonal

# Step 2: Build ML features on deseasonalized residuals
deseason_train = train_series - seasonal_component
# ... build lag/rolling features on deseason_train

# Step 3: Fit ML on deseasonalized series
ml_model = lgb.LGBMRegressor(...)
ml_model.fit(X_deseas_train, deseason_train.iloc[max_lag:])

# Step 4: Forecast
ml_forecast_deseas = ml_model.predict(X_deseas_test)
seasonal_forecast  = seasonal_component[-12:].values   # last season's pattern
final_forecast     = ml_forecast_deseas + seasonal_forecast

print(f"ML + STL Hybrid RMSE: {rmse(y_test, final_forecast):.4f}")
```

### 5.2 Strategy 2: Statistical as ML Feature

```python
# Use SARIMA, ETS, and seasonal naive as ML input features — ML learns to weight them

# 1. Generate in-sample statistical forecasts (1-step-ahead, time-safe)
sarima_insample = sarima_fit.fittedvalues
ets_insample    = ets_fit.fittedvalues
snaive_insample = train_series.shift(12)   # seasonal naive

# 2. Add as features
X_train["sarima_pred"] = sarima_insample.shift(1).reindex(X_train.index)
X_train["ets_pred"]    = ets_insample.shift(1).reindex(X_train.index)
X_train["snaive_pred"] = snaive_insample.reindex(X_train.index)

# 3. Fit ML — it learns to optimally weight and correct statistical forecasts
ml_meta = lgb.LGBMRegressor(...)
ml_meta.fit(X_train.dropna(), y_train.loc[X_train.dropna().index])
```

---

## 6. Quantile Ensembles for Prediction Intervals

For **probabilistic forecasting**, ensemble quantile predictions:

```python
# Quantile predictions from LightGBM
def lgbm_quantile_forecast(X_train, y_train, X_test, quantiles=[0.05, 0.5, 0.95]):
    quantile_forecasts = {}
    for q in quantiles:
        model_q = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,                # target quantile
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            verbose=-1,
        )
        model_q.fit(X_train, y_train)
        quantile_forecasts[f"q{int(q*100):02d}"] = model_q.predict(X_test)
    return pd.DataFrame(quantile_forecasts, index=X_test.index)

pi_df = lgbm_quantile_forecast(X_train, y_train, X_test)
print(pi_df.head())
# q05: lower 90% PI bound
# q50: median forecast
# q95: upper 90% PI bound
```

---

## 7. Practical Ensemble Guidelines

### 7.1 Model Diversity Checklist

Include models from **different families** for maximum diversity:

```
✅ At least one statistical model   (SARIMA or ETS)
✅ At least one gradient boosting   (LightGBM or XGBoost)
✅ At least one simpler model       (Ridge or Random Forest)
✅ At least one naive baseline      (Seasonal naive)

Avoid:
❌ Two LightGBM models with same hyperparameters
❌ Two ARIMA variants that are very similar
```

### 7.2 Ensemble Size Guide

| N models | Expected RMSE improvement |
|---------|--------------------------|
| 2 | 3–8% vs. best single model |
| 5 | 8–15% |
| 10 | 10–20% |
| > 20 | Diminishing returns; add diversity instead |

### 7.3 Anti-Patterns

| Anti-pattern | Problem | Fix |
|-------------|---------|-----|
| Optimize blend weights on test set | Data leakage | Always use validation set for weight tuning |
| Use same model architecture multiple times | Low diversity | Use different model families |
| Stack with complex meta-model | Overfits OOF predictions | Use Ridge or simple average as meta-model |
| Include very poor models | Drags ensemble down | Exclude models worse than seasonal naive |

---

*← [05 — Linear Models](./05_linear_models.md) | [Module README](./README.md) | Next Module: [05 — Deep Learning](../05_deep_learning_models/README.md) →*
