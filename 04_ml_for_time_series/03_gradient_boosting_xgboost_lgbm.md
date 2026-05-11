# 03 — Gradient Boosting: XGBoost & LightGBM

> **Module**: 04 ML for Time Series | **File**: 3 of 6
>
> Gradient boosting models dominate time series forecasting competitions. This file covers the internals, time series-specific configuration, and a complete production pipeline.

---

## Table of Contents

1. [Gradient Boosting Intuition](#1-gradient-boosting-intuition)
2. [XGBoost for Time Series](#2-xgboost-for-time-series)
3. [LightGBM for Time Series](#3-lightgbm-for-time-series)
4. [XGBoost vs. LightGBM](#4-xgboost-vs-lightgbm)
5. [Key Hyperparameters for TS](#5-key-hyperparameters-for-ts)
6. [Multi-Step Forecasting Strategies](#6-multi-step-forecasting-strategies)
7. [SHAP — Interpreting Feature Importance](#7-shap--interpreting-feature-importance)
8. [Production Pipeline](#8-production-pipeline)

---

## 1. Gradient Boosting Intuition

### 1.1 The Core Algorithm

Gradient boosting builds an **ensemble of decision trees** sequentially, where each new tree corrects the residuals of the previous ensemble:

```
Algorithm:
  F₀(x) = initial prediction (e.g., mean of y)
  
  For m = 1, 2, ..., M:
    1. Compute pseudo-residuals:  rᵢ = -∂L/∂F(xᵢ)
       (gradient of loss w.r.t. current prediction)
    2. Fit a decision tree hₘ to the residuals
    3. Update: Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)
       Where η = learning rate (shrinkage)

Final forecast: F_M(x) = Σₘ₌₁ᴹ η · hₘ(x)
```

### 1.2 Why Trees Work for TS Features

Time series lag features create **threshold-based patterns**:
- "If lag_1 > 500 AND is_weekend = 1 AND month = 12 → high sales"

Decision trees naturally implement these threshold conditions as splits. Gradient boosting learns thousands of such conditions simultaneously.

### 1.3 Loss Functions for TS

| Loss Function | When to Use |
|-------------|------------|
| `reg:squarederror` (MSE) | Standard, sensitive to outliers |
| `reg:absoluteerror` (MAE) | Robust to outliers |
| `reg:quantileerror` (Quantile) | Probabilistic forecasting |
| `reg:tweedie` | Count data, zero-inflated demand |
| `reg:gamma` | Non-negative, right-skewed |

---

## 2. XGBoost for Time Series

### 2.1 Key XGBoost Concepts

```
Regularization:
  Objective = Σ L(yᵢ, ŷᵢ) + Σₜ Ω(fₜ)
  
  Where Ω(f) = γ · T + (λ/2) · Σⱼ wⱼ²
    T = number of leaves
    γ = minimum gain required to split (tree complexity penalty)
    λ = L2 regularization on leaf weights
    α = L1 regularization (sparsity)

→ XGBoost penalizes both tree complexity and leaf weight magnitudes
```

### 2.2 XGBoost Implementation

```python
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Feature matrix and target (leakage-free — see Module 02)
# X.columns: [lag_1, lag_7, lag_28, roll7_mean, roll30_mean, ..., month_sin, ...]
# y: target sales

# Time-series train/test split
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# XGBoost with time series tuned parameters
model_xgb = xgb.XGBRegressor(
    n_estimators=1000,       # number of trees (use early stopping)
    learning_rate=0.05,      # small LR → more trees, better generalization
    max_depth=6,             # tree depth (3-8 for TS)
    min_child_weight=5,      # min samples in leaf (prevents over-splitting)
    subsample=0.8,           # row subsampling (reduces overfitting)
    colsample_bytree=0.8,    # feature subsampling per tree
    gamma=0.1,               # min split gain (regularization)
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",      # fast histogram method
    random_state=42,
    n_jobs=-1,
)

model_xgb.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100,
    early_stopping_rounds=50,   # stop if no improvement for 50 rounds
)

print(f"Best iteration: {model_xgb.best_iteration}")
print(f"Best RMSE:      {model_xgb.best_score:.4f}")

y_pred_xgb = model_xgb.predict(X_test)
```

### 2.3 DMatrix for Performance

```python
# Use DMatrix for maximum speed
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)

params = {
    "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "tree_method": "hist",
}
bst = xgb.train(
    params, dtrain,
    num_boost_round=1000,
    evals=[(dtrain, "train"), (dtest, "eval")],
    early_stopping_rounds=50,
    verbose_eval=100,
)
```

---

## 3. LightGBM for Time Series

### 3.1 LightGBM Advantages Over XGBoost

| Feature | XGBoost | LightGBM |
|---------|---------|----------|
| **Training speed** | Moderate | Very fast (Gradient-based One-Side Sampling) |
| **Memory usage** | High | Low |
| **Large datasets** | Good | Excellent |
| **Categorical features** | Manual encode | Native support |
| **Leaf-wise growth** | Level-wise | Leaf-wise (deeper, more expressive) |
| **GPU support** | ✅ | ✅ |
| **Default accuracy** | Slightly lower | Slightly higher on many TS tasks |

### 3.2 LightGBM Implementation

```python
import lightgbm as lgb

model_lgbm = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,               # -1 = unlimited (leaf-wise growth is bounded by num_leaves)
    num_leaves=63,              # 2^max_depth - 1 as starting point
    min_child_samples=20,       # min samples per leaf (key for TS — set higher than default)
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="regression",     # MAE: "regression_l1", MSE: "regression"
    metric="rmse",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=100),
]

model_lgbm.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=callbacks,
)

y_pred_lgbm = model_lgbm.predict(X_test)
```

### 3.3 Native Categorical Feature Handling

```python
# LightGBM handles categoricals natively (no one-hot encoding needed)
categorical_features = ["store_id", "category", "day_of_week"]

for col in categorical_features:
    X_train[col] = X_train[col].astype("category")
    X_test[col]  = X_test[col].astype("category")

model_lgbm.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    categorical_feature=categorical_features,
    callbacks=callbacks,
)
```

---

## 4. XGBoost vs. LightGBM

| Aspect | XGBoost | LightGBM |
|--------|---------|----------|
| **Speed** | Slower | Much faster (3-10×) |
| **Memory** | More | Less |
| **Large datasets** | Good | Better |
| **Overfitting** | Level-wise = less overfitting | Leaf-wise = more overfitting risk |
| **Tuning** | `max_depth` primary | `num_leaves` primary |
| **Competition results** | Strong | Slightly edge in TS competitions |
| **Categorical support** | Manual encoding | Native |

**Rule of thumb**: Start with LightGBM for speed. Validate with XGBoost for comparison. If dataset < 50K rows, both are fast enough.

---

## 5. Key Hyperparameters for TS

### 5.1 LightGBM TS-Tuned Starting Point

```python
lgb_params_ts = {
    # Trees
    "n_estimators":      500,     # start with 500; use early stopping to find best
    "learning_rate":     0.05,    # 0.01–0.1 (lower = more trees needed = slower)
    "num_leaves":        31,      # 2^(max_depth-1) rule; 31 is safe default
    "max_depth":         -1,      # controlled by num_leaves when -1

    # Regularization (critical for TS to prevent overfitting)
    "min_child_samples": 20,      # ↑ = more regularization; try 10–100
    "min_child_weight":  0.001,
    "reg_alpha":         0.1,     # L1
    "reg_lambda":        0.1,     # L2

    # Subsampling
    "subsample":         0.8,     # row subsampling per tree
    "subsample_freq":    1,       # apply subsample every tree
    "colsample_bytree":  0.8,     # feature subsampling per tree

    # Loss
    "objective":         "regression",   # "regression_l1" for MAE
    "metric":            "rmse",
}
```

### 5.2 XGBoost TS-Tuned Starting Point

```python
xgb_params_ts = {
    "n_estimators":      500,
    "learning_rate":     0.05,
    "max_depth":         5,        # keep shallow for TS — try 3, 5, 7
    "min_child_weight":  5,        # ↑ = more regularization
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "gamma":             0.1,      # min gain to split
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "objective":         "reg:squarederror",
    "eval_metric":       "rmse",
    "tree_method":       "hist",
}
```

---

## 6. Multi-Step Forecasting Strategies

### 6.1 Recursive (RMSN) Strategy

Forecast one step at a time, feeding predictions back as lag features:

```python
def recursive_forecast(model, X_last: pd.Series, h: int, feature_names: list) -> np.ndarray:
    """Recursive multi-step forecast — uses own predictions as lag features."""
    predictions = []
    current_features = X_last.copy()

    for step in range(h):
        pred = model.predict(current_features.values.reshape(1, -1))[0]
        predictions.append(pred)

        # Shift lag features: lag_2 ← lag_1, lag_7 ← lag_6, etc.
        if "lag_1" in feature_names:
            current_features["lag_2"] = current_features["lag_1"]
            current_features["lag_1"] = pred
        # (update all lag features accordingly)

    return np.array(predictions)
```

**Pros**: Single model, simplest approach  
**Cons**: Errors accumulate — each step's error is an input to the next

### 6.2 Direct (MIMO) Strategy

Train a **separate model for each horizon**:

```python
# Train one model per forecast step
direct_models = {}
for h_step in range(1, 13):   # 12-step ahead
    y_shifted = y.shift(-h_step)   # target = y(t + h_step)
    df_h = X.copy()
    df_h["target"] = y_shifted
    df_h = df_h.dropna()

    X_h = df_h.drop(columns=["target"])
    y_h = df_h["target"]
    
    split = int(len(X_h) * 0.8)
    model_h = lgb.LGBMRegressor(**lgb_params_ts)
    model_h.fit(X_h.iloc[:split], y_h.iloc[:split], ...)
    direct_models[h_step] = model_h

# Forecast
final_forecast = np.array([direct_models[h].predict(X_test.iloc[-1:]) for h in range(1, 13)])
```

**Pros**: No error accumulation; each model optimized for its horizon  
**Cons**: Need M models; features at future horizons must not use future values of target

### 6.3 MIMO (Multi-Output)

Single model predicts all `h` steps simultaneously using a multi-output regressor:

```python
from sklearn.multioutput import MultiOutputRegressor

# Target: next 12 periods stacked as a 12-column matrix
y_mimo = pd.DataFrame({f"t+{h}": y.shift(-h) for h in range(1, 13)}).dropna()
X_mimo = X.loc[y_mimo.index]

multi_model = MultiOutputRegressor(lgb.LGBMRegressor(**lgb_params_ts))
multi_model.fit(X_mimo.iloc[:split], y_mimo.iloc[:split])
forecast_12step = multi_model.predict(X_mimo.iloc[-1:]).flatten()
```

---

## 7. SHAP — Interpreting Feature Importance

### 7.1 Why SHAP Over Standard Feature Importance?

Standard feature importance (split count, gain) is biased toward high-cardinality features. **SHAP (SHapley Additive exPlanations)** provides:
- Consistent, unbiased feature attribution
- Per-prediction explanations (not just global)
- Interaction effects

### 7.2 SHAP for Time Series Interpretation

```python
import shap
import matplotlib.pyplot as plt

# Compute SHAP values
explainer = shap.TreeExplainer(model_lgbm)
shap_values = explainer.shap_values(X_test)

# Global feature importance (bar plot)
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=20)

# Beeswarm plot (shows distribution of impact)
shap.summary_plot(shap_values, X_test, max_display=20)

# Single prediction explanation
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=X_test.columns.tolist()
    )
)
```

### 7.3 Reading SHAP for TS Features

| SHAP Finding | Interpretation |
|-------------|----------------|
| `lag_1` has highest SHAP | Yesterday's value is the dominant predictor |
| `lag_7` SHAP > `lag_1` | Weekly pattern stronger than daily persistence |
| `month_sin` high SHAP | Yearly seasonality is a key driver |
| `roll30_mean` positive SHAP | Long-run trend level matters |
| `is_weekend` positive SHAP | Sales spike on weekends |

---

## 8. Production Pipeline

```python
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

class LGBMTimeSeriesForecaster:
    """
    Production-grade LightGBM time series forecasting pipeline.
    Handles: feature engineering, time-safe CV, training, forecasting.
    """

    def __init__(self, lags=None, rolling_windows=None, n_cv_splits=5, horizon=12):
        self.lags            = lags or [1, 7, 14, 28, 365]
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.n_cv_splits     = n_cv_splits
        self.horizon         = horizon
        self.model           = None
        self.feature_names_  = None

    def _build_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()
        t = target_col
        for lag in self.lags:
            df[f"lag_{lag}"] = df[t].shift(lag)
        for w in self.rolling_windows:
            s = df[t].shift(1)
            df[f"roll{w}_mean"] = s.rolling(w, min_periods=1).mean()
            df[f"roll{w}_std"]  = s.rolling(w, min_periods=1).std()
        idx = df.index
        df["month_sin"]  = np.sin(2 * np.pi * idx.month / 12)
        df["month_cos"]  = np.cos(2 * np.pi * idx.month / 12)
        df["dow_sin"]    = np.sin(2 * np.pi * idx.dayofweek / 7)
        df["dow_cos"]    = np.cos(2 * np.pi * idx.dayofweek / 7)
        df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
        return df.dropna()

    def fit(self, df: pd.DataFrame, target_col: str):
        df_feat = self._build_features(df, target_col)
        X = df_feat.drop(columns=[target_col])
        y = df_feat[target_col]
        self.feature_names_ = X.columns.tolist()

        # Time-safe CV for early stopping
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits, test_size=self.horizon)
        val_rmses = []

        for train_idx, val_idx in tscv.split(X):
            m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                   num_leaves=31, min_child_samples=20,
                                   subsample=0.8, colsample_bytree=0.8,
                                   reg_alpha=0.1, reg_lambda=0.1,
                                   objective="regression", verbose=-1)
            m.fit(X.iloc[train_idx], y.iloc[train_idx],
                  eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            val_rmses.append(np.sqrt(((y.iloc[val_idx] - m.predict(X.iloc[val_idx]))**2).mean()))

        print(f"CV RMSE: {np.mean(val_rmses):.4f} ± {np.std(val_rmses):.4f}")

        # Refit on full data with optimal n_estimators
        n_best = m.best_iteration_
        self.model = lgb.LGBMRegressor(n_estimators=n_best, learning_rate=0.05,
                                        num_leaves=31, min_child_samples=20,
                                        subsample=0.8, colsample_bytree=0.8,
                                        reg_alpha=0.1, reg_lambda=0.1,
                                        objective="regression", verbose=-1)
        self.model.fit(X, y)
        return self

    def predict(self, X_future: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "Call .fit() first"
        return self.model.predict(X_future[self.feature_names_])


# Usage
forecaster = LGBMTimeSeriesForecaster(lags=[1, 7, 14, 28], horizon=12)
forecaster.fit(train_df, target_col="sales")
predictions = forecaster.predict(test_features)
```

---

*← [02 — Time Series CV](./02_timeseries_crossvalidation.md) | [Module README](./README.md) | Next: [04 — Random Forest](./04_random_forest_and_tree_models.md) →*
