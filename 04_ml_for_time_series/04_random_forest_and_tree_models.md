# 04 — Random Forest & Tree Models

> **Module**: 04 ML for Time Series | **File**: 4 of 6
>
> Random Forest is the robust, lower-variance alternative to gradient boosting. For time series with noisy, high-variance targets, RF often provides more stable forecasts than gradient boosting, especially with limited data.

---

## Table of Contents

1. [How Random Forest Works](#1-how-random-forest-works)
2. [Random Forest vs. Gradient Boosting for TS](#2-random-forest-vs-gradient-boosting-for-ts)
3. [RF Configuration for Time Series](#3-rf-configuration-for-time-series)
4. [Extra Trees (ExtraTreesRegressor)](#4-extra-trees-extratreesregressor)
5. [Feature Importance for TS Interpretation](#5-feature-importance-for-ts-interpretation)
6. [Out-of-Bag Error as CV Approximation](#6-out-of-bag-error-as-cv-approximation)

---

## 1. How Random Forest Works

### 1.1 Algorithm

```
Random Forest = Ensemble of Decision Trees via Bootstrap Aggregation (Bagging)

For m = 1 to M:
  1. Draw bootstrap sample Bₘ (n samples with replacement) from training data
  2. At each node split, consider only a random subset of p features (max_features)
  3. Grow a full, unpruned tree on Bₘ
  
Final prediction: ŷ = (1/M) Σₘ f_Bₘ(x)  (average across all M trees)
```

### 1.2 Why Bagging Reduces Variance

```
Individual tree: High variance (overfits individual samples), low bias
Average of M trees: Variance ≈ σ²/M → shrinks as M increases
                    Bias ≈ same as individual tree

→ RF achieves the bias of a single deep tree at much lower variance
```

### 1.3 How Random Feature Subsets Help

By selecting `max_features` features at each split, trees are **decorrelated** — they make different split decisions, so their errors don't all occur in the same direction.

```
If max_features = 1/3 of all features:
  Each tree uses a different "view" of the data
  Averaging decorrelated trees reduces variance more than averaging correlated trees
```

---

## 2. Random Forest vs. Gradient Boosting for TS

| Aspect | Random Forest | LightGBM/XGBoost |
|--------|--------------|-----------------|
| **Training speed** | Parallelizable, fast | Sequential, slower |
| **Overfitting** | Lower risk (bagging) | Higher risk, needs tuning |
| **High-noise TS** | Often better | Can overfit noise |
| **Low-data regime** | More stable | Less reliable |
| **Peak accuracy** | Typically lower | Higher on large clean datasets |
| **Hyperparameter sensitivity** | Low (robust defaults) | High |
| **Feature importance** | Reliable MDI/permutation | Reliable SHAP |
| **Production stability** | High | Moderate (needs re-tuning) |

**Rule**: If your series is **noisy or short** (< 500 obs per group), try Random Forest first. For large clean datasets, gradient boosting usually wins.

---

## 3. RF Configuration for Time Series

### 3.1 Key Hyperparameters

```python
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(
    n_estimators=500,          # more trees = lower variance, diminishing returns after 200-500
    max_depth=None,            # None = fully grown trees (standard for RF)
    min_samples_leaf=5,        # minimum samples per leaf — KEY for TS
                               # Higher = more regularization
                               # For TS: try 5, 10, 20 (more than default=1)
    max_features=0.33,         # fraction of features per split (classic = 1/3)
                               # try "sqrt", 0.33, 0.5
    max_samples=0.8,           # fraction of data per bootstrap sample
                               # < 1.0 reduces overfitting
    n_jobs=-1,
    random_state=42,
    oob_score=True,            # enable out-of-bag score (free validation estimate)
)
```

### 3.2 Fitting and Forecasting

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Time-safe CV
tscv = TimeSeriesSplit(n_splits=5, test_size=12)
cv_rmses = []

for train_idx, val_idx in tscv.split(X):
    model_rf.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model_rf.predict(X.iloc[val_idx])
    rmse = np.sqrt(((y.iloc[val_idx] - pred)**2).mean())
    cv_rmses.append(rmse)
    
print(f"RF CV RMSE: {np.mean(cv_rmses):.4f} ± {np.std(cv_rmses):.4f}")

# Final model
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
print(f"OOB score: {model_rf.oob_score_:.4f}")   # approximate R² on out-of-bag samples
```

### 3.3 TS-Specific Tuning Guidance

```
min_samples_leaf controls regularization most powerfully in RF for TS:

  min_samples_leaf = 1  → Very deep trees, high variance, risk of memorizing time patterns
  min_samples_leaf = 5  → Good starting point for most TS
  min_samples_leaf = 20 → Strong regularization, better for noisy/short series
  
max_features controls diversity (decorrelation):
  "sqrt"    → √n_features — good for many features
  0.33      → 1/3 of features — classic recommendation
  0.5       → 1/2 of features — good when features are correlated (lag features often are)
```

---

## 4. Extra Trees (ExtraTreesRegressor)

### 4.1 Difference from Random Forest

ExtraTrees (Extremely Randomized Trees) adds **randomness at the split level**:
- RF: finds the *best* threshold for each random feature subset
- ExtraTrees: uses a *random* threshold for each random feature subset

```
ExtraTrees split decision:
  1. Select random subset of features (same as RF)
  2. For each selected feature, pick a RANDOM threshold
  3. Choose the best split among these random (feature, threshold) pairs
```

### 4.2 When ExtraTrees Outperforms RF

- Very high-dimensional feature spaces (many Fourier terms, one-hot encodings)
- When training speed is important (random thresholds → no optimization → faster)
- When variance needs to be further reduced (more randomness → lower variance)

```python
from sklearn.ensemble import ExtraTreesRegressor

model_et = ExtraTreesRegressor(
    n_estimators=500,
    min_samples_leaf=5,
    max_features=0.33,
    n_jobs=-1,
    random_state=42,
)
model_et.fit(X_train, y_train)
y_pred_et = model_et.predict(X_test)
```

### 4.3 RF vs. ExtraTrees Summary

| | Random Forest | ExtraTrees |
|--|--------------|-----------|
| Threshold selection | Optimal (best of random subset) | Random |
| Training speed | Moderate | Faster |
| Bias | Lower | Slightly higher |
| Variance | Low | Even lower |
| When to prefer | General purpose | Very noisy TS, speed-critical |

---

## 5. Feature Importance for TS Interpretation

### 5.1 Mean Decrease in Impurity (MDI)

Built-in feature importance — measures average reduction in impurity (Gini/MSE) when a feature is used for a split:

```python
import pandas as pd
import matplotlib.pyplot as plt

importances = pd.Series(
    model_rf.feature_importances_,
    index=X_train.columns,
).sort_values(ascending=False)

# Plot top 20 features
fig, ax = plt.subplots(figsize=(10, 7))
importances[:20].plot.barh(ax=ax, color="#2C7BB6", edgecolor="white")
ax.set_title("RF Feature Importance (MDI) — Top 20")
ax.set_xlabel("Mean Decrease in Impurity")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
```

**Limitation**: MDI is biased toward high-cardinality features. Prefer permutation importance or SHAP.

### 5.2 Permutation Importance

More reliable: measure how much the test RMSE increases when a feature is randomly shuffled:

```python
from sklearn.inspection import permutation_importance

perm_result = permutation_importance(
    model_rf,
    X_test, y_test,
    n_repeats=10,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)

perm_importances = pd.DataFrame({
    "feature":   X_test.columns,
    "importance": perm_result.importances_mean,
    "std":        perm_result.importances_std,
}).sort_values("importance", ascending=False)

print("Top features by permutation importance:")
print(perm_importances.head(10))
```

### 5.3 What to Look For in TS Feature Importance

| High-importance feature | Meaning |
|------------------------|---------|
| `lag_1` dominates | Strong autocorrelation — AR(1) effect present |
| `lag_7` > `lag_1` | Weekly pattern is the dominant signal |
| `roll30_mean` very important | Long-run mean level is the best predictor |
| `month_sin/cos` top features | Strong yearly seasonality |
| `is_weekend` high rank | Weekend effect is significant |
| External feature (price) high rank | Price elasticity drives demand |

---

## 6. Out-of-Bag Error as CV Approximation

### 6.1 OOB Explained

Each tree is trained on a bootstrap sample that leaves out ~37% of observations. These "out-of-bag" samples act as a built-in validation set — giving a free approximation of generalization error.

```
For each sample i:
  OOB prediction ŷᵢ_oob = average prediction of trees that did NOT train on i
  OOB error = mean loss across all ŷᵢ_oob vs. yᵢ
```

### 6.2 Using OOB for Quick Hyperparameter Search

```python
# OOB score as a cheap proxy for CV (no need to split data)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

best_rmse = np.inf
best_params = {}

for n_est in [100, 200, 500]:
    for min_leaf in [1, 5, 10, 20]:
        for max_feat in [0.33, 0.5, "sqrt"]:
            m = RandomForestRegressor(
                n_estimators=n_est,
                min_samples_leaf=min_leaf,
                max_features=max_feat,
                oob_score=True,
                n_jobs=-1, random_state=42,
            )
            m.fit(X_train, y_train)
            
            # OOB predictions (generated automatically during fit)
            oob_rmse = np.sqrt(mean_squared_error(y_train, m.oob_prediction_))
            
            if oob_rmse < best_rmse:
                best_rmse = oob_rmse
                best_params = {"n_estimators": n_est, "min_samples_leaf": min_leaf, "max_features": max_feat}

print(f"Best OOB RMSE: {best_rmse:.4f}")
print(f"Best params:  {best_params}")
```

> **Warning**: OOB error underestimates true test error for time series because bootstrap sampling does NOT respect temporal order — past and future samples can appear in the same tree's training set. Use proper walk-forward CV for final evaluation, but OOB is fine for fast hyperparameter screening.

---

*← [03 — XGBoost & LightGBM](./03_gradient_boosting_xgboost_lgbm.md) | [Module README](./README.md) | Next: [05 — Linear Models](./05_linear_models.md) →*
