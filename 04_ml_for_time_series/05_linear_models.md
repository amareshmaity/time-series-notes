# 05 — Linear Models for Time Series

> **Module**: 04 ML for Time Series | **File**: 5 of 6
>
> Regularized linear models are fast, interpretable, and surprisingly competitive on many time series tasks — especially when combined with rich feature engineering (Fourier terms, lags, and interactions).

---

## Table of Contents

1. [Why Linear Models for TS?](#1-why-linear-models-for-ts)
2. [Ridge Regression (L2)](#2-ridge-regression-l2)
3. [Lasso Regression (L1)](#3-lasso-regression-l1)
4. [ElasticNet](#4-elasticnet)
5. [Bayesian Ridge Regression](#5-bayesian-ridge-regression)
6. [Feature Scaling for Linear TS Models](#6-feature-scaling-for-linear-ts-models)
7. [Linear Models with Fourier Features (Regression with Regressors)](#7-linear-models-with-fourier-features-regression-with-regressors)
8. [When Linear Models Win](#8-when-linear-models-win)
9. [Conformal Prediction Intervals for Any ML Model](#9-conformal-prediction-intervals-for-any-ml-model)

---

## 1. Why Linear Models for TS?

### 1.1 Advantages

| Advantage | Details |
|-----------|---------|
| **Speed** | Train in milliseconds — ideal for many series, rapid prototyping |
| **Interpretability** | Coefficients directly readable — which features matter and by how much |
| **Reliable with small data** | Works well with 50–200 observations |
| **Fourier + Ridge = powerful** | Captures any seasonality shape with sine/cosine features |
| **Stable predictions** | Less risk of extreme out-of-distribution predictions |

### 1.2 The Linear TS Model Idea

```
ŷ(t) = β₀ + β₁·lag_1 + β₂·lag_7 + β₃·roll30_mean
       + β₄·sin(2πt/365) + β₅·cos(2πt/365)   ← yearly seasonality
       + β₆·sin(2πt/7)   + β₇·cos(2πt/7)     ← weekly seasonality
       + β₈·price + β₉·is_promo + ...

This is ARIMAX written as a regression — with explicit seasonal Fourier terms
and lag features acting as AR terms.
```

### 1.3 Linear Model as a Baseline

Always fit Ridge as a **quick baseline** before trying trees:

```python
# 60-second baseline pipeline
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
pipe.fit(X_train, y_train)
rmse_ridge = np.sqrt(((y_test - pipe.predict(X_test))**2).mean())
print(f"Ridge baseline RMSE: {rmse_ridge:.4f}")
# → beat this with LightGBM or reject the extra complexity
```

---

## 2. Ridge Regression (L2)

### 2.1 Objective Function

```
Ridge minimizes:
  ||y - Xβ||² + α · ||β||²

Where:
  ||y - Xβ||² = sum of squared errors (fit quality)
  α · ||β||²  = L2 penalty (shrinks all coefficients toward zero)
  α           = regularization strength (larger = more shrinkage)

Closed-form solution: β̂ = (XᵀX + αI)⁻¹Xᵀy
```

### 2.2 When to Use Ridge

- When all features are expected to contribute (no sparse solution needed)
- Correlated features (lag features are highly correlated — Ridge handles this)
- Need stable coefficient estimates

### 2.3 Implementation

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# RidgeCV: automatic α selection via cross-validation
tscv = TimeSeriesSplit(n_splits=5, test_size=12)

pipe_ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(
        alphas=[0.01, 0.1, 1.0, 10, 100, 1000],
        cv=tscv,
        scoring="neg_root_mean_squared_error",
    )),
])

pipe_ridge.fit(X_train, y_train)
best_alpha = pipe_ridge.named_steps["ridge"].alpha_
print(f"Best α (Ridge): {best_alpha:.4f}")

y_pred_ridge = pipe_ridge.predict(X_test)

# View coefficients
coef_df = pd.DataFrame({
    "feature":     X_train.columns,
    "coefficient": pipe_ridge.named_steps["ridge"].coef_,
}).sort_values("coefficient", key=abs, ascending=False)
print(coef_df.head(10))
```

---

## 3. Lasso Regression (L1)

### 3.1 Objective Function

```
Lasso minimizes:
  ||y - Xβ||² + α · ||β||₁

Where ||β||₁ = Σⱼ |βⱼ|  (L1 penalty — promotes sparsity)

Key property: Lasso drives many coefficients EXACTLY to zero → built-in feature selection
```

### 3.2 When to Use Lasso for TS

- You have many candidate features but suspect only a few matter
- Automatic feature selection is desired
- You are building lag features for a large range of lags (1–365) and want automatic selection

```python
from sklearn.linear_model import LassoCV

pipe_lasso = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LassoCV(
        alphas=[0.0001, 0.001, 0.01, 0.1, 1.0],
        cv=tscv,
        max_iter=10000,
        n_jobs=-1,
    )),
])
pipe_lasso.fit(X_train, y_train)
best_alpha = pipe_lasso.named_steps["lasso"].alpha_
print(f"Best α (Lasso): {best_alpha:.6f}")

# Features with non-zero coefficients (selected by Lasso)
lasso_coefs = pd.Series(
    pipe_lasso.named_steps["lasso"].coef_,
    index=X_train.columns,
)
selected = lasso_coefs[lasso_coefs != 0].sort_values(key=abs, ascending=False)
print(f"\nLasso selected {len(selected)} features out of {len(X_train.columns)}:")
print(selected.head(15))
```

### 3.3 Lasso for Lag Selection

```python
# Fit Lasso with many lags (1..60) and let it select automatically
all_lags = list(range(1, 61))
df_lasso = df.copy()
for lag in all_lags:
    df_lasso[f"lag_{lag}"] = df_lasso["sales"].shift(lag)
df_lasso = df_lasso.dropna()

X_l = df_lasso[[f"lag_{i}" for i in all_lags]]
y_l = df_lasso["sales"]

pipe_lag_sel = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LassoCV(cv=tscv, max_iter=10000)),
])
pipe_lag_sel.fit(X_l.iloc[:split], y_l.iloc[:split])

selected_lags = [i for i, c in zip(all_lags, pipe_lag_sel.named_steps["lasso"].coef_) if c != 0]
print(f"Lasso-selected significant lags: {selected_lags}")
```

---

## 4. ElasticNet

### 4.1 Objective Function

```
ElasticNet combines both L1 and L2 penalties:
  ||y - Xβ||² + α·l1_ratio·||β||₁ + α·(1-l1_ratio)/2·||β||²

Where:
  l1_ratio = 1     → pure Lasso
  l1_ratio = 0     → pure Ridge
  0 < l1_ratio < 1 → ElasticNet (sparse solution + stable with correlated features)
```

### 4.2 When ElasticNet Wins

- Features are highly correlated (lag features) AND you want sparsity
- Lasso struggles to select one from a group of correlated lags (ElasticNet selects the group)

```python
from sklearn.linear_model import ElasticNetCV

pipe_enet = Pipeline([
    ("scaler", StandardScaler()),
    ("enet", ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],   # range to search
        alphas=[0.001, 0.01, 0.1, 1.0],
        cv=tscv,
        max_iter=10000,
        n_jobs=-1,
    )),
])
pipe_enet.fit(X_train, y_train)
print(f"Best α={pipe_enet.named_steps['enet'].alpha_:.4f}, l1_ratio={pipe_enet.named_steps['enet'].l1_ratio_:.2f}")
```

---

## 5. Bayesian Ridge Regression

### 5.1 Why Bayesian for TS

Bayesian Ridge provides:
- **Automatic regularization** — α is inferred from data, no grid search needed
- **Uncertainty estimates** — returns prediction mean AND variance
- Works well when you have limited data

```python
from sklearn.linear_model import BayesianRidge

model_br = BayesianRidge(
    max_iter=300,
    tol=1e-3,
    alpha_1=1e-6,   # Gamma prior hyperparameters for noise precision α
    alpha_2=1e-6,
    lambda_1=1e-6,  # Gamma prior hyperparameters for weight precision λ
    lambda_2=1e-6,
    compute_score=True,
)

# Fit on scaled features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model_br.fit(X_train_scaled, y_train)

# Prediction with uncertainty
y_mean, y_std = model_br.predict(X_test_scaled, return_std=True)
print(f"First prediction: {y_mean[0]:.2f} ± {y_std[0]*1.96:.2f} (95% PI)")

# Plot with prediction intervals
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(y_test.values, color="black", label="Actual")
ax.plot(y_mean, color="#D7191C", linestyle="--", label="Bayesian Ridge")
ax.fill_between(range(len(y_mean)),
                y_mean - 1.96 * y_std,
                y_mean + 1.96 * y_std,
                alpha=0.2, color="#D7191C", label="95% PI")
ax.legend()
ax.set_title("Bayesian Ridge with Prediction Intervals")
plt.tight_layout()
plt.show()
```

---

## 6. Feature Scaling for Linear TS Models

**Linear models require scaled features.** Tree models do not.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler: zero mean, unit variance (most common)
# Use when features are roughly normally distributed
scaler_std = StandardScaler()

# MinMaxScaler: scales to [0, 1]
# Sensitive to outliers — avoid for TS unless data is clean
scaler_mm = MinMaxScaler()

# RobustScaler: uses median and IQR — robust to outliers
# Recommended for time series with spikes
scaler_rob = RobustScaler()

# In a Pipeline (scaler is fit on train ONLY — no leakage)
pipe = Pipeline([("scaler", RobustScaler()), ("ridge", Ridge())])
pipe.fit(X_train, y_train)   # scaler.fit_transform(X_train) internally
pipe.predict(X_test)          # scaler.transform(X_test) internally
```

> **Critical**: Always fit the scaler on **training data only** and transform the test set using the training statistics. Using `Pipeline` ensures this automatically.

---

## 7. Linear Models with Fourier Features

The most powerful use of linear models for time series is combining **regularized regression with Fourier terms**:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def add_fourier_terms(n: int, period: float, k: int) -> pd.DataFrame:
    """Add k Fourier sin/cos pairs for a given period."""
    t = np.arange(n)
    cols = {}
    for i in range(1, k + 1):
        cols[f"sin_{period:.0f}_{i}"] = np.sin(2 * np.pi * i * t / period)
        cols[f"cos_{period:.0f}_{i}"] = np.cos(2 * np.pi * i * t / period)
    return pd.DataFrame(cols)

# Build feature matrix: Fourier + lags + external
n_train = len(train)
fourier_weekly = add_fourier_terms(n_train, period=7,      k=3)
fourier_yearly = add_fourier_terms(n_train, period=365.25, k=6)

X_fourier = pd.concat([fourier_weekly, fourier_yearly], axis=1)
X_fourier["lag_1"] = train.shift(1).values
X_fourier["lag_7"] = train.shift(7).values
X_fourier = X_fourier.dropna()
y_fourier = train.iloc[7:]  # align with dropna

pipe_fourier = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=[0.1, 1.0, 10, 100], cv=tscv)),
])
pipe_fourier.fit(X_fourier, y_fourier)
print(f"Ridge+Fourier CV α: {pipe_fourier.named_steps['ridge'].alpha_:.2f}")
```

---

## 8. When Linear Models Win

| Scenario | Why Linear Model Wins |
|----------|----------------------|
| < 200 observations | Not enough data for trees to generalize |
| Highly regular, smooth seasonality | Fourier terms capture it precisely |
| Need coefficient interpretability | Direct coefficient reading |
| Rapid prototyping / iteration | 100× faster training than boosting |
| Production on low-power hardware | Tiny model size, fast inference |
| Regulatory requirements (banking, insurance) | Explainable, auditable |
| Sparse, high-dimensional feature spaces | Lasso provides automatic selection |

---

## 9. Conformal Prediction Intervals for Any ML Model

Statistical models (ARIMA, ETS) provide **exact analytical prediction intervals**. ML models don't — but **Conformal Prediction** gives distribution-free, statistically valid intervals for ANY trained model.

### 9.1 Core Idea

```
Split Conformal Prediction:

1. Split train into: proper_train + calibration_set
   (calibration set is chronologically AFTER proper_train)

2. Fit model on proper_train

3. Compute residuals on calibration set:
   r_i = |y_i - ŷ_i|   for all i in calibration set

4. Compute q = (1-α)(1 + 1/n_cal) quantile of residuals r_i
   (α = target error rate, e.g., 0.05 for 95% coverage)

5. For test predictions:
   PI = [ŷ(x) - q,  ŷ(x) + q]
   
   Coverage guarantee: P(y ∈ PI) ≥ 1 - α
   This holds WITHOUT any distributional assumption!
```

### 9.2 Time Series Split (Chronological Calibration)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def conformal_prediction_intervals(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    model,
    alpha: float = 0.05,            # error rate (0.05 = 95% coverage)
    calibration_frac: float = 0.2,  # fraction of train used for calibration
) -> pd.DataFrame:
    """
    Split conformal prediction for time series.
    
    IMPORTANT: calibration set must be CHRONOLOGICALLY AFTER proper_train
    to respect temporal ordering.
    
    Parameters:
        alpha            : desired miscoverage rate (0.05 = 95% PI)
        calibration_frac : fraction of training data held out as calibration set
    
    Returns:
        DataFrame with columns: [pred, lower, upper]
    """
    n = len(X_train)
    split_idx = int(n * (1 - calibration_frac))
    
    # Chronological split: proper train vs. calibration
    X_prop  = X_train.iloc[:split_idx]
    y_prop  = y_train.iloc[:split_idx]
    X_calib = X_train.iloc[split_idx:]
    y_calib = y_train.iloc[split_idx:]
    
    # Fit model on proper train only
    model.fit(X_prop, y_prop)
    
    # Compute nonconformity scores (absolute residuals) on calibration set
    y_calib_pred = model.predict(X_calib)
    residuals = np.abs(y_calib.values - y_calib_pred)
    
    # Conformal quantile
    n_cal = len(residuals)
    q_level = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
    q_level = min(q_level, 1.0)
    q = np.quantile(residuals, q_level)
    
    print(f"Conformal quantile (q) at {(1-alpha)*100:.0f}% level: {q:.4f}")
    
    # Generate test predictions and intervals
    y_pred = model.predict(X_test)
    return pd.DataFrame({
        "pred":  y_pred,
        "lower": y_pred - q,
        "upper": y_pred + q,
    }, index=X_test.index)


# Usage with Ridge (works with ANY sklearn-compatible model)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge",  Ridge(alpha=10.0)),
])

pi_df = conformal_prediction_intervals(
    X_train, y_train, X_test,
    model=pipe, alpha=0.05,   # 95% PI
)
print(pi_df.head())
```

### 9.3 Validation — Coverage Check

```python
# Empirical coverage should be ≥ 1 - alpha
coverage = (
    (y_test.values >= pi_df["lower"].values) &
    (y_test.values <= pi_df["upper"].values)
).mean()
print(f"Empirical coverage: {coverage:.3f}  (target: {1-0.05:.2f})")

# Average interval width (narrower = better, given coverage is met)
width = (pi_df["upper"] - pi_df["lower"]).mean()
print(f"Average PI width: {width:.4f}")
```

### 9.4 Plot Conformal Prediction Intervals

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(y_test.values, color="black", linewidth=2, label="Actual")
ax.plot(pi_df["pred"].values, color="#D7191C", linewidth=2,
        linestyle="--", label="Predicted")
ax.fill_between(
    range(len(pi_df)),
    pi_df["lower"], pi_df["upper"],
    alpha=0.2, color="#D7191C",
    label=f"95% Conformal PI (coverage={coverage:.2f})"
)
ax.legend()
ax.set_title("Conformal Prediction Intervals (Distribution-Free)")
plt.tight_layout()
plt.show()
```

### 9.5 Conformal vs. Parametric Intervals

| | Conformal PI | ARIMA Analytical PI | Bayesian PI |
|--|-------------|--------------------|--------------|
| **Distributional assumption** | None | Gaussian errors | Prior + posterior |
| **Works for any model** | ✅ | ❌ (ARIMA only) | Depends on model |
| **Valid coverage guarantee** | ✅ (finite-sample) | Asymptotic | Credible, not frequentist |
| **Interval shape** | Symmetric (split CP) | Widening with horizon | Flexible |
| **Computation** | Fast | Built-in | Slow (MCMC) |
| **Best for** | Any ML model | ARIMA/ETS pipelines | Small data, uncertainty quantification |

> **Production rule**: Use conformal prediction whenever you deploy an ML forecasting model that needs reliable prediction intervals — it requires no distributional assumptions and is straightforward to implement.

---

*← [04 — Random Forest](./04_random_forest_and_tree_models.md) | [Module README](./README.md) | Next: [06 — Ensembles](./06_model_stacking_and_ensembles.md) →*
