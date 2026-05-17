# 05 — Probabilistic Forecasting

> **Module**: 07 Forecasting Strategies | **File**: 5 of 6
>
> Point forecasts are single numbers. But the future is uncertain. Probabilistic forecasting provides the **full distribution** of possible outcomes — a critical requirement for risk management, safety stock optimization, and resource planning.

---

## Table of Contents

1. [Why Probabilistic Forecasting?](#1-why-probabilistic-forecasting)
2. [Forecast Distributions and Quantiles](#2-forecast-distributions-and-quantiles)
3. [Quantile Regression](#3-quantile-regression)
4. [Prediction Intervals from Residuals](#4-prediction-intervals-from-residuals)
5. [Bootstrapped Prediction Intervals](#5-bootstrapped-prediction-intervals)
6. [Distributional Forecasting — Full Distribution](#6-distributional-forecasting--full-distribution)
7. [Evaluation — Pinball Loss and CRPS](#7-evaluation--pinball-loss-and-crps)
8. [Production Implementation](#8-production-implementation)

---

## 1. Why Probabilistic Forecasting?

### 1.1 The Limitations of Point Forecasts

```
Point forecast: "We'll sell 1,000 units next week"

Reality:
  - If actual = 800:  we're overstocked, capital tied up
  - If actual = 1,200: we're understocked, lost sales, customer dissatisfaction

Probabilistic forecast: "We'll sell between 750–1,300 units (90% confidence)"
→ Decision maker can set safety stock at 1,300 to avoid stockout with high probability
→ Or optimize the cost/service-level tradeoff using the full distribution
```

### 1.2 When Probabilistic Forecasts Are Essential

| Application                    | What You Need                                | Why Point Forecast Fails         |
|--------------------------------|----------------------------------------------|----------------------------------|
| Inventory optimization         | P90 demand (safety stock level)              | Underprediction → stockout       |
| Energy grid dispatch           | P5 renewable generation (worst case)         | Grid stability requires margins  |
| Financial risk management      | VaR (P5 return distribution)                 | Tail risk assessment             |
| Clinical trial planning        | Distribution of patient response times       | Resource allocation              |
| Load balancing (cloud)         | P95 request rate                             | Capacity planning under peaks    |

### 1.3 Key Output Types

| Output                   | Definition                                             |
|--------------------------|--------------------------------------------------------|
| **Quantile q**           | Value below which q fraction of outcomes will fall     |
| **Prediction interval**  | [qₐ/₂, q₁₋ₐ/₂] — range containing 1-α of outcomes   |
| **Full PDF/CDF**         | Complete probability distribution over outcome space   |
| **Scenarios/samples**    | Draws from the forecast distribution                   |

---

## 2. Forecast Distributions and Quantiles

### 2.1 Quantile Definition

The τ-quantile of a random variable Y satisfies:

```
Qτ = inf{y : P(Y ≤ y) ≥ τ}

Examples:
  Q₀.₅ = median         (50% of outcomes below this)
  Q₀.₉ = 90th percentile (90% of outcomes below this)
  Q₀.₁ = 10th percentile (10% of outcomes below this)

80% Prediction Interval: [Q₀.₁, Q₀.₉]
95% Prediction Interval: [Q₀.₀₂₅, Q₀.₉₇₅]
```

### 2.2 Coverage

A (1-α)% prediction interval has correct **coverage** if, empirically:

```
Coverage = fraction of actual values falling within the interval ≈ (1 - α)

Undercoverage: actual falls outside interval too often → interval too narrow
Overcoverage:  interval too wide → uninformative but safe
```

---

## 3. Quantile Regression

### 3.1 The Pinball Loss

Standard regression minimizes squared error (targets the mean). Quantile regression minimizes the **pinball loss** (asymmetric L1), which targets any desired quantile:

```
L_τ(y, ŷ) = τ · max(y - ŷ, 0)  +  (1-τ) · max(ŷ - y, 0)

= τ(y - ŷ)    if y ≥ ŷ   (penalizes underprediction with weight τ)
  (τ-1)(y-ŷ)  if y < ŷ   (penalizes overprediction with weight 1-τ)

→ Setting τ = 0.9: predicting too low is penalized 9× more than too high
→ Optimizer learns to output values that are exceeded by ~10% of observations
```

### 3.2 Quantile Regression with LightGBM

```python
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss

def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    quantiles: list[float] = (0.1, 0.5, 0.9),
) -> dict[float, lgb.LGBMRegressor]:
    """
    Train one LightGBM quantile regression model per quantile.

    Parameters
    ----------
    X_train, y_train : training features and targets
    X_val, y_val     : validation features and targets
    quantiles        : list of quantiles to estimate

    Returns
    -------
    dict {quantile: fitted_model}
    """
    models = {}

    for tau in quantiles:
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="quantile",       # <── quantile regression loss
            alpha=tau,                  # <── which quantile to target
            metric="quantile",
            verbose=-1,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        val_preds = model.predict(X_val)
        pinball   = mean_pinball_loss(y_val, val_preds, alpha=tau)
        print(f"  τ={tau:.2f} | Pinball Loss: {pinball:.4f}")
        models[tau] = model

    return models


def predict_quantile_intervals(
    models: dict[float, lgb.LGBMRegressor],
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate quantile prediction matrix.

    Returns DataFrame with columns for each quantile.
    """
    results = pd.DataFrame(index=X_test.index)
    for tau, model in sorted(models.items()):
        results[f"q{int(tau*100):02d}"] = model.predict(X_test)

    # Enforce monotonicity: q10 ≤ q50 ≤ q90
    # (quantile crossing can occur with separate models)
    cols = sorted(results.columns)
    for i in range(1, len(cols)):
        results[cols[i]] = np.maximum(results[cols[i]], results[cols[i-1]])

    return results
```

### 3.3 Visualization

```python
import matplotlib.pyplot as plt

def plot_quantile_forecast(
    actual: pd.Series,
    quantile_df: pd.DataFrame,
    title: str = "Quantile Forecast with Prediction Intervals",
):
    """
    Plot point forecast (median) with shaded prediction intervals.

    Parameters
    ----------
    actual       : observed target values (test set)
    quantile_df  : DataFrame from predict_quantile_intervals()
                   expected columns: q10, q50, q90 (and optionally q25, q75)
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    # Actual values
    ax.plot(actual.index, actual.values, color="black", linewidth=2,
            label="Actual", zorder=5)

    # Median forecast
    if "q50" in quantile_df.columns:
        ax.plot(quantile_df.index, quantile_df["q50"], color="#E84646",
                linewidth=2, linestyle="--", label="Median (q50)")

    # 80% prediction interval
    if "q10" in quantile_df.columns and "q90" in quantile_df.columns:
        ax.fill_between(quantile_df.index, quantile_df["q10"], quantile_df["q90"],
                        color="#E84646", alpha=0.2, label="80% PI [q10, q90]")

    # 50% prediction interval (optional)
    if "q25" in quantile_df.columns and "q75" in quantile_df.columns:
        ax.fill_between(quantile_df.index, quantile_df["q25"], quantile_df["q75"],
                        color="#E84646", alpha=0.35, label="50% PI [q25, q75]")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
```

### 3.4 XGBoost Quantile Regression

```python
import xgboost as xgb

# XGBoost also supports quantile regression
model_xgb_q90 = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:quantileerror",   # requires XGBoost >= 1.7
    quantile_alpha=0.9,              # target quantile
    eval_metric="quantile",
    tree_method="hist",
    random_state=42,
)
```

---

## 4. Prediction Intervals from Residuals

### 4.1 Residual-Based Intervals (Gaussian Assumption)

If residuals are approximately normally distributed:

```
ŷₜ₊ₕ ± z_{α/2} · σ̂ₕ

Where:
  σ̂ₕ = estimated forecast standard error at horizon h
  z_{0.025} = 1.96  → 95% interval
  z_{0.05}  = 1.645 → 90% interval
```

```python
import numpy as np
from scipy import stats

def residual_based_intervals(
    point_forecasts: np.ndarray,
    train_residuals: np.ndarray,
    horizon: int,
    coverage: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Gaussian prediction intervals from in-sample residuals.

    Parameters
    ----------
    point_forecasts : (H,) array of point forecasts
    train_residuals : (n_train,) in-sample residuals
    horizon         : forecast horizon
    coverage        : desired coverage (e.g., 0.90 = 90%)

    Returns
    -------
    lower, upper : arrays of lower and upper bounds for each step
    """
    alpha   = 1 - coverage
    z_score = stats.norm.ppf(1 - alpha / 2)

    sigma   = train_residuals.std()
    # Uncertainty grows with horizon (empirical rule: σ_h ≈ σ * sqrt(h))
    sigma_h = sigma * np.sqrt(np.arange(1, horizon + 1))

    lower = point_forecasts - z_score * sigma_h
    upper = point_forecasts + z_score * sigma_h

    return lower, upper
```

### 4.2 Quantile Residuals (Non-Gaussian)

For non-Gaussian residuals, use the empirical quantiles of the residual distribution:

```python
def empirical_interval_from_residuals(
    point_forecasts: np.ndarray,
    train_residuals: np.ndarray,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Empirical prediction intervals using quantiles of the residual distribution.
    Does NOT assume Gaussian errors.
    """
    lower_offset = np.quantile(train_residuals, lower_q)
    upper_offset = np.quantile(train_residuals, upper_q)

    lower = point_forecasts + lower_offset
    upper = point_forecasts + upper_offset

    return lower, upper
```

---

## 5. Bootstrapped Prediction Intervals

### 5.1 Temporal Bootstrap for Time Series

Standard bootstrap assumes i.i.d. samples — invalid for time series (autocorrelated residuals). Use **block bootstrap** or **residual bootstrap**:

```python
import numpy as np

def residual_bootstrap_intervals(
    model,
    X_test: np.ndarray,
    train_residuals: np.ndarray,
    n_bootstrap: int = 1000,
    coverage: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap prediction intervals for a single-step regressor.

    Strategy: add random resampled residuals to point forecasts.
    Valid when residuals are approximately i.i.d. (check ACF of residuals first).

    Parameters
    ----------
    model           : fitted sklearn-compatible model
    X_test          : test features (n_test, n_features)
    train_residuals : in-sample residuals (n_train,)
    n_bootstrap     : number of bootstrap replicates
    coverage        : desired coverage level

    Returns
    -------
    lower, upper : prediction interval bounds, shape (n_test,)
    """
    point_preds = model.predict(X_test)   # shape: (n_test,)
    n_test      = len(point_preds)
    alpha       = 1 - coverage

    # Bootstrap distribution
    boot_preds = np.zeros((n_bootstrap, n_test))
    for b in range(n_bootstrap):
        sampled_residuals = np.random.choice(train_residuals, size=n_test, replace=True)
        boot_preds[b]     = point_preds + sampled_residuals

    lower = np.quantile(boot_preds, alpha / 2, axis=0)
    upper = np.quantile(boot_preds, 1 - alpha / 2, axis=0)

    return lower, upper


def block_bootstrap_intervals(
    model,
    X_test: np.ndarray,
    train_residuals: np.ndarray,
    block_length: int = 7,
    n_bootstrap: int = 1000,
    coverage: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Block bootstrap — resamples contiguous blocks of residuals to preserve autocorrelation.

    Parameters
    ----------
    block_length : length of each block (≈ autocorrelation decay lag)
    """
    point_preds = model.predict(X_test)
    n_test      = len(point_preds)
    n_res       = len(train_residuals)
    alpha       = 1 - coverage
    boot_preds  = np.zeros((n_bootstrap, n_test))

    for b in range(n_bootstrap):
        blocks = []
        while sum(len(bl) for bl in blocks) < n_test:
            start = np.random.randint(0, n_res - block_length + 1)
            blocks.append(train_residuals[start:start + block_length])
        resampled = np.concatenate(blocks)[:n_test]
        boot_preds[b] = point_preds + resampled

    lower = np.quantile(boot_preds, alpha / 2, axis=0)
    upper = np.quantile(boot_preds, 1 - alpha / 2, axis=0)

    return lower, upper
```

---

## 6. Distributional Forecasting — Full Distribution

### 6.1 Distributional Loss Functions

Some deep learning frameworks output the **parameters of a distribution** rather than quantiles:

| Distribution    | Parameters         | Use Case                          |
|-----------------|--------------------|-----------------------------------|
| Normal          | (μ, σ)             | Symmetric, continuous targets     |
| Student-t       | (μ, σ, ν)          | Heavy-tailed, outlier-robust      |
| Negative Binomial | (μ, α)           | Count data (demand, arrivals)     |
| Beta            | (α, β)             | Bounded [0,1] targets             |
| Log-Normal      | (μ_log, σ_log)     | Non-negative, right-skewed        |

### 6.2 Distributional Forecasting with neuralforecast

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss

import pandas as pd

# Multiple quantile loss — outputs a vector of quantile forecasts
model_mq = NHITS(
    h=12,
    input_size=48,
    max_steps=500,
    loss=MQLoss(
        quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]   # 7 quantiles
    ),
    random_seed=42,
)

# Distributional loss — outputs Normal(μ, σ) parameters
model_dist = NHITS(
    h=12,
    input_size=48,
    max_steps=500,
    loss=DistributionLoss(
        distribution="Normal",     # or "StudentT", "NegativeBinomial"
        level=[80, 90, 95],        # coverage levels for interval output
    ),
    random_seed=42,
)

nf = NeuralForecast(models=[model_mq, model_dist], freq="M")
nf.fit(df=train_df)

forecasts = nf.predict()
# Columns include: NHITS-q5, NHITS-q10, ..., NHITS-q95  (for MQLoss)
# And: NHITS-lo-80, NHITS-hi-80, NHITS-lo-90, NHITS-hi-90  (for DistributionLoss)
print(forecasts.columns.tolist())
```

---

## 7. Evaluation — Pinball Loss and CRPS

### 7.1 Pinball Loss (per quantile)

```python
from sklearn.metrics import mean_pinball_loss
import numpy as np

def evaluate_quantile_forecasts(
    y_true: np.ndarray,
    quantile_preds: dict[float, np.ndarray],
) -> pd.DataFrame:
    """
    Compute pinball loss for each quantile level.

    Parameters
    ----------
    y_true         : actual values (n_test,)
    quantile_preds : dict {tau: predictions_array}

    Returns
    -------
    DataFrame with columns [quantile, pinball_loss]
    """
    rows = []
    for tau, preds in sorted(quantile_preds.items()):
        pb = mean_pinball_loss(y_true, preds, alpha=tau)
        rows.append({"quantile": tau, "pinball_loss": pb})
    return pd.DataFrame(rows)
```

### 7.2 CRPS — Continuous Ranked Probability Score

CRPS is the gold standard metric for probabilistic forecasting. It evaluates the full forecast distribution:

```
CRPS(F, y) = ∫₋∞^∞ (F(x) - 1{x ≥ y})² dx

Where F is the forecast CDF and y is the actual value.

Properties:
  - Equals MAE when F is a point forecast (special case)
  - Proper scoring rule: minimized only by the true distribution
  - Lower = better
  - Captures both calibration and sharpness simultaneously
```

```python
import numpy as np

def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Analytical CRPS for Gaussian forecast distributions.

    Parameters
    ----------
    mu    : predicted means (n,)
    sigma : predicted std deviations (n,)
    y     : actual values (n,)

    Returns
    -------
    crps : per-sample CRPS scores (n,)
    """
    from scipy.stats import norm
    z    = (y - mu) / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return crps


def crps_empirical_from_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Empirical CRPS from Monte Carlo samples (works for any distribution).

    CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    Where X, X' ~ F (independent draws from forecast distribution)

    Parameters
    ----------
    samples : (n_samples, n_test) — forecast samples for each test point
    y       : (n_test,) — actual values

    Returns
    -------
    crps : (n_test,) — per-observation CRPS
    """
    n_samples, n_test = samples.shape
    crps_vals = np.zeros(n_test)

    for i in range(n_test):
        s = samples[:, i]
        # E[|X - y|]
        term1 = np.mean(np.abs(s - y[i]))
        # E[|X - X'|] (pairwise distances between samples — approximate with n_samples)
        term2 = 0.5 * np.mean(np.abs(s[:, None] - s[None, :]))
        crps_vals[i] = term1 - term2

    return crps_vals


def mean_crps(mu, sigma, y):
    """Scalar CRPS averaged over all test observations."""
    return crps_gaussian(mu, sigma, y).mean()
```

### 7.3 Coverage Evaluation

```python
def empirical_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_coverage: float,
) -> dict:
    """
    Compute empirical coverage of prediction intervals.

    A well-calibrated 90% PI should contain ~90% of actual values.
    """
    in_interval = ((y_true >= lower) & (y_true <= upper)).mean()
    return {
        "nominal_coverage":  nominal_coverage,
        "empirical_coverage": float(in_interval),
        "coverage_error":     float(in_interval - nominal_coverage),
        "calibrated":         abs(in_interval - nominal_coverage) < 0.02,
    }
```

---

## 8. Production Implementation

### 8.1 Multi-Quantile Pipeline

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_pinball_loss

class ProbabilisticForecaster:
    """
    Production quantile regression forecaster.
    Outputs calibrated prediction intervals using multiple quantile models.
    """

    def __init__(
        self,
        quantiles: list = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
        lags: list = None,
    ):
        self.quantiles   = list(quantiles)
        self.lags        = lags or [1, 7, 14, 28]
        self.models_: dict[float, lgb.LGBMRegressor] = {}
        self.feature_cols_: list = []

    def _build_features(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_lag = max(self.lags)
        X, y    = [], []
        for i in range(max_lag, len(series)):
            X.append([series[i - lag] for lag in self.lags])
            y.append(series[i])
        return np.array(X), np.array(y)

    def fit(self, series: np.ndarray) -> "ProbabilisticForecaster":
        X, y   = self._build_features(series)
        split  = int(len(X) * 0.85)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        for tau in self.quantiles:
            model = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, num_leaves=31,
                objective="quantile", alpha=tau, metric="quantile", verbose=-1,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            self.models_[tau] = model

        return self

    def predict(self, series_end: np.ndarray) -> pd.Series:
        """Predict all quantiles from the last observed values."""
        features = np.array([series_end[-lag] for lag in self.lags]).reshape(1, -1)
        return pd.Series(
            {f"q{int(tau*100):02d}": self.models_[tau].predict(features)[0]
             for tau in self.quantiles}
        )

    def predict_horizon(self, series: np.ndarray, h: int) -> pd.DataFrame:
        """Generate quantile forecasts for h steps using recursive strategy."""
        from collections import deque
        buffer  = deque(series[-max(self.lags):], maxlen=max(self.lags))
        records = []

        for step in range(h):
            features = np.array([list(buffer)[-lag] for lag in self.lags]).reshape(1, -1)
            row = {"step": step + 1}
            for tau in self.quantiles:
                row[f"q{int(tau*100):02d}"] = self.models_[tau].predict(features)[0]
            records.append(row)
            # Feed median prediction back as next lag
            buffer.append(row["q50"])

        return pd.DataFrame(records).set_index("step")
```

---

*← [04 — Hierarchical Forecasting](./04_hierarchical_forecasting.md) | [Module README](./README.md) | Next: [06 — Conformal Prediction](./06_conformal_prediction_for_ts.md) →*
