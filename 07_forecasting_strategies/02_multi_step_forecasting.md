# 02 — Multi-Step Forecasting

> **Module**: 07 Forecasting Strategies | **File**: 2 of 6
>
> A comprehensive guide to multi-step forecasting mechanics — horizon selection, feature engineering without data leakage, chained predictions, and practical implementation patterns.

---

## Table of Contents

1. [Defining the Forecast Horizon](#1-defining-the-forecast-horizon)
2. [Leakage-Free Feature Engineering for Multi-Step](#2-leakage-free-feature-engineering-for-multi-step)
3. [Chained Prediction (Recursive) — Deep Dive](#3-chained-prediction-recursive--deep-dive)
4. [Direct Multi-Output — Deep Dive](#4-direct-multi-output--deep-dive)
5. [Rolling Origin Evaluation](#5-rolling-origin-evaluation)
6. [Horizon-Dependent Accuracy Decay](#6-horizon-dependent-accuracy-decay)
7. [Production Multi-Step Pipeline](#7-production-multi-step-pipeline)

---

## 1. Defining the Forecast Horizon

### 1.1 What is the Horizon?

The **forecast horizon** `H` is the number of future steps you predict beyond the last observed point. Choosing `H` is a business decision, not a modelling one:

| Domain              | Typical H      | Frequency | Reasoning                              |
|---------------------|---------------|-----------|----------------------------------------|
| Retail (weekly)     | 8–12 weeks    | Weekly    | Lead time for procurement              |
| Energy (hourly)     | 24–48 hours   | Hourly    | Day-ahead power market settlement      |
| Finance             | 1–5 days      | Daily     | Position risk window                   |
| Macroeconomics      | 4–8 quarters  | Quarterly | Policy planning horizon                |
| Manufacturing       | 4–12 weeks    | Weekly    | Production scheduling                  |

### 1.2 Short vs. Long Horizon Considerations

```
Short horizon (H ≤ 6 steps):
  ✅ High accuracy achievable
  ✅ All forecasting strategies competitive
  ✅ Recent lags dominate — model is simpler
  ❌ Limited planning value in some domains

Long horizon (H > 12 steps):
  ✅ High operational value (procurement, staffing)
  ❌ Accuracy degrades rapidly — uncertainty explodes
  ❌ Recursive strategy accumulates errors heavily
  ❌ Need explicit uncertainty quantification (→ Module 07/05)
  → Use MIMO or Direct; provide prediction intervals
```

### 1.3 Horizon as a Hyperparameter

In production systems, the horizon is fixed by the business. However, during development:

```python
# Evaluate MAE as a function of horizon to understand accuracy decay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def horizon_accuracy_profile(y_true: np.ndarray, y_pred_matrix: np.ndarray) -> dict:
    """
    Compute per-horizon MAE from a Direct/MIMO prediction matrix.

    Parameters
    ----------
    y_true        : (n_test, H) matrix of true future values
    y_pred_matrix : (n_test, H) matrix of predicted future values

    Returns
    -------
    dict with keys 'horizon_steps' and 'mae_per_step'
    """
    H = y_true.shape[1]
    mae_per_step = [
        mean_absolute_error(y_true[:, h], y_pred_matrix[:, h])
        for h in range(H)
    ]
    return {"horizon_steps": list(range(1, H + 1)), "mae_per_step": mae_per_step}
```

---

## 2. Leakage-Free Feature Engineering for Multi-Step

### 2.1 The Leakage Problem

Feature engineering for one-step-ahead models is straightforward. Multi-step forecasting introduces **temporal leakage risk** when features use data from the forecast period:

```
Dangerous (leaks future data):
  At predict time for step h=5, using roll_7_mean computed over t+1..t+5
  — those values don't exist yet!

Safe rule: At any forecast origin t, all features must use data from t or earlier.
```

### 2.2 Safe Feature Construction

```python
import pandas as pd
import numpy as np

def build_ts_features_no_leakage(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Build a complete leakage-free feature matrix for multi-step forecasting.

    All rolling/lag features are computed with `shift(1)` before any window
    operation — ensuring no future information is used.

    Parameters
    ----------
    df         : DataFrame with datetime index and target column
    target_col : name of the column to forecast

    Returns
    -------
    DataFrame with features; target column preserved for label construction
    """
    df = df.copy()
    t = target_col

    # ── Lag features ─────────────────────────────────────────────────────────
    for lag in [1, 2, 3, 6, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = df[t].shift(lag)

    # ── Rolling statistics — always shift(1) before rolling ──────────────────
    s1 = df[t].shift(1)   # one-step-shifted series (excludes current t)
    for window in [7, 14, 30, 90]:
        df[f"roll{window}_mean"] = s1.rolling(window, min_periods=1).mean()
        df[f"roll{window}_std"]  = s1.rolling(window, min_periods=1).std()
        df[f"roll{window}_min"]  = s1.rolling(window, min_periods=1).min()
        df[f"roll{window}_max"]  = s1.rolling(window, min_periods=1).max()

    # ── Expanding mean (trend proxy) ─────────────────────────────────────────
    df["expanding_mean"] = s1.expanding(min_periods=30).mean()

    # ── Exponentially weighted moving average ────────────────────────────────
    df["ewm_alpha0.3"] = s1.ewm(alpha=0.3, adjust=False).mean()
    df["ewm_alpha0.1"] = s1.ewm(alpha=0.1, adjust=False).mean()

    # ── Calendar features (always available, no leakage) ─────────────────────
    idx = df.index
    if hasattr(idx, "month"):
        df["month_sin"]    = np.sin(2 * np.pi * idx.month / 12)
        df["month_cos"]    = np.cos(2 * np.pi * idx.month / 12)
        df["dow_sin"]      = np.sin(2 * np.pi * idx.dayofweek / 7)
        df["dow_cos"]      = np.cos(2 * np.pi * idx.dayofweek / 7)
        df["week_sin"]     = np.sin(2 * np.pi * idx.isocalendar().week.values / 52)
        df["week_cos"]     = np.cos(2 * np.pi * idx.isocalendar().week.values / 52)
        df["is_weekend"]   = (idx.dayofweek >= 5).astype(int)
        df["quarter"]      = idx.quarter
        df["day_of_year"]  = idx.dayofyear

    # ── Difference features (captures momentum) ──────────────────────────────
    df["diff_1"]  = df[t].diff(1)    # day-over-day change
    df["diff_7"]  = df[t].diff(7)    # week-over-week change
    df["diff_28"] = df[t].diff(28)   # month-over-month change

    return df.dropna()
```

### 2.3 Feature Availability Matrix

Not all features are available at all forecast horizons when using direct strategy:

```
Horizon h=1:   lag_1 ✅  lag_7 ✅  roll7 ✅  calendar ✅
Horizon h=2:   lag_1 ❌  lag_2 ✅  lag_7 ✅  calendar ✅
               (lag_1 at t+2 would be ŷₜ₊₁ — only valid for Recursive)
Horizon h=7:   lag_1..6 ❌  lag_7 ✅  roll7 ⚠️  calendar ✅
Horizon h=14:  lag_1..13 ❌  lag_14 ✅  calendar ✅
```

**Practical guideline**: When using Direct strategy with horizon `h`, include lags ≥ `h`. For calendar features, always include them — they are always known.

```python
def get_safe_feature_cols(all_feature_cols: list[str], horizon_step: int) -> list[str]:
    """
    Filter feature columns to those safe for a given horizon step.

    Lags < horizon_step cannot be used without recursive contamination.
    Rolling features that rely on lag_1 are safe only at h=1.
    Calendar features are always safe.
    """
    safe = []
    for col in all_feature_cols:
        if col.startswith("lag_"):
            lag_n = int(col.split("_")[1])
            if lag_n >= horizon_step:
                safe.append(col)
        elif col.startswith(("month_", "dow_", "week_", "day_of", "quarter", "is_weekend")):
            safe.append(col)   # calendar features always available
        elif col.startswith("roll") or col.startswith("ewm"):
            # Rolling features use shift(1) base — safe at h=1, risky at h>1
            # Conservative: only include if base lag is >= horizon
            safe.append(col)   # acceptable if base was shift(1)
        elif col.startswith("diff_"):
            lag_n = int(col.split("_")[1])
            if lag_n >= horizon_step:
                safe.append(col)
    return safe
```

---

## 3. Chained Prediction (Recursive) — Deep Dive

### 3.1 State Buffer Pattern

The cleanest implementation uses a **state buffer** — a deque that holds the last `max_lag` observations and grows with predictions:

```python
from collections import deque
import numpy as np

def chained_forecast(
    model,
    history: np.ndarray,
    lags: list[int],
    calendar_features: np.ndarray,
    h: int,
) -> np.ndarray:
    """
    Recursive chained forecast with calendar features.

    Parameters
    ----------
    model             : fitted sklearn-compatible model (one-step-ahead)
    history           : array of observed values (at least max(lags) points)
    lags              : lag indices used during training
    calendar_features : (h, n_calendar) array of future calendar features
                        (day-of-week, month, etc. — always known)
    h                 : forecast horizon

    Returns
    -------
    predictions : 1D array of length h
    """
    max_lag = max(lags)
    buffer  = deque(history[-max_lag:], maxlen=max_lag)
    preds   = []

    for step in range(h):
        lag_feats  = np.array([list(buffer)[-lag] for lag in lags])
        cal_feats  = calendar_features[step]
        features   = np.concatenate([lag_feats, cal_feats]).reshape(1, -1)

        pred = model.predict(features)[0]
        preds.append(pred)
        buffer.append(pred)   # inject prediction back into rolling window

    return np.array(preds)
```

### 3.2 Error Accumulation Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_error_accumulation(n_sims: int = 500, H: int = 30, sigma: float = 0.5):
    """
    Simulate recursive forecast error growth over horizon H.

    At each step, a Gaussian noise ε ~ N(0, σ²) is added to the prediction.
    Errors from previous steps propagate through the lag structure.
    """
    errors = np.zeros((n_sims, H))

    for s in range(n_sims):
        pred = 0.0
        for h in range(H):
            noise = np.random.normal(0, sigma)
            pred  = pred + noise   # recursive: error adds each step
            errors[s, h] = pred

    mean_abs_err = np.abs(errors).mean(axis=0)
    std_err      = errors.std(axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, H+1), mean_abs_err, label="Mean |Error|", color="tomato")
    plt.fill_between(range(1, H+1),
                     mean_abs_err - std_err,
                     mean_abs_err + std_err,
                     alpha=0.3, color="tomato", label="±1 Std")
    plt.xlabel("Forecast Horizon Step")
    plt.ylabel("Absolute Error")
    plt.title("Recursive Error Accumulation (simulated)")
    plt.legend()
    plt.tight_layout()
    plt.show()
```

---

## 4. Direct Multi-Output — Deep Dive

### 4.1 Horizon-Specific Training

```python
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

class DirectForecaster:
    """
    Direct multi-step forecaster: trains H independent models.

    Each model fₕ is fitted on (X_observed, y_t+h) pairs.
    Features are always computed from real observed values.
    """

    def __init__(self, horizon: int, lags: list[int]):
        self.horizon = horizon
        self.lags    = lags
        self.models_: dict[int, LGBMRegressor] = {}

    def _build_X(self, series: np.ndarray) -> np.ndarray:
        max_lag = max(self.lags)
        rows = []
        for i in range(max_lag, len(series)):
            rows.append([series[i - lag] for lag in self.lags])
        return np.array(rows)

    def fit(self, series: np.ndarray) -> "DirectForecaster":
        """Train one model per horizon step."""
        max_lag = max(self.lags)
        X_full  = self._build_X(series)   # shape: (n - max_lag, n_lags)

        for h in range(1, self.horizon + 1):
            # Target: y[max_lag + i + h - 1] for i = 0, 1, ...
            n_valid = len(series) - max_lag - h + 1
            if n_valid <= 0:
                raise ValueError(f"Not enough data for horizon h={h}")

            X_h = X_full[:n_valid]
            y_h = series[max_lag + h - 1 : max_lag + h - 1 + n_valid]

            split = int(len(X_h) * 0.8)
            model = LGBMRegressor(
                n_estimators=300, learning_rate=0.05,
                num_leaves=31, min_child_samples=10,
                verbose=-1,
            )
            model.fit(X_h[:split], y_h[:split])
            self.models_[h] = model

        return self

    def predict(self, series_end: np.ndarray) -> np.ndarray:
        """
        Generate H-step direct forecast from the last `max_lag` observations.

        Parameters
        ----------
        series_end : last observed values (at least max(lags) points)
        """
        max_lag  = max(self.lags)
        features = np.array([series_end[-lag] for lag in self.lags]).reshape(1, -1)
        return np.array([self.models_[h].predict(features)[0]
                         for h in range(1, self.horizon + 1)])
```

---

## 5. Rolling Origin Evaluation

### 5.1 What is Rolling Origin?

Rolling origin (also called **expanding window** or **walk-forward**) evaluation is the correct way to evaluate multi-step models. It simulates how the model will be used in production:

```
T = 200, H = 7, n_origins = 5

Origin 1: train on [1..150],  forecast [151..157], evaluate
Origin 2: train on [1..157],  forecast [158..164], evaluate
Origin 3: train on [1..164],  forecast [165..171], evaluate
Origin 4: train on [1..171],  forecast [172..178], evaluate
Origin 5: train on [1..178],  forecast [179..185], evaluate

Average metrics across all origins → unbiased multi-step estimate
```

### 5.2 Implementation

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def rolling_origin_evaluation(
    model_class,
    model_kwargs: dict,
    series: np.ndarray,
    horizon: int,
    n_origins: int,
    min_train_size: int = 100,
) -> dict:
    """
    Walk-forward rolling origin evaluation for multi-step forecasters.

    Parameters
    ----------
    model_class    : class with .fit(series) and .predict(series_end) methods
    model_kwargs   : kwargs passed to model_class constructor
    series         : full time series (1D array)
    horizon        : forecast horizon H
    n_origins      : number of evaluation origins
    min_train_size : minimum training length before first evaluation

    Returns
    -------
    dict with 'mae_per_step', 'rmse_per_step', 'mean_mae', 'mean_rmse'
    """
    n = len(series)
    origins = np.linspace(min_train_size, n - horizon, n_origins, dtype=int)

    all_preds  = np.zeros((n_origins, horizon))
    all_actuals = np.zeros((n_origins, horizon))

    for idx, origin in enumerate(origins):
        train = series[:origin]
        actual = series[origin:origin + horizon]

        model = model_class(**model_kwargs)
        model.fit(train)
        preds = model.predict(train)   # last observations used internally

        all_preds[idx]   = preds
        all_actuals[idx] = actual

    mae_per_step  = [mean_absolute_error(all_actuals[:, h], all_preds[:, h])
                     for h in range(horizon)]
    rmse_per_step = [np.sqrt(mean_squared_error(all_actuals[:, h], all_preds[:, h]))
                     for h in range(horizon)]

    return {
        "mae_per_step":  mae_per_step,
        "rmse_per_step": rmse_per_step,
        "mean_mae":      np.mean(mae_per_step),
        "mean_rmse":     np.mean(rmse_per_step),
    }
```

---

## 6. Horizon-Dependent Accuracy Decay

### 6.1 Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_horizon_accuracy(results_dict: dict[str, dict], title: str = "Forecast Accuracy vs Horizon"):
    """
    Compare multiple strategies' accuracy across horizon steps.

    Parameters
    ----------
    results_dict : dict of {strategy_name: rolling_origin_evaluation() output}
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, res in results_dict.items():
        H = len(res["mae_per_step"])
        steps = range(1, H + 1)
        axes[0].plot(steps, res["mae_per_step"],  marker="o", label=name)
        axes[1].plot(steps, res["rmse_per_step"], marker="s", label=name)

    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        ax.set_xlabel("Forecast Horizon Step")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} per Horizon Step")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


# Example usage (after running rolling_origin_evaluation for each strategy):
# plot_horizon_accuracy({
#     "Recursive": recursive_results,
#     "Direct":    direct_results,
#     "MIMO":      mimo_results,
# })
```

### 6.2 Key Empirical Rules

| Observation                          | Implication                                              |
|--------------------------------------|----------------------------------------------------------|
| MAE doubles by step 6 (Recursive)    | Switch to Direct/MIMO for horizons > 6                   |
| MAE plateau after step 12            | Model has no signal beyond seasonal period               |
| Step-1 MAE >> Naive baseline MAE     | Model is not extracting signal well — check features     |
| RMSE grows faster than MAE           | Large errors at long horizons — outlier forecasts        |

---

## 7. Production Multi-Step Pipeline

```python
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from typing import Optional

class MultiStepPipeline:
    """
    Production multi-step forecasting pipeline supporting Recursive, Direct, and MIMO.

    Usage:
        pipe = MultiStepPipeline(strategy="direct", horizon=12, lags=[1,7,14,28])
        pipe.fit(train_series)
        forecasts = pipe.predict()   # returns (12,) array
    """

    STRATEGIES = ("recursive", "direct", "mimo")

    def __init__(
        self,
        strategy: str,
        horizon: int,
        lags: Optional[list] = None,
        model_params: Optional[dict] = None,
    ):
        assert strategy in self.STRATEGIES, f"strategy must be one of {self.STRATEGIES}"
        self.strategy     = strategy
        self.horizon      = horizon
        self.lags         = lags or [1, 7, 14, 28]
        self.model_params = model_params or {
            "n_estimators": 300, "learning_rate": 0.05,
            "num_leaves": 31, "verbose": -1,
        }
        self.models_: dict = {}
        self.train_series_: Optional[np.ndarray] = None

    def _build_lag_features(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_lag = max(self.lags)
        X, y = [], []
        for i in range(max_lag, len(series)):
            X.append([series[i - lag] for lag in self.lags])
            y.append(series[i])
        return np.array(X), np.array(y)

    def fit(self, series: np.ndarray) -> "MultiStepPipeline":
        self.train_series_ = series
        max_lag = max(self.lags)

        if self.strategy == "recursive":
            X, y = self._build_lag_features(series)
            model = LGBMRegressor(**self.model_params)
            model.fit(X, y)
            self.models_[0] = model

        elif self.strategy == "direct":
            for h in range(1, self.horizon + 1):
                n_valid = len(series) - max_lag - h + 1
                X_h = []
                y_h = []
                for i in range(max_lag, len(series) - h + 1):
                    X_h.append([series[i - lag] for lag in self.lags])
                    y_h.append(series[i + h - 1])
                model = LGBMRegressor(**self.model_params)
                model.fit(np.array(X_h), np.array(y_h))
                self.models_[h] = model

        elif self.strategy == "mimo":
            from sklearn.multioutput import MultiOutputRegressor
            X_rows, Y_rows = [], []
            for i in range(max_lag, len(series) - self.horizon + 1):
                X_rows.append([series[i - lag] for lag in self.lags])
                Y_rows.append([series[i + h] for h in range(self.horizon)])
            model = MultiOutputRegressor(LGBMRegressor(**self.model_params))
            model.fit(np.array(X_rows), np.array(Y_rows))
            self.models_[0] = model

        return self

    def predict(self) -> np.ndarray:
        series  = self.train_series_
        max_lag = max(self.lags)
        latest  = np.array([[series[-lag] for lag in self.lags]])

        if self.strategy == "recursive":
            from collections import deque
            buffer = deque(series[-max_lag:], maxlen=max_lag)
            preds  = []
            for _ in range(self.horizon):
                feat = np.array([[list(buffer)[-lag] for lag in self.lags]])
                p    = self.models_[0].predict(feat)[0]
                preds.append(p)
                buffer.append(p)
            return np.array(preds)

        elif self.strategy == "direct":
            return np.array([self.models_[h].predict(latest)[0]
                             for h in range(1, self.horizon + 1)])

        elif self.strategy == "mimo":
            return self.models_[0].predict(latest).flatten()

        raise ValueError(f"Unknown strategy: {self.strategy}")


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    series = 100 + np.cumsum(np.random.randn(500) * 0.5)

    for strat in ["recursive", "direct", "mimo"]:
        pipe = MultiStepPipeline(strategy=strat, horizon=12, lags=[1, 7, 14, 28])
        pipe.fit(series[:400])
        preds = pipe.predict()
        print(f"{strat:12s}: {preds.round(2)}")
```

---

*← [01 — Direct vs Recursive vs MIMO](./01_direct_vs_recursive_vs_MIMO.md) | [Module README](./README.md) | Next: [03 — Global vs Local Models](./03_global_vs_local_models.md) →*
