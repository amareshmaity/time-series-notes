# 06 — Conformal Prediction for Time Series

> **Module**: 07 Forecasting Strategies | **File**: 6 of 6
>
> Conformal prediction provides **coverage-guaranteed** prediction intervals without any distributional assumptions. Unlike quantile regression (which can be miscalibrated) or Gaussian intervals (which assume normality), conformal intervals are guaranteed to contain the true value at the specified frequency — regardless of the underlying data distribution.

---

## Table of Contents

1. [Why Conformal Prediction?](#1-why-conformal-prediction)
2. [The Conformal Framework](#2-the-conformal-framework)
3. [Split Conformal Prediction](#3-split-conformal-prediction)
4. [Adaptive Conformal Inference (ACI)](#4-adaptive-conformal-inference-aci)
5. [EnbPI — Online Conformal for Time Series](#5-enbpi--online-conformal-for-time-series)
6. [Conformalized Quantile Regression (CQR)](#6-conformalized-quantile-regression-cqr)
7. [Evaluation and Coverage Diagnostics](#7-evaluation-and-coverage-diagnostics)
8. [Full Implementation Pipeline](#8-full-implementation-pipeline)

---

## 1. Why Conformal Prediction?

### 1.1 The Coverage Problem with Standard Intervals

```
Gaussian interval (requires normality):
  Actual coverage = 87% when nominal = 90%  ← miscalibrated if residuals are skewed

Quantile regression interval (requires large data for calibration):
  Actual coverage = 83% in tail regime  ← not guaranteed

Conformal interval (distribution-free):
  Actual coverage ≥ 90%  ← guaranteed by construction (marginal coverage guarantee)
```

### 1.2 Conformal vs. Other Interval Methods

| Method                  | Distribution-Free | Coverage Guarantee | Adapts to Complexity | Handles TS |
|-------------------------|-------------------|-------------------|----------------------|------------|
| Gaussian residuals      | ❌               | ❌ (approximate)  | ❌                   | Partial    |
| Quantile regression     | ✅               | ❌ (asymptotic)   | ✅                   | ✅         |
| Bootstrap               | ✅               | ❌ (approximate)  | Partial              | Partial    |
| **Split Conformal**     | ✅               | **✅ (finite-sample)** | ❌             | ⚠️ (IID assumption) |
| **Adaptive Conformal**  | ✅               | **✅ (long-run)**  | ✅                   | ✅         |
| **EnbPI**               | ✅               | **✅ (online)**    | ✅                   | ✅✅        |

---

## 2. The Conformal Framework

### 2.1 Core Idea

Conformal prediction calibrates a model's uncertainty using a **held-out calibration set**:

```
1. Fit model on training set
2. Compute nonconformity scores on calibration set:
     sᵢ = |yᵢ - ŷᵢ|  (absolute residual — simplest score)
3. Find the (1-α) quantile of calibration scores: q̂
4. For test point x*:
     Prediction Interval = [ŷ* - q̂,  ŷ* + q̂]

Guarantee: P(y* ∈ PI) ≥ 1 - α  (marginal, finite-sample)
```

### 2.2 Nonconformity Score

The nonconformity score `s(x, y)` measures how "strange" an observation is relative to the model. Common choices:

| Score                              | Formula                         | Best For                    |
|------------------------------------|---------------------------------|-----------------------------|
| Absolute residual                  | `|y - ŷ|`                       | Symmetric errors            |
| Signed residual                    | `y - ŷ`                         | Directional intervals       |
| Normalized residual                | `|y - ŷ| / σ̂(x)`              | Heteroscedastic data        |
| Conformalized Quantile Regression  | `max(q̂α - y, y - q̂_{1-α})`   | Asymmetric distributions    |

---

## 3. Split Conformal Prediction

### 3.1 Algorithm

```
Data split:
  Train      → fit model f
  Calibration → compute nonconformity scores
  Test        → apply conformal intervals

Step-by-step:
  1. Fit f on D_train
  2. For each (xᵢ, yᵢ) in D_cal:
       sᵢ = |yᵢ - f(xᵢ)|
  3. q̂ = quantile(s₁,...,s_n, level = ⌈(1-α)(n+1)⌉/n)
  4. For test point x*:
       ŷ* = f(x*)
       PI = [ŷ* - q̂,  ŷ* + q̂]
```

### 3.2 Implementation

```python
import numpy as np
import pandas as pd

def split_conformal_intervals(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split conformal prediction intervals with marginal coverage guarantee.

    Parameters
    ----------
    model            : sklearn-compatible model (already fitted or will be fitted here)
    X_train, y_train : training data
    X_cal, y_cal     : calibration data (held out from training)
    X_test           : test features
    alpha            : miscoverage rate (0.10 = 90% coverage target)

    Returns
    -------
    y_hat   : point predictions on test set
    lower   : lower bound of prediction interval
    upper   : upper bound of prediction interval
    """
    # Fit on training data only
    model.fit(X_train, y_train)

    # Nonconformity scores on calibration set
    cal_preds = model.predict(X_cal)
    scores    = np.abs(y_cal - cal_preds)   # absolute residuals

    # Conformal quantile — inflated slightly to ensure finite-sample guarantee
    n   = len(scores)
    lvl = np.ceil((1 - alpha) * (n + 1)) / n
    lvl = min(lvl, 1.0)
    q_hat = np.quantile(scores, lvl)

    # Test predictions
    y_hat = model.predict(X_test)
    lower = y_hat - q_hat
    upper = y_hat + q_hat

    print(f"Calibration nonconformity quantile (q̂): {q_hat:.4f}")
    print(f"Interval half-width: ±{q_hat:.4f}")

    return y_hat, lower, upper


def verify_split_conformal_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> dict:
    """Compute empirical coverage and check against nominal 1-α."""
    covered = ((y_true >= lower) & (y_true <= upper)).mean()
    return {
        "nominal_coverage":   1 - alpha,
        "empirical_coverage": float(covered),
        "marginal_guarantee_met": covered >= (1 - alpha - 0.02),   # allow 2% slack
    }
```

### 3.3 Limitation for Time Series

Split conformal requires **exchangeability** — observations in the calibration set should be i.i.d. (interchangeable). Time series data is **not** exchangeable due to temporal dependence. This means:

- Coverage guarantee holds approximately, not exactly
- For short-term forecasting: the approximation is often acceptable
- For longer horizons or nonstationary series: use Adaptive Conformal (Section 4) or EnbPI (Section 5)

---

## 4. Adaptive Conformal Inference (ACI)

### 4.1 Motivation

In non-stationary time series, the required interval width changes over time (e.g., wider in volatile regimes). ACI adapts `α` online to maintain target coverage:

```
ACI Update Rule (Gibbs & Candès, 2021):

αₜ₊₁ = αₜ + γ (α - errₜ)

Where:
  αₜ  = current miscoverage rate estimate
  γ   = step size (learning rate, e.g., 0.005–0.05)
  errₜ = 1{yₜ ∉ PIₜ}  (1 if prediction missed, 0 otherwise)
  α   = target miscoverage rate (e.g., 0.10 for 90% PI)

Intuition:
  If we missed coverage last step (err=1): decrease α → wider intervals
  If we covered last step (err=0): increase α → narrower intervals
  Long-run average coverage converges to 1-α
```

### 4.2 Implementation

```python
import numpy as np

class AdaptiveConformalForecaster:
    """
    Adaptive Conformal Inference for time series.

    Maintains online coverage guarantee by adapting the miscoverage rate αₜ
    at each step based on whether the previous interval covered the actual.

    Reference: Gibbs & Candès (2021) — "Adaptive Conformal Inference Under Distribution Shift"
    """

    def __init__(
        self,
        model,
        alpha: float = 0.10,
        gamma: float = 0.02,
        initial_scores: np.ndarray = None,
    ):
        """
        Parameters
        ----------
        model          : fitted point forecast model (sklearn API)
        alpha          : target miscoverage rate (1-alpha = target coverage)
        gamma          : adaptation step size (0.005–0.05 typical)
        initial_scores : nonconformity scores from a warm-up calibration period
        """
        self.model   = model
        self.alpha   = alpha
        self.gamma   = gamma
        self.alpha_t = alpha   # current adaptive alpha

        # Initialize calibration scores
        self.scores  = list(initial_scores) if initial_scores is not None else []
        self.history = []      # track (alpha_t, covered) for diagnostics

    def _get_quantile(self) -> float:
        """Current conformal quantile from accumulated nonconformity scores."""
        if not self.scores:
            return 0.0
        n   = len(self.scores)
        lvl = np.ceil((1 - self.alpha_t) * (n + 1)) / n
        return float(np.quantile(self.scores, min(lvl, 1.0)))

    def predict_interval(self, x: np.ndarray) -> tuple[float, float, float]:
        """
        Predict point estimate and adaptive conformal interval for one observation.

        Returns
        -------
        y_hat, lower, upper
        """
        y_hat = float(self.model.predict(x.reshape(1, -1))[0])
        q_hat = self._get_quantile()
        return y_hat, y_hat - q_hat, y_hat + q_hat

    def update(self, x: np.ndarray, y_true: float) -> dict:
        """
        Update calibration scores and adaptive alpha after observing true value.

        Call this after each new observation arrives in production.
        """
        y_hat, lower, upper = self.predict_interval(x)
        score   = abs(y_true - y_hat)
        covered = lower <= y_true <= upper

        # Update alpha using ACI rule
        err          = 0 if covered else 1
        self.alpha_t = np.clip(self.alpha_t + self.gamma * (self.alpha - err), 0.01, 0.99)
        self.scores.append(score)

        record = {"y_true": y_true, "y_hat": y_hat,
                  "lower": lower, "upper": upper,
                  "covered": covered, "alpha_t": self.alpha_t}
        self.history.append(record)
        return record

    def rolling_coverage(self, window: int = 50) -> float:
        """Compute rolling empirical coverage over the last `window` steps."""
        if len(self.history) < window:
            return float(np.mean([h["covered"] for h in self.history]))
        recent = self.history[-window:]
        return float(np.mean([h["covered"] for h in recent]))

    def diagnostics(self) -> pd.DataFrame:
        """Return full history as DataFrame for analysis."""
        return pd.DataFrame(self.history)
```

---

## 5. EnbPI — Online Conformal for Time Series

### 5.1 What is EnbPI?

**Ensemble Bootstrap Prediction Intervals (EnbPI)** (Xu & Xie, 2021) is the state-of-the-art online conformal method designed explicitly for time series. It:

1. Trains an ensemble of bootstrap models on overlapping windows
2. Computes leave-one-out-style nonconformity scores
3. Updates intervals **online** as new data arrives — no full retrain needed

```
Key innovation: uses a sliding window of recent nonconformity scores
rather than the full calibration set, making it adaptive to distributional shifts.

Update at time t:
  1. Make prediction ŷₜ
  2. Observe yₜ
  3. Add sₜ = |yₜ - ŷₜ| to recent score buffer
  4. Remove oldest score from buffer (sliding window)
  5. Recompute q̂ from updated buffer → new interval for t+1
```

### 5.2 EnbPI Simplified Implementation

```python
from collections import deque
import numpy as np

class EnbPIForecaster:
    """
    Simplified EnbPI-style online conformal predictor for time series.

    Uses a sliding window of recent nonconformity scores for adaptive calibration.
    Suitable for production use — updates in O(1) per step.

    Reference: Xu & Xie (2021) — "Conformal Prediction Interval for Dynamic Time-Series"
    """

    def __init__(
        self,
        model,
        window_size: int = 100,
        alpha: float = 0.10,
    ):
        """
        Parameters
        ----------
        model       : fitted point forecast model (sklearn API)
        window_size : number of recent scores retained (sliding window)
        alpha       : miscoverage rate (0.10 = 90% PI)
        """
        self.model       = model
        self.window_size = window_size
        self.alpha       = alpha
        self.score_buffer: deque = deque(maxlen=window_size)
        self.history     = []

    def warm_up(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Initialize the score buffer using a calibration set.
        Call this once before starting online prediction.
        """
        cal_preds = self.model.predict(X_cal)
        scores    = np.abs(y_cal - cal_preds)
        for s in scores[-self.window_size:]:   # use most recent scores
            self.score_buffer.append(float(s))
        print(f"EnbPI warmed up with {len(self.score_buffer)} calibration scores")

    def _current_quantile(self) -> float:
        """Current q̂ from sliding window of nonconformity scores."""
        if not self.score_buffer:
            return 0.0
        scores = np.array(self.score_buffer)
        n      = len(scores)
        lvl    = np.ceil((1 - self.alpha) * (n + 1)) / n
        return float(np.quantile(scores, min(lvl, 1.0)))

    def predict(self, x: np.ndarray) -> dict:
        """Predict with current conformal interval (before observing true value)."""
        y_hat = float(self.model.predict(x.reshape(1, -1))[0])
        q_hat = self._current_quantile()
        return {"y_hat": y_hat, "lower": y_hat - q_hat,
                "upper": y_hat + q_hat, "q_hat": q_hat}

    def update(self, x: np.ndarray, y_true: float) -> dict:
        """Predict, then update the score buffer after observing true value."""
        result = self.predict(x)
        score  = abs(y_true - result["y_hat"])
        self.score_buffer.append(score)   # sliding window auto-removes oldest

        result.update({
            "y_true":  y_true,
            "score":   score,
            "covered": result["lower"] <= y_true <= result["upper"],
        })
        self.history.append(result)
        return result

    def coverage_report(self) -> dict:
        """Compute empirical coverage and interval width statistics."""
        if not self.history:
            return {}
        df = pd.DataFrame(self.history)
        return {
            "empirical_coverage": df["covered"].mean(),
            "nominal_coverage":   1 - self.alpha,
            "mean_interval_width": (df["upper"] - df["lower"]).mean(),
            "n_observations":      len(df),
        }
```

---

## 6. Conformalized Quantile Regression (CQR)

### 6.1 Motivation

Standard split conformal uses a **symmetric** interval `[ŷ - q̂, ŷ + q̂]`. CQR combines quantile regression with conformal calibration to produce **asymmetric, data-adaptive** intervals:

```
Step 1: Train quantile regression models for τ_low and τ_high
          q̂_low(x)  = τ_low  quantile model
          q̂_high(x) = τ_high quantile model

Step 2: Compute CQR nonconformity scores on calibration set:
          sᵢ = max(q̂_low(xᵢ) - yᵢ,  yᵢ - q̂_high(xᵢ))
          (positive = actual is outside the quantile band)

Step 3: Conformal quantile of CQR scores:
          q̂_CQR = quantile(s₁,...,s_n, 1-α)

Step 4: Test interval:
          PI = [q̂_low(x*) - q̂_CQR,  q̂_high(x*) + q̂_CQR]
```

### 6.2 Implementation

```python
import numpy as np
from lightgbm import LGBMRegressor

class CQRForecaster:
    """
    Conformalized Quantile Regression forecaster.

    Combines quantile regression (asymmetric, data-adaptive intervals)
    with conformal calibration (coverage guarantee).

    Reference: Romano, Patterson & Candès (2019) — "Conformalized Quantile Regression"
    """

    def __init__(self, alpha: float = 0.10, **lgbm_kwargs):
        self.alpha      = alpha
        self.tau_low    = alpha / 2
        self.tau_high   = 1 - alpha / 2
        self.lgbm_kw    = lgbm_kwargs or {"n_estimators": 300, "learning_rate": 0.05,
                                           "num_leaves": 31, "verbose": -1}
        self.model_low  = LGBMRegressor(objective="quantile", alpha=self.tau_low,  **self.lgbm_kw)
        self.model_high = LGBMRegressor(objective="quantile", alpha=self.tau_high, **self.lgbm_kw)
        self.q_cqr_     = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_cal: np.ndarray, y_cal: np.ndarray) -> "CQRForecaster":
        """
        Train quantile models and calibrate using held-out calibration set.
        """
        self.model_low.fit(X_train, y_train)
        self.model_high.fit(X_train, y_train)

        # CQR nonconformity scores on calibration set
        q_low_cal  = self.model_low.predict(X_cal)
        q_high_cal = self.model_high.predict(X_cal)
        scores     = np.maximum(q_low_cal - y_cal, y_cal - q_high_cal)

        n         = len(scores)
        lvl       = np.ceil((1 - self.alpha) * (n + 1)) / n
        self.q_cqr_ = float(np.quantile(scores, min(lvl, 1.0)))
        print(f"CQR conformal adjustment (q̂_CQR): {self.q_cqr_:.4f}")
        return self

    def predict(self, X_test: np.ndarray) -> pd.DataFrame:
        """
        Generate CQR prediction intervals for test set.

        Returns
        -------
        DataFrame with columns: q_low, q_high, lower (CQR), upper (CQR)
        """
        assert self.q_cqr_ is not None, "Call .fit() first"
        q_low  = self.model_low.predict(X_test)
        q_high = self.model_high.predict(X_test)
        return pd.DataFrame({
            "q_low":  q_low,
            "q_high": q_high,
            "lower":  q_low  - self.q_cqr_,
            "upper":  q_high + self.q_cqr_,
        })
```

---

## 7. Evaluation and Coverage Diagnostics

```python
import numpy as np
import matplotlib.pyplot as plt

def coverage_diagnostics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    title: str = "Conformal Prediction Interval Diagnostics",
) -> dict:
    """
    Full diagnostics for a prediction interval method.

    Checks:
    - Empirical coverage (should be ≥ 1-alpha)
    - Interval width distribution
    - Coverage over time (should be stable, not drift)
    """
    covered = (y_true >= lower) & (y_true <= upper)
    widths  = upper - lower

    results = {
        "empirical_coverage":  float(covered.mean()),
        "nominal_coverage":    1 - alpha,
        "mean_width":          float(widths.mean()),
        "median_width":        float(np.median(widths)),
        "coverage_guaranteed": bool(covered.mean() >= 1 - alpha - 0.01),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Rolling coverage
    roll = pd.Series(covered.astype(float)).rolling(50, min_periods=10).mean()
    axes[0].plot(roll, color="steelblue")
    axes[0].axhline(1 - alpha, color="red", linestyle="--", label=f"Target {1-alpha:.0%}")
    axes[0].set_title("Rolling 50-Step Coverage")
    axes[0].set_ylabel("Coverage")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Interval widths
    axes[1].plot(widths, color="darkorange", alpha=0.7)
    axes[1].set_title("Prediction Interval Width Over Time")
    axes[1].set_ylabel("Width")
    axes[1].grid(alpha=0.3)

    # 3. Coverage bar
    axes[2].bar(["Nominal", "Empirical"],
                [1 - alpha, covered.mean()],
                color=["#2196F3", "#4CAF50" if covered.mean() >= 1 - alpha else "#F44336"])
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Coverage Summary")
    axes[2].set_ylabel("Coverage Rate")
    for i, v in enumerate([1 - alpha, covered.mean()]):
        axes[2].text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()

    return results
```

---

## 8. Full Implementation Pipeline

```python
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

# ── Generate synthetic non-stationary series ─────────────────────────────────
np.random.seed(42)
n      = 600
trend  = np.linspace(0, 20, n)
seas   = 10 * np.sin(2 * np.pi * np.arange(n) / 52)
noise  = np.random.normal(0, 1 + np.linspace(0, 3, n))  # increasing volatility
series = trend + seas + noise

# ── Feature matrix ────────────────────────────────────────────────────────────
lags = [1, 2, 3, 7, 14]
X, y = [], []
for i in range(max(lags), len(series)):
    X.append([series[i - lag] for lag in lags])
    y.append(series[i])

X = np.array(X)
y = np.array(y)

# Time-ordered splits: train | calibration | test
n_all  = len(X)
n_test = 100
n_cal  = 100

X_train, y_train = X[:n_all - n_test - n_cal], y[:n_all - n_test - n_cal]
X_cal,   y_cal   = X[n_all - n_test - n_cal:n_all - n_test], y[n_all - n_test - n_cal:n_all - n_test]
X_test,  y_test  = X[n_all - n_test:], y[n_all - n_test:]

# ── Fit base model ────────────────────────────────────────────────────────────
base_model = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                           num_leaves=31, verbose=-1)
base_model.fit(X_train, y_train)

# ── Split Conformal ───────────────────────────────────────────────────────────
_, sc_lower, sc_upper = split_conformal_intervals(
    base_model, X_train, y_train, X_cal, y_cal, X_test, alpha=0.10
)
sc_cov = verify_split_conformal_coverage(y_test, sc_lower, sc_upper, 0.10)
print("Split Conformal:", sc_cov)

# ── Conformalized Quantile Regression ────────────────────────────────────────
cqr = CQRForecaster(alpha=0.10)
cqr.fit(X_train, y_train, X_cal, y_cal)
cqr_preds  = cqr.predict(X_test)

cqr_cov = coverage_diagnostics(
    y_test, cqr_preds["lower"].values, cqr_preds["upper"].values,
    alpha=0.10, title="CQR Coverage Diagnostics"
)
print("CQR:", cqr_cov)

# ── EnbPI (online, sliding window) ───────────────────────────────────────────
enbpi = EnbPIForecaster(base_model, window_size=80, alpha=0.10)
enbpi.warm_up(X_cal, y_cal)

for i in range(len(X_test)):
    enbpi.update(X_test[i], y_test[i])

print("EnbPI:", enbpi.coverage_report())
```

### Key Takeaways

| Method           | Coverage Type        | Adaptivity   | Best For                               |
|------------------|----------------------|--------------|----------------------------------------|
| Split Conformal  | Marginal (exact)     | ❌ Static    | Stationary series, quick baseline      |
| ACI              | Long-run (exact)     | ✅ Online α  | Nonstationary, distribution shifts     |
| EnbPI            | Online (sliding win) | ✅ Sliding W | Production streaming; concept drift    |
| CQR              | Marginal (exact)     | ✅ Asymmetric| Asymmetric errors; skewed distributions |

---

*← [05 — Probabilistic Forecasting](./05_probabilistic_forecasting.md) | [Module README](./README.md)*
