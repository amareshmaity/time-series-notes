# 01 — Direct vs. Recursive vs. MIMO Forecasting

> **Module**: 07 Forecasting Strategies | **File**: 1 of 6
>
> How you generate a multi-step forecast matters as much as *which* model you use. This note covers the three fundamental strategies — Direct, Recursive, and MIMO — and the engineering trade-offs between them.

---

## Table of Contents

1. [The Multi-Step Forecasting Problem](#1-the-multi-step-forecasting-problem)
2. [Recursive Strategy](#2-recursive-strategy)
3. [Direct Strategy](#3-direct-strategy)
4. [MIMO Strategy](#4-mimo-strategy)
5. [DIRMO — The Hybrid](#5-dirmo--the-hybrid)
6. [Strategy Comparison](#6-strategy-comparison)
7. [Choosing the Right Strategy](#7-choosing-the-right-strategy)

---

## 1. The Multi-Step Forecasting Problem

Given a time series `y₁, y₂, ..., yₜ`, the goal is to forecast the next `H` steps:

```
ŷₜ₊₁, ŷₜ₊₂, ..., ŷₜ₊ₕ
```

This is straightforward for `H = 1` (one-step-ahead), but multi-step forecasting (`H > 1`) requires a deliberate strategy. Three canonical approaches exist:

| Strategy   | Models | Trains on Future Targets | Error Accumulation |
|------------|--------|--------------------------|--------------------|
| Recursive  | 1      | ❌ (uses own predictions) | ✅ High             |
| Direct     | H      | ✅ (shifted targets)      | ❌ None             |
| MIMO       | 1      | ✅ (all H steps at once)  | ❌ None             |

---

## 2. Recursive Strategy

### 2.1 How It Works

Train a single **one-step-ahead** model. At inference time, feed predictions back as inputs:

```
Step 1:  ŷₜ₊₁ = f(yₜ, yₜ₋₁, ..., yₜ₋ₗ)
Step 2:  ŷₜ₊₂ = f(ŷₜ₊₁, yₜ, ..., yₜ₋ₗ₊₁)   ← uses predicted value
Step 3:  ŷₜ₊₃ = f(ŷₜ₊₂, ŷₜ₊₁, ..., yₜ₋ₗ₊₂) ← errors compound further
...
Step H:  ŷₜ₊ₕ = f(ŷₜ₊ₕ₋₁, ..., ŷₜ₊ₕ₋ₗ)
```

**Error accumulation**: Each predicted value is an imperfect estimate. When fed back as a lag feature, its error is treated as ground truth by the next step — errors compound multiplicatively.

### 2.2 Mathematical Bias

Let `ε` be the one-step prediction error. By step `h`:
```
Var(ŷₜ₊ₕ) ≈ Var(yₜ₊ₕ) + h · Var(ε) + O(ε²)

→ Uncertainty grows linearly with horizon h
→ For long horizons, variance can exceed the signal itself
```

### 2.3 Implementation

```python
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

def build_lag_features(series: np.ndarray, lags: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Build lag feature matrix and aligned target for one-step-ahead training."""
    max_lag = max(lags)
    X, y = [], []
    for i in range(max_lag, len(series)):
        X.append([series[i - lag] for lag in lags])
        y.append(series[i])
    return np.array(X), np.array(y)


def recursive_forecast(model, history: np.ndarray, lags: list[int], h: int) -> np.ndarray:
    """
    Recursive multi-step forecast.

    Parameters
    ----------
    model   : trained sklearn-compatible regressor (one-step-ahead)
    history : full observed series as 1D array
    lags    : list of lag indices used during training (e.g., [1, 2, 7])
    h       : forecast horizon

    Returns
    -------
    predictions : 1D array of h forecasts
    """
    buffer = list(history)   # mutable buffer — will grow with predictions
    predictions = []

    for _ in range(h):
        # Build features from the latest observations (real + predicted)
        features = np.array([[buffer[-lag] for lag in lags]])
        pred = model.predict(features)[0]
        predictions.append(pred)
        buffer.append(pred)  # feed prediction back into buffer

    return np.array(predictions)


# ── Training ──────────────────────────────────────────────────────────────────
lags = [1, 2, 3, 7, 14]

# Synthetic daily series
np.random.seed(42)
series = np.cumsum(np.random.randn(500)) + 100

split = 400
train_series = series[:split]
test_series  = series[split:]

X_train, y_train = build_lag_features(train_series, lags)
model = LGBMRegressor(n_estimators=200, learning_rate=0.05, verbose=-1)
model.fit(X_train, y_train)

# ── Forecasting ───────────────────────────────────────────────────────────────
H = 30
recursive_preds = recursive_forecast(model, train_series, lags, h=H)
print("Recursive forecasts (first 5):", recursive_preds[:5].round(2))
```

### 2.4 Pros & Cons

| ✅ Pros                                    | ❌ Cons                                        |
|--------------------------------------------|------------------------------------------------|
| Single model — minimal memory              | Error accumulates with each step               |
| Simple training (standard one-step data)   | Unreliable for `H > 12`                        |
| Works with any single-output regressor     | Confidence intervals are hard to compute       |
| Natural fit for ARIMA-family models        | Lag features become stale at long horizons     |

---

## 3. Direct Strategy

### 3.1 How It Works

Train **one model per horizon step**. Each model `fₕ` is trained to predict `yₜ₊ₕ` directly from observed past values:

```
f₁ : yₜ₊₁ = f₁(yₜ, yₜ₋₁, ...) — trained on (Xₜ, yₜ₊₁) pairs
f₂ : yₜ₊₂ = f₂(yₜ, yₜ₋₁, ...) — trained on (Xₜ, yₜ₊₂) pairs
...
fₕ : yₜ₊ₕ = fₕ(yₜ, yₜ₋₁, ...) — trained on (Xₜ, yₜ₊ₕ) pairs
```

**Key insight**: Features are always constructed from real observed values (no predictions as inputs), so no error accumulation.

### 3.2 Target Construction

```python
import pandas as pd

def build_direct_targets(df: pd.DataFrame, target_col: str, horizon: int) -> dict[int, pd.DataFrame]:
    """
    Create a dictionary of (X, y_h) DataFrames for Direct strategy.

    For each horizon step h, the target y_h = target shifted back by h steps.
    Features remain the same lag-based features built from observed history.

    Parameters
    ----------
    df         : DataFrame with lag features already computed (no future leakage)
    target_col : name of the target column
    horizon    : maximum forecast horizon H

    Returns
    -------
    dict mapping h → DataFrame with feature columns + 'target_h' column
    """
    datasets = {}
    for h in range(1, horizon + 1):
        df_h = df.copy()
        df_h[f"target_{h}"] = df_h[target_col].shift(-h)  # shift target forward by h
        df_h = df_h.dropna()
        datasets[h] = df_h
    return datasets
```

### 3.3 Training and Forecasting

```python
from lightgbm import LGBMRegressor
import numpy as np

def train_direct_models(datasets: dict, feature_cols: list[str], horizon: int) -> dict:
    """Train one LightGBM model per horizon step."""
    models = {}
    for h in range(1, horizon + 1):
        df_h = datasets[h]
        X_h = df_h[feature_cols]
        y_h = df_h[f"target_{h}"]

        split = int(len(X_h) * 0.8)
        model = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, verbose=-1)
        model.fit(X_h.iloc[:split], y_h.iloc[:split])
        models[h] = model
        print(f"  Horizon h={h:02d}: trained on {split} samples")

    return models


def direct_forecast(models: dict, X_latest: pd.DataFrame, horizon: int) -> np.ndarray:
    """
    Generate multi-step direct forecast.

    Each model predicts its specific horizon step independently.
    Features come from the LAST observed row — no predictions as inputs.
    """
    return np.array([models[h].predict(X_latest)[0] for h in range(1, horizon + 1)])
```

### 3.4 Pros & Cons

| ✅ Pros                                       | ❌ Cons                                          |
|-----------------------------------------------|--------------------------------------------------|
| Zero error accumulation                       | `H` models to train, store, and maintain         |
| Each model optimized for its specific horizon | Models ignore inter-step dependencies            |
| More accurate at long horizons                | High memory footprint for large `H`              |
| Prediction intervals per horizon step         | Training data shrinks as `h` increases           |

---

## 4. MIMO Strategy

### 4.1 How It Works

**Multi-Input Multi-Output (MIMO)**: A single model predicts all `H` future steps simultaneously as a vector output:

```
[ŷₜ₊₁, ŷₜ₊₂, ..., ŷₜ₊ₕ] = f(yₜ, yₜ₋₁, ..., yₜ₋ₗ)
```

The model output is a vector of length `H`. This preserves **inter-step correlations** — the model learns that `ŷₜ₊₃` is related to `ŷₜ₊₁` through the joint distribution.

### 4.2 Target Construction

```python
import pandas as pd
import numpy as np

def build_mimo_targets(series: np.ndarray, lags: list[int], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, Y) for MIMO forecasting.

    X : (n_samples, n_features) — lag features from observed values
    Y : (n_samples, horizon)    — matrix of H future values as targets
    """
    max_lag = max(lags)
    X_rows, Y_rows = [], []

    for i in range(max_lag, len(series) - horizon + 1):
        features = [series[i - lag] for lag in lags]
        targets  = [series[i + h] for h in range(horizon)]
        X_rows.append(features)
        Y_rows.append(targets)

    return np.array(X_rows), np.array(Y_rows)
```

### 4.3 Implementation Options

```python
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
import numpy as np

# ── Option A: MultiOutputRegressor wrapper (independent models per output) ──
multi_lgbm = MultiOutputRegressor(
    LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1),
    n_jobs=-1,
)

lags    = [1, 2, 3, 7, 14]
horizon = 12

np.random.seed(42)
series = np.cumsum(np.random.randn(500)) + 100

X, Y = build_mimo_targets(series, lags, horizon)
split = int(len(X) * 0.8)

multi_lgbm.fit(X[:split], Y[:split])
mimo_preds = multi_lgbm.predict(X[split:split+1]).flatten()   # shape: (12,)
print("MIMO 12-step forecast:", mimo_preds.round(2))

# ── Option B: Neural network (joint output — true MIMO) ──────────────────────
# Use seq2seq, N-BEATS, NHITS, or TFT — these natively output H-step vectors
# (see Module 05 for implementation)
```

### 4.4 Pros & Cons

| ✅ Pros                                               | ❌ Cons                                          |
|-------------------------------------------------------|--------------------------------------------------|
| Single model — simpler deployment                     | Higher model complexity                          |
| No error accumulation                                 | Tree-based MIMO ignores output correlations      |
| Preserves joint distribution of future steps          | Requires redesigned target matrix                |
| Best with neural nets (seq2seq, TFT)                  | Harder to add prediction intervals               |

---

## 5. DIRMO — The Hybrid

**DIRMO** (Direct MIMO) splits the horizon into `G` groups and applies MIMO within each group:

```
Group 1: model₁ predicts [ŷₜ₊₁, ..., ŷₜ₊ₘ]
Group 2: model₂ predicts [ŷₜ₊ₘ₊₁, ..., ŷₜ₊₂ₘ]
...
Group G: model_G predicts [ŷₜ₊(G-1)m+1, ..., ŷₜ₊ₕ]
```

Where `H = G × m`. It trades off the number of models vs. error accumulation:

```python
def train_dirmo_models(
    series: np.ndarray,
    lags: list[int],
    horizon: int,
    group_size: int,
) -> list:
    """
    DIRMO: train one multi-output model per group of horizon steps.

    Parameters
    ----------
    series     : full observed time series
    lags       : lag indices for feature construction
    horizon    : total forecast horizon H (must be divisible by group_size)
    group_size : number of steps per group (m)

    Returns
    -------
    list of trained models, one per group
    """
    assert horizon % group_size == 0, "horizon must be divisible by group_size"
    n_groups = horizon // group_size
    models = []

    for g in range(n_groups):
        # Offset targets: group g predicts steps [g*m+1 .. (g+1)*m]
        offset = g * group_size
        X_rows, Y_rows = [], []
        max_lag = max(lags)
        for i in range(max_lag, len(series) - horizon + 1):
            features = [series[i - lag] for lag in lags]
            targets  = [series[i + offset + s] for s in range(group_size)]
            X_rows.append(features)
            Y_rows.append(targets)

        X = np.array(X_rows)
        Y = np.array(Y_rows)
        split = int(len(X) * 0.8)

        model = MultiOutputRegressor(
            LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1)
        )
        model.fit(X[:split], Y[:split])
        models.append(model)
        print(f"  Group {g+1}/{n_groups}: predicts steps {offset+1}–{offset+group_size}")

    return models
```

---

## 6. Strategy Comparison

| Dimension                  | Recursive     | Direct      | MIMO          | DIRMO         |
|----------------------------|---------------|-------------|---------------|---------------|
| **Number of models**       | 1             | H           | 1             | H/m           |
| **Error accumulation**     | High          | None        | None          | Minimal       |
| **Inter-step correlation** | Implicit      | Ignored     | Preserved     | Partial       |
| **Training data size**     | Full          | Shrinks at high h | Full   | Full          |
| **Memory at inference**    | Low           | High        | Low           | Medium        |
| **Best for short horizon** | ✅            | ✅          | ✅            | ✅            |
| **Best for long horizon**  | ❌            | ✅          | ✅            | ✅            |
| **Works with any model**   | ✅            | ✅          | Needs MO API  | Needs MO API  |

### Empirical findings (Taieb & Atiya, 2016)

- **Short horizons (H ≤ 6)**: Recursive is competitive; its bias is small
- **Medium horizons (6 < H ≤ 24)**: MIMO or DIRMO outperform Recursive
- **Long horizons (H > 24)**: Direct or MIMO consistently outperform Recursive
- **Neural networks**: MIMO is the standard — seq2seq, TFT, NHITS all use MIMO internally

---

## 7. Choosing the Right Strategy

```
                     ┌─────────────────────────────┐
                     │   What is your horizon H?   │
                     └──────────┬──────────────────┘
                                │
              ┌─────────────────┼──────────────────┐
              ▼                 ▼                  ▼
          H ≤ 6             6 < H ≤ 24          H > 24
          │                 │                   │
    Recursive OK      MIMO or DIRMO         Direct or MIMO
    (simple, fast)    (best tradeoff)       (no accumulation)
```

**Decision checklist**:

1. **Computational budget**:
   - Tight → Recursive (1 model) or MIMO (1 model)
   - Ample → Direct (H models, most accurate per step)

2. **Model type**:
   - Neural network (LSTM, TFT, N-BEATS) → MIMO built-in
   - Tree-based (XGBoost, LightGBM) → Recursive or Direct

3. **Horizon length**:
   - Short (≤ 6) → Any strategy works
   - Long (> 12) → Avoid Recursive

4. **Deployment constraints**:
   - Low memory → Recursive or MIMO (1 model)
   - Per-step calibration needed → Direct (each model tuned per horizon)

---

*← [Module README](./README.md) | Next: [02 — Multi-Step Forecasting](./02_multi_step_forecasting.md) →*
