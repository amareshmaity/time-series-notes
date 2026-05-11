# 04 — N-BEATS & N-HiTS

> **Module**: 05 Deep Learning Models | **File**: 4 of 6
>
> N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series) is a pure deep learning forecasting model that doesn't use RNNs, convolutions, or attention — yet achieves state-of-the-art accuracy on M4 and M5. N-HiTS extends it for long-horizon efficiency.

---

## Table of Contents

1. [N-BEATS Core Idea](#1-n-beats-core-idea)
2. [Doubly Residual Stacking](#2-doubly-residual-stacking)
3. [Interpretable vs. Generic N-BEATS](#3-interpretable-vs-generic-n-beats)
4. [N-HiTS — Hierarchical Interpolation](#4-n-hits--hierarchical-interpolation)
5. [Implementation with Neuralforecast](#5-implementation-with-neuralforecast)

---

## 1. N-BEATS Core Idea

### 1.1 Design Philosophy

N-BEATS (Oreshkin et al., 2020) uses **only fully-connected layers** and achieves:
- No domain-specific inductive biases (no explicit trend/seasonal components)
- Interpretable decomposition via basis expansion
- State-of-the-art on M4 competition (18K series)

```
Input: lookback window x = [y(t-L), ..., y(t)]

Architecture:
  Stack of blocks, each producing:
    - backcast:  reconstructs part of the input (what the block "explains")
    - forecast:  contributes to the output prediction

Final output = sum of all block forecasts
```

### 1.2 Single N-BEATS Block

```
Input x ──→ FC ──→ FC ──→ FC ──→ FC ──→ θ (basis coefficients)
                                        │
                      ┌─────────────────┤
                      ▼                 ▼
               g_b(θ_b) = backcast   g_f(θ_f) = forecast
               (reconstructs input)  (contributes to prediction)
```

Each block learns:
- **Backcast**: `x̂ = Σⱼ θⱼᵇ · bⱼ(t)` — the part of the input it "explains"
- **Forecast**: `ŷ = Σⱼ θⱼᶠ · fⱼ(t)` — its contribution to the output

---

## 2. Doubly Residual Stacking

### 2.1 The Key Innovation

N-BEATS stacks blocks in a **doubly residual** pattern:

```
Stack 1 (Block 1):
  backcast₁, forecast₁ ← Block(x)
  residual₁ = x - backcast₁     ← pass unexplained part forward

Stack 1 (Block 2):
  backcast₂, forecast₂ ← Block(residual₁)
  residual₂ = residual₁ - backcast₂

...
Final forecast = Σ forecast_i     ← sum of all block contributions
```

```
Intuition:
  Block 1: "I can explain THIS part of the input" → subtracts it out
  Block 2: "I can explain part of what's left" → subtracts further
  Block k: "I can explain part of the remaining signal"
  
  Residual stack: each block sees only what previous blocks couldn't explain
  Forecast sum:   each block contributes what it predicts for those components
```

### 2.2 Within a Stack — Multiple Blocks

```
Stack 1 (Trend Stack):      Blocks use polynomial basis → learn trend
Stack 2 (Seasonality Stack): Blocks use Fourier basis → learn seasonality
Stack 3 (Generic Stack):     Blocks use learned basis → learn remaining patterns

Final: forecast = Σ(trend forecasts) + Σ(seasonal forecasts) + Σ(generic forecasts)
```

---

## 3. Interpretable vs. Generic N-BEATS

### 3.1 Generic N-BEATS

Uses a **learned** (unconstrained) basis — the model decides what patterns to decompose into:

```python
# Basis expansion with learned basis vectors
# θ has dimension d_theta (usually << lookback)
# Backcast: x̂ = V_b · θ   where V_b is a learned (lookback × d_theta) matrix
# Forecast: ŷ = V_f · θ   where V_f is a learned (horizon × d_theta) matrix
```

### 3.2 Interpretable N-BEATS

Uses **fixed mathematical basis**:

**Trend stack** — Polynomial basis:
```
Basis vectors: [1, t, t², t³, ..., t^p]  (p = polynomial degree)
Forecast = θ₀ + θ₁·t + θ₂·t² + ... + θₚ·tᵖ   ← smooth trend curve
```

**Seasonality stack** — Fourier basis:
```
Basis vectors: [1, cos(2πt/H), sin(2πt/H), cos(4πt/H), sin(4πt/H), ...]
Forecast = combination of sine/cosine → captures periodic patterns
```

This gives the model inherent interpretability:
- **Trend component** = sum of trend stack forecasts
- **Seasonal component** = sum of seasonality stack forecasts

---

## 4. N-HiTS — Hierarchical Interpolation

### 4.1 The Long-Horizon Problem

Standard N-BEATS uses the same lookback and output resolution for all stacks. For **long horizons** (H > 100), this is inefficient — most future information comes from low-frequency components:

```
Day 1: needs fine-grained daily patterns
Day 7: weekly pattern dominates
Day 30: monthly trend dominates
Day 365: yearly trend only

→ Different forecast horizons need different temporal resolutions
```

### 4.2 N-HiTS Solution

N-HiTS (Challu et al., 2023) uses **multi-rate sampling**:

```
Stack 1 (fast, high-frequency): 
  Sub-sample input by factor r₁=1 → use full resolution → forecast short term

Stack 2 (medium):
  Sub-sample by r₂=4 → see slower patterns → forecast medium term

Stack 3 (slow, low-frequency):
  Sub-sample by r₃=16 → see only trend → forecast long term

Final: interpolate each stack's forecast to full output resolution → sum
```

**Result**: N-HiTS outperforms N-BEATS for long horizons (H > 48) while using fewer parameters.

---

## 5. Implementation with Neuralforecast

### 5.1 N-BEATS

```python
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

# Prepare data in Nixtla long format: columns [unique_id, ds, y]
def to_nixtla_format(series: pd.Series, unique_id: str = "series_1") -> pd.DataFrame:
    return pd.DataFrame({
        "unique_id": unique_id,
        "ds":        series.index,
        "y":         series.values,
    })

df_nf = to_nixtla_format(train_series)

# N-BEATS model
model_nbeats = NBEATS(
    h=12,                    # forecast horizon
    input_size=36,           # lookback window (3× horizon is a good start)
    stack_types=["trend", "seasonality", "identity"],   # interpretable
    n_blocks=[3, 3, 1],      # blocks per stack
    mlp_units=[[512, 512]] * 7,
    n_harmonics=2,           # Fourier harmonics for seasonality stack
    n_polynomials=2,         # polynomial degree for trend stack
    max_steps=500,
    batch_size=32,
    learning_rate=1e-3,
    val_check_steps=50,
    random_seed=42,
)

nf = NeuralForecast(models=[model_nbeats], freq="MS")
nf.fit(df=df_nf)
forecast_df = nf.predict()
print(forecast_df)

# Extract decomposed components (interpretable N-BEATS)
# nf.models[0].decomposition() — available in some versions
```

### 5.2 N-HiTS for Long Horizons

```python
from neuralforecast.models import NHITS

model_nhits = NHITS(
    h=96,                        # long horizon (96 steps)
    input_size=96 * 2,           # 2× horizon lookback
    stack_types=["identity", "identity", "identity"],
    n_blocks=[1, 1, 1],
    n_pool_kernel_size=[8, 4, 1],  # multi-rate sampling (8×, 4×, 1×)
    n_freq_downsample=[24, 12, 1],
    max_steps=1000,
    batch_size=32,
    learning_rate=1e-3,
    random_seed=42,
)

nf_nhits = NeuralForecast(models=[model_nhits], freq="H")
nf_nhits.fit(df=df_hourly)
forecast_nhits = nf_nhits.predict()
```

### 5.3 Cross-Validation with Neuralforecast

```python
# Walk-forward CV built into neuralforecast
cv_df = nf.cross_validation(
    df=df_nf,
    n_windows=5,           # number of validation windows
    step_size=12,          # advance origin by 12 each window
    refit=True,            # refit model at each origin
)
print(cv_df.head())

# Compute metrics
from neuralforecast.losses.numpy import mse, mae
for model_col in [c for c in cv_df.columns if c not in ["unique_id", "ds", "y", "cutoff"]]:
    rmse = np.sqrt(mse(cv_df["y"], cv_df[model_col]))
    print(f"  {model_col}: RMSE={rmse:.4f}")
```

### 5.4 N-BEATS vs. N-HiTS vs. LSTM

| Aspect | N-BEATS | N-HiTS | LSTM |
|--------|---------|--------|------|
| **Architecture** | FC + residuals | Multi-rate FC | Recurrent |
| **Short horizon** | Excellent | Good | Excellent |
| **Long horizon** | Good | Excellent | Moderate |
| **Interpretability** | ✅ Trend+Seasonal | ✅ Multi-scale | ❌ |
| **Covariates** | ❌ | ❌ (basic) | ✅ |
| **Training speed** | Fast | Fast | Moderate |
| **Data need** | Moderate | Moderate | Low |

---

*← [03 — Seq2Seq](./03_seq2seq_and_attention.md) | [Module README](./README.md) | Next: [05 — TFT](./05_temporal_fusion_transformer.md) →*
