# 05 — Temporal Fusion Transformer (TFT)

> **Module**: 05 Deep Learning Models | **File**: 5 of 6
>
> TFT (Lim et al., 2021) is the most comprehensive production-grade deep learning forecaster — handling static metadata, known future covariates, past-only covariates, temporal attention, and prediction intervals in one unified architecture.

---

## Table of Contents

1. [TFT Overview and Motivation](#1-tft-overview-and-motivation)
2. [Input Types](#2-input-types)
3. [Key Components](#3-key-components)
4. [Architecture Flow](#4-architecture-flow)
5. [Interpretability Features](#5-interpretability-features)
6. [Implementation with Neuralforecast](#6-implementation-with-neuralforecast)

---

## 1. TFT Overview and Motivation

### 1.1 What Makes TFT Different

Most DL forecasters handle only past observations. Real production forecasting often has:
- **Static metadata** — store region, product category (time-invariant)
- **Known future covariates** — promotions, holidays, prices (known in advance)
- **Past-only covariates** — past sales of related products (historical only)
- **Multi-horizon** output with calibrated prediction intervals

**TFT handles all of these simultaneously** in a single model.

### 1.2 TFT vs. Other DL Models

| Capability | LSTM | N-BEATS | TFT |
|-----------|------|---------|-----|
| Static metadata | Manual encoding | ❌ | ✅ Native VSN |
| Known future covariates | Manual | ❌ | ✅ Native |
| Historical covariates | ✅ | ❌ | ✅ Native |
| Prediction intervals | ❌ | ✅ (quantile) | ✅ Quantile + calibrated |
| Attention weights | ❌ | ❌ | ✅ Interpretable |
| Variable importance | ❌ | ❌ | ✅ VSN weights |

---

## 2. Input Types

```python
# TFT distinguishes 4 types of inputs:

static_categoricals   = ["store_id", "category", "region"]
# Time-invariant; same for all timesteps of a series
# Examples: store type, product category, country

static_reals          = ["store_size_sqft", "avg_price_level"]
# Time-invariant continuous features

time_varying_known    = ["day_of_week", "is_holiday", "promo_flag", "price"]
# Vary over time; known IN ADVANCE for future dates
# Examples: calendar features, scheduled promotions

time_varying_unknown  = ["sales", "returns", "inventory_level"]
# Vary over time; observed historically but NOT known in future
# The TARGET is always in this category
```

---

## 3. Key Components

### 3.1 Variable Selection Networks (VSN)

VSN learns to **weight each input variable's importance** using a softmax-gated mechanism:

```
For each input variable v:
  1. Pass through individual FC layer → ξᵥ
  2. Concatenate all ξᵥ → pass through FC → softmax weights pᵥ
  3. Output: Σᵥ pᵥ · ξᵥ   (weighted combination)

→ VSN learns which variables are important at which timesteps
→ Variable importance weights are interpretable
```

### 3.2 Gated Residual Network (GRN)

The core building block — a gated skip connection with ELU activation:

```
GRN(x, c=None):
  η₁ = LayerNorm(x + GLU(fc₁(ELU(fc₀(x) + fc_c(c))) + fc_skip(x)))
  
Where:
  c   = optional context input (static covariate embedding)
  GLU = Gated Linear Unit: σ(a) ⊙ b   (splits linear into two halves)
  
Purpose: Suppresses unnecessary components via gating (learns what to pass through)
```

### 3.3 LSTM Encoder (Past Context)

Processes past observations and unknown covariates:

```
Inputs: past time_varying_unknown + past time_varying_known
→ LSTM → produces sequence of hidden states h̃(t) for t ≤ 0
```

### 3.4 LSTM Decoder (Future Context)

Processes known future covariates:

```
Inputs: future time_varying_known (promo, holidays, prices, calendar)
→ LSTM → produces sequence h̃(t) for t > 0
```

### 3.5 Multi-Head Self-Attention (Temporal Attention)

The attention layer operates on the concatenated encoder + decoder hidden states:

```
Q = K = V = [h̃_enc ; h̃_dec]   (concatenated past + future hidden states)
Attention(Q, K, V) = softmax(QKᵀ / √d_model) · V

→ Each output position attends to all input positions
→ Long-range dependencies captured across the full lookback window
→ Attention weights show which past timesteps influenced each future step
```

---

## 4. Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Static Inputs → Embedding → VSN → Context vectors c_s,c_e,c_h,c_c │
└─────────────────────────────────┬───────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────┐
│  Past inputs (t ≤ 0):                                        │
│  time_varying_unknown + time_varying_known                    │
│  → VSN → Locality Enhancement (LSTM Encoder) → h̃_enc        │
└──────────────────────────────────────────────────────────────┘
                                  +
┌──────────────────────────────────────────────────────────────┐
│  Future inputs (t > 0):                                      │
│  time_varying_known only                                      │
│  → VSN → Locality Enhancement (LSTM Decoder) → h̃_dec        │
└──────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────┐
│  Concatenate [h̃_enc ; h̃_dec]  │
│  → GRN → Multi-Head Attention   │
│  → Gated Add & Norm             │
│  → GRN × position-wise          │
│  → Quantile FC Layers           │
│  → [q10, q50, q90] per horizon  │
└─────────────────────────────────┘
```

---

## 5. Interpretability Features

### 5.1 Variable Importance Scores

TFT's VSN produces interpretable variable weights:

```python
# After training
importance = model.get_variable_importance()
print("Static variable importance:")
for name, weight in sorted(importance["static"].items(), key=lambda x: -x[1]):
    print(f"  {name:<25}: {weight:.4f}")

print("\nPast variable importance:")
for name, weight in sorted(importance["encoder"].items(), key=lambda x: -x[1]):
    print(f"  {name:<25}: {weight:.4f}")

print("\nFuture variable importance:")
for name, weight in sorted(importance["decoder"].items(), key=lambda x: -x[1]):
    print(f"  {name:<25}: {weight:.4f}")
```

### 5.2 Temporal Attention Patterns

The attention heatmap shows **which past timesteps** the model focuses on when making each future prediction:

```
Attention weights: (n_heads, forecast_horizon, lookback_window)

Visualization:
  x-axis: lookback time steps (past)
  y-axis: forecast time steps (future)
  color:  attention weight (brighter = more attention)

Common patterns:
  - Diagonal band: recent past is most relevant (short-range dependence)
  - Seasonal peaks: same period last year/week gets high attention
  - Periodic spikes: at multiples of the seasonal period
```

---

## 6. Implementation with Neuralforecast

### 6.1 Data Preparation

```python
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT

# Nixtla long format with covariates
df = pd.DataFrame({
    "unique_id": "store_A",
    "ds":        pd.date_range("2020-01-01", periods=365*3, freq="D"),
    "y":         sales_series,

    # Time-varying known (available for future)
    "price":        price_series,
    "is_promo":     promo_series,
    "day_of_week":  pd.date_range("2020-01-01", periods=365*3, freq="D").dayofweek,
    "is_holiday":   holiday_series,

    # Static
    "store_size":   1500,     # same for all rows of this series
    "region":       "north",  # categorical static
})
```

### 6.2 TFT Model Configuration

```python
model_tft = TFT(
    h=30,                              # forecast horizon
    input_size=90,                     # lookback window (3× horizon recommended)

    # Covariates
    hist_exog_list=["price", "is_promo"],  # historical-only (past observations)
    futr_exog_list=["day_of_week", "is_holiday", "price", "is_promo"],  # known future
    stat_exog_list=["store_size"],     # static (time-invariant)

    # Architecture
    hidden_size=64,
    n_head=4,                          # attention heads
    attn_dropout=0.0,
    dropout=0.1,
    ffn_dim=64,                        # feed-forward dimension

    # Training
    max_steps=1000,
    batch_size=32,
    learning_rate=1e-3,
    loss="MQLoss",                     # multiple quantile loss for prediction intervals
    quantiles=[0.1, 0.5, 0.9],        # 10th, 50th, 90th percentiles

    # Validation
    val_check_steps=100,
    early_stop_patience_steps=5,
    random_seed=42,
)

nf = NeuralForecast(models=[model_tft], freq="D")
nf.fit(df=df)

# Forecast (provide future covariates for forecast period)
future_df = create_future_df(df, h=30)   # helper to create future covariate rows
forecast_df = nf.predict(futr_df=future_df)
print(forecast_df.head())
# Columns: unique_id, ds, TFT-q10, TFT-q50, TFT-q90
```

### 6.3 Prediction Intervals

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(actual_test, color="black", linewidth=2, label="Actual")
ax.plot(forecast_df["TFT-q50"], color="#D7191C",
        linewidth=2, linestyle="--", label="TFT Median (q50)")
ax.fill_between(
    forecast_df.index,
    forecast_df["TFT-q10"],
    forecast_df["TFT-q90"],
    color="#D7191C", alpha=0.2, label="80% Prediction Interval"
)
ax.legend()
ax.set_title("TFT Forecast with Prediction Intervals")
plt.tight_layout()
plt.show()
```

---

*← [04 — N-BEATS](./04_nbeats_and_nhits.md) | [Module README](./README.md) | Next: [06 — Training Best Practices](./06_training_best_practices.md) →*
