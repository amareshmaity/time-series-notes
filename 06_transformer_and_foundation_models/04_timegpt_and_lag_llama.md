# 04 — TimeGPT & Lag-Llama

> **Module**: 06 Transformers & Foundation Models | **File**: 4 of 7
>
> TimeGPT (Nixtla, 2023) and Lag-Llama (Mila, 2024) represent the first serious application of **GPT-style** and **LLaMA-style** large language model architectures to time series forecasting — enabling zero-shot inference on new datasets without any training.

---

## Table of Contents

1. [The Foundation Model Paradigm for Time Series](#1-the-foundation-model-paradigm-for-time-series)
2. [TimeGPT — Architecture & Training](#2-timegpt--architecture--training)
3. [TimeGPT — API Usage (Nixtla SDK)](#3-timegpt--api-usage-nixtla-sdk)
4. [Lag-Llama — LLaMA Backbone for TS](#4-lag-llama--llama-backbone-for-ts)
5. [Zero-Shot vs. Fine-Tuned TimeGPT](#5-zero-shot-vs-fine-tuned-timegpt)
6. [Anomaly Detection with TimeGPT](#6-anomaly-detection-with-timegpt)
7. [TimeGPT vs. Lag-Llama vs. Classical Models](#7-timegpt-vs-lag-llama-vs-classical-models)

---

## 1. The Foundation Model Paradigm for Time Series

The key shift foundation models bring to time series:

```
Traditional approach:
  For each new dataset:
    1. Collect data
    2. Feature engineer
    3. Train model
    4. Tune hyperparameters
    → Weeks of work per dataset

Foundation model approach:
  1. Pre-trained on BILLIONS of time series
  2. Call API with new series
  3. Get forecast immediately
  → Minutes to production

Just like GPT-4 can answer questions about ANY topic without topic-specific training,
TimeGPT can forecast ANY time series without dataset-specific training.
```

### 1.1 What Makes TS Foundation Models Possible?

Three enablers:

| Enabler | Details |
|---------|---------|
| **Scale** | Trained on millions/billions of time series (LOTSA, M-competitions, M5, proprietary) |
| **Architecture** | Transformer attention can learn cross-series patterns at scale |
| **Pre-training objective** | Masked forecasting (predict held-out windows) — analogous to BERT's masked language modeling |

---

## 2. TimeGPT — Architecture & Training

> **Paper**: Garza & Mergenthaler-Canseco, 2023. *TimeGPT-1*. arXiv:2310.03589.

### 2.1 Architecture

TimeGPT uses a **standard encoder-decoder Transformer** (similar to T5) with adaptations for time series:

```
Input: time series window of length L (the "context")
       + optional exogenous features (known future covariates)

Encoder:
  Tokenization: each timestep → 1 token
  + Positional encoding (learned)
  + Multi-head self-attention
  + Feed-forward layers
  → Context representation: (L, d_model)

Decoder:
  Forecast tokens (autoregressive): generates H steps
  Cross-attention over encoder output
  Causal self-attention
  → Quantile output: q10, q50, q90 per step

Output: H-step probabilistic forecast
```

### 2.2 Training Data

TimeGPT-1 training corpus:
- **100 billion data points** across multiple domains
- M4, M5, FRED (economic), electricity, traffic, weather, web traffic
- Multiple frequencies: daily, weekly, monthly, quarterly, annual
- Horizon range: 1–720 steps

### 2.3 Training Objective

```
Masked Forecasting:
  Input:  [y₁, ..., y_{L-H}, MASK, MASK, ..., MASK]   ← H masked positions
  Target: [y_{L-H+1}, ..., y_L]                         ← predict masked values

Quantile loss (pinball):
  Train simultaneously for q = 0.1, 0.5, 0.9
  → Model learns to produce calibrated uncertainty

This is identical to BERT/GPT pre-training but for numeric sequences.
```

---

## 3. TimeGPT — API Usage (Nixtla SDK)

TimeGPT is available exclusively through the Nixtla API. No model weights are released (commercial product).

### 3.1 Installation & Setup

```bash
pip install nixtla
```

```python
import os
from nixtla import NixtlaClient

# Initialize client — set NIXTLA_API_KEY environment variable
# Get free API key at: https://dashboard.nixtla.io/
client = NixtlaClient(api_key=os.environ["NIXTLA_API_KEY"])

# Validate connection
print(client.validate_api_key())
```

### 3.2 Basic Forecast

```python
import pandas as pd
import numpy as np

# Create sample series
np.random.seed(42)
n = 365 * 2
idx = pd.date_range("2022-01-01", periods=n, freq="D")
t = np.arange(n)

series = (
    100
    + 0.1 * t
    + 30 * np.sin(2 * np.pi * t / 365.25)
    + 10 * np.sin(2 * np.pi * t / 7)
    + np.random.normal(0, 5, n)
)

df = pd.DataFrame({"ds": idx, "y": series, "unique_id": "series_1"})

# Zero-shot forecast — NO training required
forecast_df = client.forecast(
    df=df,
    h=30,                      # forecast horizon (steps ahead)
    freq="D",                  # frequency: D=daily, W=weekly, MS=monthly
    time_col="ds",
    target_col="y",
    id_col="unique_id",
    level=[80, 95],            # prediction interval levels
)

print(forecast_df.head())
# Columns: unique_id, ds, TimeGPT, TimeGPT-lo-80, TimeGPT-hi-80, TimeGPT-lo-95, TimeGPT-hi-95
```

### 3.3 Forecast with Known Future Covariates

```python
# Build future covariates dataframe
# For features known in advance (calendar, promotions)
future_covariates = pd.DataFrame({
    "unique_id":  "series_1",
    "ds":         pd.date_range(df["ds"].max() + pd.Timedelta(days=1), periods=30, freq="D"),
    "is_holiday": [1 if d.weekday() >= 5 else 0 for d in
                   pd.date_range(df["ds"].max() + pd.Timedelta(days=1), periods=30, freq="D")],
    "day_of_week": pd.date_range(df["ds"].max() + pd.Timedelta(days=1), periods=30, freq="D").dayofweek,
})

# Add historical covariates to the training df too
df["is_holiday"]  = [1 if d.weekday() >= 5 else 0 for d in idx]
df["day_of_week"] = idx.dayofweek

forecast_with_cov = client.forecast(
    df=df,
    X_df=future_covariates,   # exogenous features for forecast period
    h=30,
    freq="D",
    level=[90],
)
print(f"Columns: {forecast_with_cov.columns.tolist()}")
```

### 3.4 Multiple Series (Global Forecast)

```python
# Prepare multi-series long format
dfs = []
for store_id in range(5):
    store_series = (
        100 * (1 + store_id * 0.3)
        + 0.1 * t
        + 25 * np.sin(2 * np.pi * t / 365.25)
        + np.random.normal(0, 5, n)
    )
    dfs.append(pd.DataFrame({
        "unique_id": f"store_{store_id}",
        "ds": idx,
        "y":  store_series,
    }))

df_multi = pd.concat(dfs, ignore_index=True)

# Forecast ALL series in one API call
forecast_multi = client.forecast(
    df=df_multi,
    h=30,
    freq="D",
    level=[90],
)
print(f"Forecasting {df_multi['unique_id'].nunique()} series")
print(f"Forecast rows: {len(forecast_multi)}")   # 5 series × 30 steps = 150 rows
```

### 3.5 Anomaly Detection

```python
# TimeGPT can also flag anomalies using prediction intervals
anomaly_df = client.detect_anomalies(
    df=df,
    freq="D",
    level=99,   # flag points outside 99% prediction interval as anomalies
)
print(anomaly_df[anomaly_df["anomaly"] == 1].head())
```

---

## 4. Lag-Llama — LLaMA Backbone for TS

> **Paper**: Rasul et al., 2024. *Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting*. ICLR.

### 4.1 Key Differences from TimeGPT

| Aspect | TimeGPT | Lag-Llama |
|--------|---------|-----------|
| **Architecture** | Encoder-Decoder Transformer | Decoder-only (LLaMA) |
| **Inference** | Autoregressive (step-by-step) | Autoregressive |
| **Probabilistic output** | Quantile regression | Full distribution (Student-t, NegBin) |
| **Weights** | Closed (API only) | Open (HuggingFace) |
| **Tokenization** | Scalar values | Lag features as tokens |
| **Training data** | Proprietary (100B points) | Open (27 datasets, ~350M points) |

### 4.2 Lag Feature Tokenization

Lag-Llama's key innovation is using **lag features** as the input tokens instead of raw values. This allows the model to see long-range history within a fixed context window:

```python
# Standard approach (single value per token):
#   Context length 512 → sees 512 steps back
#
# Lag-Llama approach (lag features as tokens):
#   At each step t, the "token" contains:
#   [y_t, y_{t-1}, y_{t-7}, y_{t-30}, y_{t-365}, ...]
#   → Model sees up to 365+ steps back with a small context window

def build_lag_features(series: np.ndarray, lags: list) -> np.ndarray:
    """
    Build lag feature matrix for Lag-Llama-style tokenization.
    
    series: (T,) time series
    lags:   list of lag indices [1, 7, 30, 365, ...]
    
    Returns: (T, len(lags)) feature matrix
    """
    n = len(series)
    features = np.zeros((n, len(lags)), dtype=np.float32)
    for i, lag in enumerate(lags):
        features[lag:, i] = series[:n - lag]
    return features


# Typical Lag-Llama lags for hourly data
HOURLY_LAGS = [1, 2, 3, 4, 5, 6, 7, 23, 24, 25, 47, 48, 49,
               71, 72, 73, 95, 96, 119, 120, 143, 144, 167, 168]
```

### 4.3 Using Lag-Llama (Open Source)

```python
# Install: pip install git+https://github.com/time-series-foundation-models/lag-llama.git
# Requires: pip install gluonts torch huggingface_hub

from huggingface_hub import hf_hub_download
import torch

# Download model weights from HuggingFace
ckpt_path = hf_hub_download(
    repo_id="time-series-foundation-models/Lag-Llama",
    filename="lag-llama.ckpt",
)
print(f"Model downloaded to: {ckpt_path}")

# Load the model
from lag_llama.gluon.estimator import LagLlamaEstimator

estimator = LagLlamaEstimator(
    ckpt_path=ckpt_path,
    prediction_length=24,    # forecast horizon
    context_length=32,       # look-back window (lag features extend this)
    device=torch.device("cpu"),
    batch_size=64,
    num_parallel_samples=100,  # samples for probabilistic output
)

# Convert to GluonTS predictor
predictor = estimator.create_predictor(
    training_data=None,   # None = zero-shot mode (no fine-tuning)
    cache_data=True,
)

# Create a GluonTS dataset entry
import pandas as pd
from gluonts.dataset.common import ListDataset

dataset = ListDataset(
    [{"start": pd.Period("2022-01-01", freq="H"),
      "target": np.random.randn(500).cumsum() + 100}],
    freq="H",
)

# Get probabilistic forecast
forecasts = list(predictor.predict(dataset))
forecast  = forecasts[0]

# Extract samples
samples = forecast.samples   # (num_samples, horizon)
print(f"Forecast quantile 50%: {np.median(samples, axis=0)[:5]}")
print(f"Forecast quantile 90%: {np.quantile(samples, 0.9, axis=0)[:5]}")
```

---

## 5. Zero-Shot vs. Fine-Tuned TimeGPT

### 5.1 When Zero-Shot Suffices

```python
import numpy as np

def compare_zero_shot_vs_tuned(
    client,
    df: pd.DataFrame,
    h: int = 30,
    n_fine_tune_steps: int = 100,
):
    """
    Compare zero-shot TimeGPT vs. fine-tuned TimeGPT on a held-out test set.
    Splits df into train/test (last h rows = test).
    """
    train_df = df.iloc[:-h]
    test_df  = df.iloc[-h:]
    actual   = test_df["y"].values

    rmse = lambda a, p: np.sqrt(((a - p)**2).mean())

    # ── Zero-shot ─────────────────────────────────────────────────────────
    fc_zero = client.forecast(df=train_df, h=h, freq="D", level=[90])
    rmse_zero = rmse(actual, fc_zero["TimeGPT"].values)
    print(f"Zero-shot RMSE:     {rmse_zero:.4f}")

    # ── Fine-tuned ────────────────────────────────────────────────────────
    # Fine-tuning adapts TimeGPT's weights to your specific series
    fc_tuned = client.forecast(
        df=train_df,
        h=h,
        freq="D",
        level=[90],
        finetune_steps=n_fine_tune_steps,  # gradient steps on your data
        finetune_loss="default",            # or "mae", "mse", "quantile"
    )
    rmse_tuned = rmse(actual, fc_tuned["TimeGPT"].values)
    print(f"Fine-tuned RMSE:    {rmse_tuned:.4f}")
    print(f"Improvement:        {(rmse_zero - rmse_tuned) / rmse_zero * 100:.1f}%")

    return fc_zero, fc_tuned
```

### 5.2 Fine-Tuning Decision Guide

```
When to use ZERO-SHOT TimeGPT:
  ✅ Cold start: new series with no history (< 50 observations)
  ✅ Many series with very short history (retail new products)
  ✅ Rapid prototype / exploratory analysis
  ✅ Benchmark comparison — is ML worth the effort?

When to FINE-TUNE:
  ✅ Your domain is unusual (medical, satellite, industrial sensor)
  ✅ You have >200 training observations
  ✅ Zero-shot accuracy is insufficient for business requirements
  ✅ Strong known-future covariates (promotions, holidays)

When to use a DIFFERENT MODEL entirely:
  ✅ Very long horizons (H > 720) → PatchTST or N-HiTS
  ✅ Many covariates with complex interactions → TFT
  ✅ Real-time low-latency requirement → LightGBM (no API call)
  ✅ Regulatory/explainability requirement → Ridge + Fourier features
```

---

## 6. Anomaly Detection with TimeGPT

```python
def detect_anomalies_with_timegpt(client, df, level=99):
    """
    Use TimeGPT's forecast-based anomaly detection.
    
    Strategy: fit model on training portion, forecast validation,
    flag points outside the prediction interval as anomalies.
    
    level: confidence level for interval (99 → very conservative)
    """
    anomaly_results = client.detect_anomalies(
        df=df,
        freq="D",
        level=level,
        clean_ex_first=True,   # use clean series for interval estimation
    )

    n_anomalies = anomaly_results["anomaly"].sum()
    print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.1f}% of points)")
    return anomaly_results

# Inject a synthetic anomaly for demonstration
df_with_anomaly = df.copy()
df_with_anomaly.loc[df_with_anomaly.index[200], "y"] = 1000  # spike

result = detect_anomalies_with_timegpt(client, df_with_anomaly, level=99)
print(result[result["anomaly"] == 1])
```

---

## 7. TimeGPT vs. Lag-Llama vs. Classical Models

| Criterion | TimeGPT | Lag-Llama | SARIMA | LightGBM |
|-----------|---------|-----------|--------|---------|
| **Setup time** | 2 min (API key) | 20 min (download) | 5 min | 5 min |
| **Training required** | ❌ Zero-shot | ❌ Zero-shot | ✅ Per-series | ✅ Per-task |
| **Probabilistic output** | ✅ Quantiles | ✅ Full distribution | ✅ Analytical | Via quantile loss |
| **Covariate support** | ✅ | ❌ | ✅ (SARIMAX) | ✅ |
| **Cold start** | ✅ Excellent | ✅ Excellent | ❌ Needs history | ❌ Needs features |
| **Latency** | Network latency | Fast (local) | Fast | Fast |
| **Cost** | API credits | Free | Free | Free |
| **Interpretability** | Low | Low | High (AIC/BIC) | Medium (SHAP) |
| **Open weights** | ❌ | ✅ HuggingFace | N/A | N/A |

### Decision Rule

```
< 50 observations per series                → TimeGPT zero-shot
Cold start (new product, new sensor)        → TimeGPT zero-shot
Domain specialty + sufficient history       → Fine-tune TimeGPT or Chronos
Need full distribution (not just quantiles) → Lag-Llama
Need interpretability                       → SARIMA or Ridge
Need speed/low cost at scale                → LightGBM
Production at scale with covariates         → TFT or LightGBM
```

---

*← [03 — PatchTST & TimesNet](./03_patchtst_timesnet.md) | [Module README](./README.md) | Next: [05 — Moirai & Chronos](./05_moirai_chronos_foundation_models.md) →*
