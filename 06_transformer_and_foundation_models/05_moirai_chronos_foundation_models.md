# 05 — Moirai & Chronos (Foundation Models)

> **Module**: 06 Transformers & Foundation Models | **File**: 5 of 7
>
> Moirai (Salesforce, 2024) and Chronos (Amazon, 2024) are fully open-source foundation models — unlike TimeGPT, their weights are publicly available. Both target universal zero-shot forecasting and represent the frontier of pre-trained time series models.

---

## Table of Contents

1. [The Open Foundation Model Landscape](#1-the-open-foundation-model-landscape)
2. [Amazon Chronos — T5 Backbone for TS](#2-amazon-chronos--t5-backbone-for-ts)
3. [Chronos — Implementation & Inference](#3-chronos--implementation--inference)
4. [Salesforce Moirai — Universal Forecasting](#4-salesforce-moirai--universal-forecasting)
5. [Moirai — Implementation & Inference](#5-moirai--implementation--inference)
6. [Fine-Tuning Foundation Models](#6-fine-tuning-foundation-models)
7. [Benchmarking Zero-Shot Models](#7-benchmarking-zero-shot-models)

---

## 1. The Open Foundation Model Landscape

```
Foundation Model Timeline:

2023 ─── TimeGPT-1 (Nixtla)          ← First commercial TS foundation model
2024 ─┬─ Chronos (Amazon)             ← Open, T5-based, tokenizes TS values
      ├─ Moirai (Salesforce)          ← Open, trained on LOTSA corpus
      ├─ Lag-Llama (Mila)             ← Open, LLaMA backbone, probabilistic
      └─ UniTS (MIT)                   ← Unified architecture (classification + forecast)

Key distinction:
  Commercial (closed):  TimeGPT → API only, no weights
  Open source (open):   Chronos, Moirai, Lag-Llama → HuggingFace weights
```

### Why Open Matters

| Benefit | Closed (TimeGPT) | Open (Chronos/Moirai) |
|---------|-----------------|----------------------|
| **Fine-tuning** | Limited API | Full control over training |
| **On-premise** | ❌ | ✅ No data leaves company |
| **Cost at scale** | API credits | One-time GPU cost |
| **Customization** | None | Full architecture access |
| **Research** | ❌ | ✅ Reproducible |

---

## 2. Amazon Chronos — T5 Backbone for TS

> **Paper**: Ansari et al., 2024. *Chronos: Learning the Language of Time Series*. TMLR.

### 2.1 The Key Innovation: Quantization as Tokenization

Chronos treats time series forecasting as a **language modeling problem** — literally converting real-valued time series into a sequence of integer tokens, then using a language model (T5) to predict the next tokens.

```
Step 1: Normalize each time series instance-wise (zero-mean, unit-variance)

Step 2: Quantize values into B discrete bins:
        - Map [-∞, ∞] → B integer bins using uniform quantization
        - Each bin = one "token" (like a word in NLP)
        - Typical B = 4096 tokens

Step 3: Encode as token sequence: [t₁, t₂, ..., tₙ] (integer IDs)

Step 4: Feed to T5 encoder-decoder:
        - Encoder: processes context tokens
        - Decoder: generates forecast tokens autoregressively
        
Step 5: Sample M trajectories from the decoder's distribution
        → Get M sample paths → compute quantiles

This is exactly how GPT generates text, but applied to numeric bins!
```

### 2.2 T5 Architecture Recap

```
T5 (Text-to-Text Transfer Transformer):
  Encoder: bidirectional self-attention → context representation
  Decoder: causal self-attention + cross-attention to encoder
  
In Chronos:
  Encoder input: quantized HISTORICAL values
  Decoder output: quantized FORECAST values
  Training: next-token prediction (cross-entropy on token bins)
```

### 2.3 Model Sizes

| Model | Parameters | Speed | Accuracy |
|-------|-----------|-------|---------|
| `chronos-t5-tiny` | 8M | Fastest | Baseline |
| `chronos-t5-mini` | 20M | Fast | Good |
| `chronos-t5-small` | 46M | Moderate | Better |
| `chronos-t5-base` | 200M | Slower | Best (open) |
| `chronos-t5-large` | 710M | Slow | Best overall |

---

## 3. Chronos — Implementation & Inference

### 3.1 Installation

```bash
pip install chronos-forecasting
```

### 3.2 Basic Zero-Shot Forecast

```python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chronos import ChronosPipeline

# Load pre-trained model (downloads automatically from HuggingFace)
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",   # try 'tiny' for speed, 'large' for accuracy
    device_map="cpu",             # "cuda" if GPU available
    torch_dtype=torch.float32,
)

# Create synthetic time series
np.random.seed(42)
n = 365 * 2
t = np.arange(n)
series = (
    100 + 0.1 * t
    + 30 * np.sin(2 * np.pi * t / 365.25)
    + 10 * np.sin(2 * np.pi * t / 7)
    + np.random.normal(0, 5, n)
)

# Convert to tensor (Chronos accepts both numpy arrays and tensors)
context = torch.tensor(series, dtype=torch.float32)

H = 30   # forecast horizon

# Zero-shot forecast — NO training required
forecasts = pipeline.predict(
    context=context,
    prediction_length=H,
    num_samples=100,          # sample 100 trajectories from the model
    temperature=1.0,          # sampling temperature (1.0 = default)
    top_k=50,                 # top-k sampling
    top_p=1.0,                # nucleus sampling
)
# forecasts: (num_series, num_samples, H) = (1, 100, 30)

# Extract quantiles from samples
low    = np.quantile(forecasts[0].numpy(), 0.1, axis=0)  # 10th percentile
median = np.quantile(forecasts[0].numpy(), 0.5, axis=0)  # median
high   = np.quantile(forecasts[0].numpy(), 0.9, axis=0)  # 90th percentile

print(f"Forecast shape: {forecasts.shape}")
print(f"Median forecast (first 5 steps): {median[:5]}")
```

### 3.3 Visualization

```python
fig, ax = plt.subplots(figsize=(13, 5))

# Plot last 60 history points
hist_idx  = np.arange(n - 60, n)
fore_idx  = np.arange(n, n + H)

ax.plot(hist_idx, series[-60:], color="black", linewidth=2, label="History")
ax.plot(fore_idx, median, color="#D7191C", linewidth=2,
        linestyle="--", label="Chronos Median (q50)")
ax.fill_between(fore_idx, low, high,
                color="#D7191C", alpha=0.2, label="80% Prediction Interval")
ax.axvline(n - 0.5, color="black", linewidth=0.8, linestyle=":")
ax.set_title("Chronos Zero-Shot Forecast", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.show()
```

### 3.4 Multiple Series (Batch Inference)

```python
# Chronos handles multiple series efficiently in one batch call
series_list = [
    torch.tensor(series[:365], dtype=torch.float32),   # series 1 (shorter)
    torch.tensor(series, dtype=torch.float32),          # series 2 (full)
    torch.tensor(series * 2, dtype=torch.float32),      # series 3 (scaled)
]

# Pad shorter series automatically
forecasts_batch = pipeline.predict(
    context=series_list,     # list of tensors (can have different lengths!)
    prediction_length=30,
    num_samples=100,
)
print(f"Batch forecast shape: {forecasts_batch.shape}")  # (3, 100, 30)
```

### 3.5 Evaluation Against Baselines

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Split: train = first 80%, test = last H steps
split  = n - H
train  = series[:split]
actual = series[split:]

# ── Seasonal Naive ───────────────────────────────────────────────────────────
period    = 7   # weekly for daily data
snaive    = train[-period:].mean() * np.ones(H)  # simplified

# ── Chronos Zero-Shot ─────────────────────────────────────────────────────────
context_tr = torch.tensor(train, dtype=torch.float32)
fc_samples = pipeline.predict(context_tr, prediction_length=H, num_samples=100)
fc_chronos = np.median(fc_samples[0].numpy(), axis=0)

# ── SARIMA ───────────────────────────────────────────────────────────────────
model_sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                       enforce_stationarity=False)
fit_sarima   = model_sarima.fit(disp=False)
fc_sarima    = fit_sarima.forecast(H)

rmse = lambda a, p: np.sqrt(((np.array(a) - np.array(p))**2).mean())
print(f"{'Model':<20} {'RMSE':>10}")
print("-" * 32)
print(f"{'Seasonal Naive':<20} {rmse(actual, snaive):>10.4f}")
print(f"{'SARIMA':<20} {rmse(actual, fc_sarima):>10.4f}")
print(f"{'Chronos (small)':<20} {rmse(actual, fc_chronos):>10.4f}")
```

---

## 4. Salesforce Moirai — Universal Forecasting

> **Paper**: Woo et al., 2024. *Unified Training of Universal Time Series Forecasting Transformers*. ICML.

### 4.1 Key Contributions

| Aspect | Moirai Innovation |
|--------|-----------------|
| **Training corpus** | LOTSA (Large-scale Open Time Series Archive) — 27 open datasets, 9 domains, 100+ billion observations |
| **Frequencies** | All: secondly → yearly (6 frequencies in training) |
| **Architecture** | Unified transformer with **patch-based** tokenization (like PatchTST) |
| **Covariates** | ✅ Supports time-varying features |
| **Distribution head** | Mixture distribution: learns which parametric family fits the data |
| **Any-variate** | Channel-mixing head can handle variable numbers of channels |

### 4.2 LOTSA Training Corpus

```
LOTSA Domains:
  ├── Energy (electricity, solar, wind)
  ├── Transport (traffic, ride-sharing)
  ├── Finance (stock prices, exchange rates)
  ├── Weather (temperature, precipitation)
  ├── Healthcare (patient vitals, flu counts)
  ├── Retail (M4, M5 competition data)
  ├── Web (Wikipedia page views, Google Trends)
  ├── Nature (air quality, seismic)
  └── Industrial (sensor readings, manufacturing)

Scale: ~100 billion total observations across thousands of series
Frequencies: secondly (1/60Hz) to yearly
```

### 4.3 Mixture Distribution Head

Unlike Chronos (categorical quantization) or TFT (fixed quantiles), Moirai uses a **mixture of distributions** — it picks the right output distribution automatically:

```
Output options (learned mixture weights):
  - Student-t (heavy tails → financial data)
  - Log-Normal (positive-only → sales, demand)
  - Normal (symmetric → temperature, residuals)
  - Negative Binomial (count data → item demand)
  - Weibull (time-to-event, right-skewed)

The model learns: "For THIS type of series, use THIS distribution"
→ Automatically adapts to the data generating process
```

---

## 5. Moirai — Implementation & Inference

### 5.1 Installation

```bash
pip install uni2ts
```

### 5.2 Zero-Shot Forecast

```python
import torch
import numpy as np
import pandas as pd
from einops import rearrange
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# Model sizes: 'moirai-1.0-R-small', 'moirai-1.0-R-base', 'moirai-1.0-R-large'
model = MoiraiForecast.from_pretrained(
    "Salesforce/moirai-1.0-R-small",
    prediction_length=30,
    context_length=200,
    patch_size=32,              # patch length for tokenization
    num_samples=100,            # number of trajectory samples
    target_dim=1,               # univariate
    feat_dynamic_real_dim=0,    # no dynamic features
    past_feat_dynamic_real_dim=0,
)

# Build input tensor
np.random.seed(42)
series = np.random.randn(500).cumsum() + 100
past_target = rearrange(
    torch.tensor(series, dtype=torch.float32),
    "t -> 1 1 t"    # (batch=1, variate=1, time)
)

# Inference
with torch.no_grad():
    forecast = model(past_target=past_target)  # (1, num_samples, H)

samples   = forecast[0].numpy()  # (100, 30)
median_fc = np.median(samples, axis=0)
q10       = np.quantile(samples, 0.1, axis=0)
q90       = np.quantile(samples, 0.9, axis=0)

print(f"Moirai forecast (median first 5): {median_fc[:5]}")
```

### 5.3 Moirai with GluonTS Integration

```python
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from uni2ts.eval_util.evaluation import evaluate_forecasts

# Package data in GluonTS format
train_data = ListDataset(
    [{"start": "2022-01-01",
      "target": series[:400],
      "item_id": "series_1"}],
    freq="D",
)
test_data = ListDataset(
    [{"start": "2022-01-01",
      "target": series,
      "item_id": "series_1"}],
    freq="D",
)

# GluonTS predictor interface
predictor = model.create_predictor(batch_size=32)
forecasts  = list(predictor.predict(train_data))

# Evaluate
results = evaluate_forecasts(
    forecasts,
    test_data=test_data,
    metrics=["mase", "mape", "rmse"],
    prediction_length=30,
)
print(results)
```

---

## 6. Fine-Tuning Foundation Models

### 6.1 When to Fine-Tune

```
Zero-shot is sufficient:
  ✅ General datasets (retail, weather, energy)
  ✅ Standard frequencies (daily, weekly, monthly)
  ✅ Rapid prototyping / baseline

Fine-tuning adds value:
  ✅ Specialized domain (medical, industrial sensor, satellite)
  ✅ Unusual patterns not in training data
  ✅ Consistent zero-shot underperformance vs. simple baselines
  ✅ You have > 500 training observations per series
```

### 6.2 Fine-Tuning Chronos

```python
# Fine-tuning Chronos on custom data using the official script
# https://github.com/amazon-science/chronos-forecasting

import torch
from torch.utils.data import DataLoader
from chronos import ChronosPipeline
from chronos.training import ChronosDataset

# Your custom dataset
train_series = [
    torch.tensor(my_series_1, dtype=torch.float32),
    torch.tensor(my_series_2, dtype=torch.float32),
]

# Wrap in ChronosDataset
# (handles tokenization, patching, and loss computation internally)
dataset = ChronosDataset(
    target_series=train_series,
    context_length=512,
    prediction_length=24,
    mode="training",
)

train_dl = DataLoader(dataset, batch_size=32, shuffle=True)

# Load model in fine-tune mode
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    torch_dtype=torch.float32,
)
model  = pipeline.model
optim  = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Fine-tuning loop (few hundred steps usually sufficient)
model.train()
for step, batch in enumerate(train_dl):
    optim.zero_grad()
    loss = model(**batch).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()

    if step % 50 == 0:
        print(f"Step {step}: loss={loss.item():.4f}")

    if step >= 200:   # short fine-tune (full-scale: 1000–5000 steps)
        break

print("✅ Fine-tuning complete")
```

### 6.3 Fine-Tuning Moirai

```python
# Moirai fine-tuning uses PyTorch Lightning
# https://github.com/SalesforceAIResearch/uni2ts

from uni2ts.model.moirai import MoiraiFinetune

model_ft = MoiraiFinetune.from_pretrained(
    "Salesforce/moirai-1.0-R-small",
    prediction_length=30,
    context_length=200,
    patch_size=32,
)

# Configure training
import pytorch_lightning as pl

trainer = pl.Trainer(
    max_epochs=10,
    gradient_clip_val=1.0,
    precision="bf16-mixed",    # use bf16 for efficiency
    accelerator="auto",
    log_every_n_steps=10,
)

# Train on your dataset (uses GluonTS format)
trainer.fit(model_ft, train_dataloaders=your_train_dataloader)
```

---

## 7. Benchmarking Zero-Shot Models

A rigorous evaluation framework to compare foundation models:

```python
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

def evaluate_zero_shot_models(
    series:          np.ndarray,
    H:               int = 30,
    n_chronos_samples: int = 100,
) -> pd.DataFrame:
    """
    Full evaluation pipeline comparing zero-shot foundation models
    against classical and ML baselines.
    
    Returns: metrics DataFrame with RMSE, MAE, MAPE per model.
    """
    split  = len(series) - H
    train  = series[:split]
    actual = series[split:]

    rmse_fn = lambda a, p: np.sqrt(((np.array(a) - np.array(p))**2).mean())
    mae_fn  = lambda a, p: np.abs(np.array(a) - np.array(p)).mean()
    mape_fn = lambda a, p: (np.abs((np.array(a) - np.array(p)) /
                             (np.array(a) + 1e-8)) * 100).mean()

    results = {}

    # ── 1. Seasonal Naive ────────────────────────────────────────────────────
    period = 7   # weekly (adjust for your frequency)
    snaive = np.tile(train[-period:], H // period + 1)[:H]
    results["Seasonal Naive"] = (rmse_fn(actual, snaive),
                                 mae_fn(actual, snaive),
                                 mape_fn(actual, snaive))

    # ── 2. SARIMA ────────────────────────────────────────────────────────────
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        fit = SARIMAX(train, order=(1,1,1), seasonal_order=(1,0,1,7),
                      enforce_stationarity=False).fit(disp=False)
        fc_sarima = fit.forecast(H).values
        results["SARIMA"] = (rmse_fn(actual, fc_sarima),
                             mae_fn(actual, fc_sarima),
                             mape_fn(actual, fc_sarima))
    except Exception as e:
        results["SARIMA"] = (float("nan"), float("nan"), float("nan"))
        print(f"SARIMA failed: {e}")

    # ── 3. Chronos Small ─────────────────────────────────────────────────────
    try:
        pipe = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small", device_map="cpu",
            torch_dtype=torch.float32
        )
        ctx = torch.tensor(train, dtype=torch.float32)
        samples = pipe.predict(ctx, prediction_length=H,
                               num_samples=n_chronos_samples)
        fc_chronos = np.median(samples[0].numpy(), axis=0)
        results["Chronos-small"] = (rmse_fn(actual, fc_chronos),
                                    mae_fn(actual, fc_chronos),
                                    mape_fn(actual, fc_chronos))
    except ImportError:
        print("chronos-forecasting not installed. Run: pip install chronos-forecasting")

    # ── Summary table ────────────────────────────────────────────────────────
    df_results = pd.DataFrame(results, index=["RMSE", "MAE", "MAPE (%)"]).T
    df_results = df_results.sort_values("RMSE")
    print("\n" + "="*55)
    print("Zero-Shot Evaluation Leaderboard")
    print("="*55)
    print(df_results.to_string(float_format="{:.4f}".format))
    print("="*55)
    return df_results
```

---

*← [04 — TimeGPT & Lag-Llama](./04_timegpt_and_lag_llama.md) | [Module README](./README.md) | Next: [06 — Zero-Shot Forecasting](./06_zero_shot_forecasting.md) →*
