# 02 — Informer, Autoformer & FEDformer

> **Module**: 06 Transformers & Foundation Models | **File**: 2 of 7
>
> Standard transformers struggle with long time series due to O(T²) attention complexity. Informer, Autoformer, and FEDformer each solve this differently — through sparse attention, autocorrelation, and frequency-domain decomposition.

---

## Table of Contents

1. [The Long-Sequence Forecasting Challenge](#1-the-long-sequence-forecasting-challenge)
2. [Informer — ProbSparse Attention](#2-informer--probsparse-attention)
3. [Autoformer — AutoCorrelation Mechanism](#3-autoformer--autocorrelation-mechanism)
4. [FEDformer — Frequency-Domain Decomposition](#4-fedformer--frequency-domain-decomposition)
5. [Comparison Table](#5-comparison-table)
6. [Implementation with neuralforecast](#6-implementation-with-neuralforecast)
7. [When to Use Each](#7-when-to-use-each)

---

## 1. The Long-Sequence Forecasting Challenge

Standard self-attention requires computing attention between **every pair** of T positions:

```
Memory: O(T²·d_model)   → infeasible for T > 5000
Time:   O(T²·d_model)   → slow for T > 1000

Real-world long-sequence examples:
  - 5-minute traffic: T=720 for 60 hours lookback
  - Hourly electricity: T=8,760 for 1 year lookback
  - 15-minute IoT:  T=35,040 for 1 year lookback
```

All three models target the **LTSF (Long-Term Time Series Forecasting)** setting: lookback 96–720 steps, horizon 96–720 steps.

---

## 2. Informer — ProbSparse Attention

> **Paper**: Zhou et al., 2021. *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*. AAAI.

### 2.1 Key Insight: Most Attention is Redundant

In standard attention, the attention distribution over T keys is nearly **uniform** for most queries — they contribute little information. Only a few "dominant" queries actually have a sharply-peaked distribution.

```
Standard attention: all T queries → all T keys  → O(T²)

ProbSparse insight: 
  Most attention maps ≈ uniform → negligible info
  Only the TOP-K queries matter (K = O(log T))
  → Only compute attention for those K queries
```

### 2.2 ProbSparse Attention Algorithm

```
Step 1: Sample M = U·ln(L_K) random key-query pairs
        (U is a constant, L_K = number of keys)

Step 2: Compute sparsity score for each query q_i:
        M(q_i, K) = max_j(q_i·k_j/√d) - (1/L_K)·Σⱼ(q_i·k_j/√d)
        → Measures how much q_i's distribution differs from uniform

Step 3: Select TOP-K queries by sparsity score
        (these are the "important" queries)

Step 4: Compute full attention ONLY for top-K queries
        Fill remaining with mean values

Complexity: O(T log T) — saves 5–10× over standard attention
```

### 2.3 Distilling Architecture (Encoder → Encoder)

Informer also uses a **distilling** operation between encoder layers:

```
Layer 1 output: (B, T, d_model)
             ↓ MaxPool(stride=2)
Layer 2 input: (B, T/2, d_model)   ← halved sequence length
             ↓ MaxPool(stride=2)
Layer 3 input: (B, T/4, d_model)

→ Final encoder output: (B, T/2ⁿ, d_model)  ← much shorter
→ Decoder only sees compact representation
```

```python
import torch
import torch.nn as nn

class InformerDistillingLayer(nn.Module):
    """Convolutional downsampling between encoder stacks."""
    def __init__(self, d_model):
        super().__init__()
        self.conv     = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm     = nn.BatchNorm1d(d_model)
        self.act      = nn.ELU()
        self.maxpool  = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: (B, T, d_model) → (B, d_model, T) for conv
        x = self.act(self.norm(self.conv(x.transpose(1, 2))))
        x = self.maxpool(x)          # halve sequence length
        return x.transpose(1, 2)     # back to (B, T//2, d_model)
```

### 2.4 Informer Forecast Head

```python
class InformerDecoder(nn.Module):
    """Generative decoder: predict full horizon in one shot (not autoregressive)."""
    def __init__(self, d_model, horizon):
        super().__init__()
        # Initialize decoder input with last known half of lookback
        # Concatenated with zeros for forecast positions
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x_dec, enc_output):
        # x_dec: (B, label_len + horizon, d_model)
        # Attention over encoder memory, then project
        return self.fc_out(x_dec).squeeze(-1)  # (B, label_len + horizon)
```

---

## 3. Autoformer — AutoCorrelation Mechanism

> **Paper**: Wu et al., 2021. *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*. NeurIPS.

### 3.1 Key Insight: Series Periodicity

Time series exhibit **periodicity** — tomorrow's value is closely related to the same time last week/month/year. Traditional attention doesn't exploit this.

**AutoCorrelation** replaces self-attention with a frequency-domain operation that directly finds and aggregates periodic dependencies.

### 3.2 AutoCorrelation Mechanism

```
Step 1: Compute FFT of Q and K to find periodic patterns
        R_QK(τ) = IFFT(FFT(Q) · conj(FFT(K)))   ← correlation at lag τ

Step 2: Find top-K lags with highest autocorrelation
        (K = number of dominant periods found)

Step 3: For each top lag τ_k:
        Roll the series by τ_k positions and aggregate

Step 4: Aggregate rolled values weighted by autocorrelation strength

Complexity: O(T log T)  ← FFT is O(T log T)
Key property: Captures PERIODIC dependencies (lag-based), not pairwise
```

```python
import torch
import torch.nn.functional as F

def autocorrelation_attention(Q, K, V, top_k=5):
    """
    AutoCorrelation: find top-k periodic lags and aggregate values.
    
    Q, K, V: (B, H, T, d_k)  — batch, heads, time, head_dim
    Returns: (B, H, T, d_k)
    """
    B, n_heads, T, d_k = Q.shape

    # Step 1: FFT-based correlation
    Q_fft = torch.fft.rfft(Q, dim=-2, norm="ortho")   # (B, H, T//2+1, d_k)
    K_fft = torch.fft.rfft(K, dim=-2, norm="ortho")

    # Cross-correlation via FFT (circular)
    R = Q_fft * K_fft.conj()
    R = torch.fft.irfft(R, n=T, dim=-2, norm="ortho")  # (B, H, T, d_k)

    # Step 2: Find top-k lags
    # Average over d_k to get single score per (B, H, τ)
    R_mean = R.mean(dim=-1)  # (B, H, T)
    weights, delays = R_mean.topk(top_k, dim=-1)       # (B, H, k)
    weights = torch.softmax(weights, dim=-1)            # normalize

    # Step 3: For each lag, roll V and aggregate
    V_rolled = torch.zeros_like(V)
    for i in range(top_k):
        delay = delays[:, :, i]   # (B, H)
        w     = weights[:, :, i].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        # Roll V by -delay positions
        V_shifted = torch.cat(
            [V[:, :, delay[0, 0]:, :],
             V[:, :, :delay[0, 0], :]], dim=-2
        )   # simplified: full impl uses vmap or scatter
        V_rolled += w * V_shifted

    return V_rolled
```

### 3.3 Progressive Decomposition

Autoformer uses **trend-seasonal decomposition** at each layer (not just at the input):

```
Each Autoformer Block:
  x → AutoCorrelation → Residual → LayerNorm
    → Feed-Forward → Residual → LayerNorm
    → Decomposition: split into trend (t) + seasonal (s) parts
  
  Trend part accumulates across blocks (becomes global trend)
  Seasonal part passed to next block for further refinement
```

```python
class SeriesDecomposition(nn.Module):
    """Moving average decomposition: trend = MA(x); seasonal = x - trend."""
    def __init__(self, kernel_size=25):
        super().__init__()
        # Padding to keep output length same as input
        self.avg_pool = nn.AvgPool1d(kernel_size, stride=1,
                                      padding=(kernel_size - 1) // 2)

    def forward(self, x):
        # x: (B, T, C)
        x_perm = x.permute(0, 2, 1)            # (B, C, T)
        trend   = self.avg_pool(x_perm).permute(0, 2, 1)  # (B, T, C)
        seasonal = x - trend
        return seasonal, trend
```

---

## 4. FEDformer — Frequency-Domain Decomposition

> **Paper**: Zhou et al., 2022. *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting*. ICML.

### 4.1 Key Insight: Sparse Frequency Representation

Time series are typically **sparse in the frequency domain** — most energy is concentrated in a few dominant frequencies (annual, weekly, daily cycles). FEDformer operates entirely in the frequency domain.

```
FED (Frequency Enhanced Decomposition):

Step 1: FFT the input → frequency spectrum F(x) of length T
Step 2: Randomly select r frequency components (r << T)
        → Random frequency selection as "attention"
Step 3: Operate in frequency domain (element-wise product)
Step 4: IFFT → back to time domain

Complexity: O(T log T)  ← FFT
Even more efficient: r is typically 32-64, regardless of T
```

### 4.2 Frequency Enhanced Block (FEB)

```python
import numpy as np

class FEDBlock(nn.Module):
    """
    Frequency Enhanced Block — attention in frequency domain.
    """
    def __init__(self, d_model, n_modes=32):
        """
        n_modes: number of frequency components to keep (r << T)
                 Typically 32–64; independent of sequence length T
        """
        super().__init__()
        self.n_modes   = n_modes
        # Complex-valued weights in frequency domain
        self.weight_real = nn.Parameter(torch.randn(d_model, d_model, n_modes) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(d_model, d_model, n_modes) * 0.02)

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, d = x.shape

        # Step 1: FFT over time dimension
        x_ft = torch.fft.rfft(x, dim=1)  # (B, T//2+1, d_model)

        # Step 2: Select first n_modes frequency components
        n_modes = min(self.n_modes, x_ft.shape[1])
        out_ft  = torch.zeros_like(x_ft)

        # Step 3: Apply complex-valued linear in frequency domain
        w = torch.complex(self.weight_real[..., :n_modes],
                          self.weight_imag[..., :n_modes])  # (d, d, n_modes)
        # Einsum: combine frequency features
        out_ft[:, :n_modes, :] = torch.einsum(
            'bid,djm->bjm',
            x_ft[:, :n_modes, :],    # (B, n_modes, d)
            w.permute(2, 0, 1)       # (n_modes, d, d)
        )

        # Step 4: IFFT → back to time domain
        x_out = torch.fft.irfft(out_ft, n=T, dim=1)  # (B, T, d_model)
        return x_out
```

### 4.3 FEDformer Architecture

```
Input → Decomposition (seasonal + trend)
         ↓
Seasonal → FED Block (frequency attention) → residual
Trend    → Moving average projection → accumulate
         ↓
Each FEDformer layer outputs:
  seasonal_output + trend_output  → passed to next layer
         ↓
Final: seasonal head + trend head → forecast
```

---

## 5. Comparison Table

| Aspect | Informer | Autoformer | FEDformer |
|--------|---------|-----------|---------|
| **Attention mechanism** | ProbSparse (top-K queries) | AutoCorrelation (lag-based FFT) | Frequency-enhanced (sparse FFT modes) |
| **Complexity** | O(T log T) | O(T log T) | O(T log T) |
| **Periodicity exploit** | ❌ | ✅ (lag aggregation) | ✅ (frequency modes) |
| **Decomposition** | ❌ | ✅ (at every layer) | ✅ (at every layer) |
| **Sequence length strength** | Medium (512–1440) | Long (720–960) | Very long (any) |
| **Best use case** | General long-seq | Seasonal data | Strongly periodic data |
| **Interpretability** | Low | Medium (lag weights) | Low |
| **neuralforecast support** | Via `Informer` | ❌ | ❌ |

---

## 6. Implementation with neuralforecast

```python
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import Informer

# Prepare data in Nixtla long format
np.random.seed(42)
n = 365 * 3   # 3 years of daily data
idx = pd.date_range("2021-01-01", periods=n, freq="D")
t = np.arange(n)

series = (
    100 + 0.1 * t
    + 30 * np.sin(2 * np.pi * t / 365.25)   # yearly seasonality
    + 10 * np.sin(2 * np.pi * t / 7)         # weekly seasonality
    + np.random.normal(0, 5, n)
)

df = pd.DataFrame({
    "unique_id": "series_1",
    "ds": idx,
    "y":  series,
})

H          = 96    # forecast 96 days ahead (long-horizon)
train_df   = df.iloc[:-H]
test_df    = df.iloc[-H:]
actual     = test_df["y"].values

# ── Informer ─────────────────────────────────────────────────────────────────
model_informer = Informer(
    h=H,
    input_size=H * 2,     # lookback = 2× horizon

    # Architecture
    hidden_size=64,
    n_head=4,
    e_layers=2,           # encoder layers
    d_layers=1,           # decoder layers
    d_ff=256,             # feed-forward dimension
    factor=5,             # ProbSparse: sample factor c (K = c·ln(T))
    dropout=0.1,

    # Training
    max_steps=1000,
    batch_size=32,
    learning_rate=1e-3,
    val_check_steps=100,
    early_stop_patience_steps=5,
    random_seed=42,
)

nf = NeuralForecast(models=[model_informer], freq="D")
nf.fit(df=train_df)
forecast = nf.predict()

pred_informer = forecast["Informer"].values
rmse = lambda a, p: np.sqrt(((a - p)**2).mean())
print(f"Informer RMSE: {rmse(actual, pred_informer):.4f}")
```

### Walk-Forward Cross-Validation

```python
# Time-safe walk-forward CV built into neuralforecast
cv_df = nf.cross_validation(
    df=df,
    n_windows=3,
    step_size=H,
)
from neuralforecast.losses.numpy import mse
cv_rmse = np.sqrt(mse(cv_df["y"].values, cv_df["Informer"].values))
print(f"Informer CV RMSE ({3} windows): {cv_rmse:.4f}")
```

---

## 7. When to Use Each

| Scenario | Best Choice | Reason |
|----------|-------------|--------|
| Horizon H < 96 steps | LightGBM or LSTM | Simpler models usually win short-horizon |
| H = 96–720, strong seasonality | **Autoformer** | AutoCorrelation matches seasonal structure |
| H = 96–720, mixed patterns | **Informer** | Good default for general long-horizon |
| Strongly periodic (electricity, traffic) | **FEDformer** | Frequency decomposition matches data structure |
| Zero budget for tuning | **Chronos** (foundation model) | Zero-shot competitive baseline |

> **Practical note**: Despite their theoretical appeal, empirical results in the literature (Zeng et al., 2023 "Are Transformers Effective for Time Series Forecasting?") showed that a simple **linear model** (DLinear) often beats Informer/Autoformer on standard benchmarks. Always validate against simple baselines before committing to complex transformer architectures.

---

*← [01 — Attention for TS](./01_attention_for_ts.md) | [Module README](./README.md) | Next: [03 — PatchTST & TimesNet](./03_patchtst_timesnet.md) →*
