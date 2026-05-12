# 03 — PatchTST & TimesNet

> **Module**: 06 Transformers & Foundation Models | **File**: 3 of 7
>
> PatchTST (2023) and TimesNet (2023) represent the next generation of TS transformers — moving away from point-by-point token representations toward **patch-based** and **2D temporal** modeling that is both more efficient and more accurate.

---

## Table of Contents

1. [Why Patch-Based Transformers?](#1-why-patch-based-transformers)
2. [PatchTST — Patch Tokenization + Channel Independence](#2-patchtst--patch-tokenization--channel-independence)
3. [TimesNet — 2D Temporal Variation Modeling](#3-timesnet--2d-temporal-variation-modeling)
4. [DLinear — The Surprisingly Strong Baseline](#4-dlinear--the-surprisingly-strong-baseline)
5. [Comparison Table](#5-comparison-table)
6. [Implementation with neuralforecast](#6-implementation-with-neuralforecast)
7. [PatchTST from Scratch (Educational)](#7-patchtst-from-scratch-educational)

---

## 1. Why Patch-Based Transformers?

The core problem with earlier TS transformers (Informer, Autoformer):

```
Problem 1: Point tokenization
  Input: (T=720 timesteps, 1 value each) → 720 tokens → O(720²) = 518K attention pairs
  
Problem 2: Channel mixing
  Mix all variables together → loses individual series structure
  
Problem 3: Position-unaware
  Temporal local patterns (like "the last 7 days") are ignored
```

**PatchTST's solution**:

```
Patches: group P=16 consecutive timesteps → one token
  T=720 → 720/16 = 45 tokens → O(45²) = 2,025 pairs  (256× fewer!)
  
Channel independence: each variable processed separately
  Prevents spurious cross-variable correlations
  
Local semantics: each patch captures a local temporal window
```

---

## 2. PatchTST — Patch Tokenization + Channel Independence

> **Paper**: Nie et al., 2023. *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. ICLR.

### 2.1 Patching Strategy

```
Input series: [y₁, y₂, ..., yₙ] (T timesteps)

Non-overlapping patches (stride S = patch_len P):
  Patch 1: [y₁, ..., yₚ]
  Patch 2: [yₚ₊₁, ..., y₂ₚ]
  ...
  
Overlapping patches (stride S < P):
  Patch 1: [y₁, ..., yₚ]
  Patch 2: [y_{S+1}, ..., y_{S+P}]
  ...
  → (T - P) / S + 1 patches total

Typical: P=16, S=8  → much less than T tokens
```

### 2.2 Channel-Independent Architecture

```
Input: (B, T, n_vars)

For each variable v independently:
  x_v: (B, T) → patch: (B, N_patches, P) → embed: (B, N_patches, d_model)
  ↓
  + Positional Encoding
  ↓
  Transformer Encoder (standard, no modifications)
  ↓
  Flatten + Linear Head → (B, H)

Output: (B, H, n_vars)  — stack forecasts for all variables

Key: weights SHARED across variables (less overfitting, more data)
```

### 2.3 PatchTST Implementation

```python
import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
    """Convert time series into patch tokens."""
    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        # Project each patch (of length P) to d_model dimensions
        self.projection = nn.Linear(patch_len, d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T)  — one variable's time series
        Returns: (B, N_patches, d_model)
        """
        # Create overlapping patches using unfold
        # Pads the left with the first value to keep patch count predictable
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: (B, N_patches, patch_len)
        return self.dropout(self.projection(x))


class PatchTSTEncoder(nn.Module):
    """Standard Transformer Encoder for patch tokens."""
    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 ffn_dim: int, dropout: float = 0.1, max_patches: int = 64):
        super().__init__()
        self.pos_embed = nn.Embedding(max_patches, d_model)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN (more stable than Post-LN)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N_patches, d_model)"""
        N = x.shape[1]
        positions = torch.arange(N, device=x.device)
        x = x + self.pos_embed(positions)
        return self.encoder(x)   # (B, N_patches, d_model)


class PatchTST(nn.Module):
    """
    PatchTST: Channel-Independent Patch-Based Transformer.
    
    Architecture:
      - Each channel processed independently with shared weights
      - Patch tokenization: P consecutive timesteps → 1 token
      - Standard Transformer encoder (no fancy attention)
      - Linear prediction head
    """
    def __init__(
        self,
        n_vars:     int,
        lookback:   int,
        horizon:    int,
        patch_len:  int   = 16,
        stride:     int   = 8,
        d_model:    int   = 128,
        n_heads:    int   = 8,
        n_layers:   int   = 3,
        ffn_dim:    int   = 256,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_vars   = n_vars
        self.horizon  = horizon

        # Compute number of patches
        n_patches = (lookback - patch_len) // stride + 1
        self.patch_embed = PatchEmbedding(patch_len, stride, d_model, dropout)
        self.encoder     = PatchTSTEncoder(d_model, n_heads, n_layers,
                                           ffn_dim, dropout, n_patches)
        # Flatten all patch representations → forecast head
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),            # (B, N_patches * d_model)
            nn.Linear(n_patches * d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, n_vars)
        Returns: (B, H, n_vars)
        """
        outputs = []
        for v in range(self.n_vars):
            x_v   = x[:, :, v]                  # (B, T) — one variable
            tok_v = self.patch_embed(x_v)        # (B, N_patches, d_model)
            enc_v = self.encoder(tok_v)          # (B, N_patches, d_model)
            out_v = self.head(enc_v)             # (B, H)
            outputs.append(out_v)

        return torch.stack(outputs, dim=-1)      # (B, H, n_vars)


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T, n_vars, H = 4, 336, 7, 96
    model = PatchTST(n_vars=n_vars, lookback=T, horizon=H,
                     patch_len=16, stride=8, d_model=128)
    x = torch.randn(B, T, n_vars)
    y = model(x)
    print(f"PatchTST | Input: {x.shape} → Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Expected output: (4, 96, 7)
```

### 2.4 RevIN + PatchTST (Production Pattern)

PatchTST + RevIN (Reversible Instance Normalization) is the standard production recipe:

```python
class RevIN(nn.Module):
    """Instance normalization with learnable affine parameters — applied per-series."""
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps  = eps
        self.gamma = nn.Parameter(torch.ones(num_features))  # learnable scale
        self.beta  = nn.Parameter(torch.zeros(num_features)) # learnable shift

    def forward(self, x: torch.Tensor, mode: str = "norm"):
        """
        x: (B, T, C)
        mode: 'norm' → normalize; 'denorm' → reverse
        """
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std  = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self.mean) / self.std
            x = x * self.gamma + self.beta
        elif mode == "denorm":
            x = (x - self.beta) / (self.gamma + self.eps)
            x = x * self.std[:, 0:1, :] + self.mean[:, 0:1, :]
        return x


class PatchTSTWithRevIN(nn.Module):
    """PatchTST + RevIN — the production recipe."""
    def __init__(self, **kwargs):
        super().__init__()
        self.revin  = RevIN(kwargs.get("n_vars", 1))
        self.model  = PatchTST(**kwargs)

    def forward(self, x):
        x_norm = self.revin(x, "norm")
        y_norm = self.model(x_norm)
        return self.revin(y_norm, "denorm")
```

---

## 3. TimesNet — 2D Temporal Variation Modeling

> **Paper**: Wu et al., 2023. *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis*. ICLR.

### 3.1 Core Insight: 1D → 2D

A 1D time series with period p can be **reshaped into a 2D matrix**:

```
1D series: [y₁, y₂, ..., yₙ]  with period p
                    ↓ reshape
2D matrix: [y₁   y₂   ... yₚ  ]   ← intra-period (within one cycle)
            [yₚ₊₁ yₚ₊₂ ... y₂ₚ]   ← inter-period (across cycles)
            [...]

Rows = intra-period variation  (daily pattern within a week)
Cols = inter-period variation  (Monday pattern across all weeks)

→ Apply 2D convolutions to capture BOTH simultaneously!
```

### 3.2 Period Finding via FFT

```python
def find_top_periods(x: torch.Tensor, k: int = 5) -> list:
    """
    Find top-k dominant periods in a time series using FFT amplitude spectrum.
    
    x: (B, T, C) — time series
    Returns: list of k period lengths
    """
    # Take mean over batch and channels
    x_mean = x.mean(dim=(0, 2))           # (T,)

    # FFT amplitude spectrum (single-sided)
    fft_vals  = torch.fft.rfft(x_mean)
    amplitudes = fft_vals.abs()
    T = x_mean.shape[0]

    # Skip DC component (frequency 0)
    amplitudes[0] = 0

    # Top-k frequencies
    _, top_k_freq = torch.topk(amplitudes, k)
    # Convert frequency index → period length
    periods = [T // f.item() if f.item() > 0 else T for f in top_k_freq]
    return periods


class TimesBlock(nn.Module):
    """
    TimesNet core block: 1D → reshape to 2D → 2D Conv → reshape back to 1D.
    """
    def __init__(self, d_model: int, d_ff: int, top_k: int = 5):
        super().__init__()
        self.top_k    = top_k
        self.conv2d   = nn.Sequential(
            nn.Conv2d(d_model, d_ff, kernel_size=(3, 3), padding=(1, 1)),
            nn.GELU(),
            nn.Conv2d(d_ff, d_model, kernel_size=(3, 3), padding=(1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        B, T, C = x.shape
        periods = find_top_periods(x, self.top_k)

        outputs = []
        for p in periods:
            if p <= 1:
                continue

            # Pad to make T divisible by period p
            pad_len = (p - T % p) % p
            x_pad = torch.cat([x, x[:, :pad_len, :]], dim=1)   # (B, T+pad, C)

            # Reshape to 2D: (B, T/p, p, C) → (B, C, T/p, p)
            T_pad  = T + pad_len
            n_rows = T_pad // p
            x_2d   = x_pad.reshape(B, n_rows, p, C).permute(0, 3, 1, 2)  # (B, C, n_rows, p)

            # 2D convolution — captures intra+inter period patterns
            x_2d   = self.conv2d(x_2d)     # (B, C, n_rows, p)

            # Reshape back to 1D
            x_back = x_2d.permute(0, 2, 3, 1).reshape(B, -1, C)  # (B, T+pad, C)
            outputs.append(x_back[:, :T, :])   # trim padding

        if outputs:
            # Average across all periods
            out = torch.stack(outputs, dim=0).mean(dim=0)   # (B, T, C)
        else:
            out = x   # fallback

        return out + x  # residual connection
```

---

## 4. DLinear — The Surprisingly Strong Baseline

> **Paper**: Zeng et al., 2023. *Are Transformers Effective for Time Series Forecasting?* AAAI.

This paper showed that a **simple linear decomposition model** outperforms Informer, Autoformer, and FEDformer on standard LTSF benchmarks. A sobering lesson in model complexity.

```python
class DLinear(nn.Module):
    """
    DLinear: Decomposition-Linear forecasting model.
    
    Despite its simplicity, outperforms many transformer-based LTSF models.
    Key insight: trend and residual components may be forecast independently.
    """
    def __init__(self, lookback: int, horizon: int, n_vars: int = 1):
        super().__init__()
        self.decomp = SeriesDecompositionSimple(kernel_size=25)

        # Separate linear layers for trend and seasonal
        # Channel independence: one layer per variable
        self.trend_proj    = nn.Linear(lookback, horizon)
        self.seasonal_proj = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, n_vars)
        Returns: (B, H, n_vars)
        """
        seasonal, trend = self.decomp(x)    # each: (B, T, n_vars)

        # Transpose to (B, n_vars, T) for channel-wise linear
        trend_f    = self.trend_proj(trend.permute(0, 2, 1))     # (B, n_vars, H)
        seasonal_f = self.seasonal_proj(seasonal.permute(0, 2, 1))

        return (trend_f + seasonal_f).permute(0, 2, 1)   # (B, H, n_vars)


class SeriesDecompositionSimple(nn.Module):
    """Moving-average based trend-seasonal split."""
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size, stride=1,
                                 padding=(kernel_size - 1) // 2)

    def forward(self, x: torch.Tensor):
        """x: (B, T, C) → trend: (B, T, C), seasonal: (B, T, C)"""
        # AvgPool1d needs (B, C, T)
        trend    = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend
```

> **Lesson**: Always benchmark against DLinear before claiming a transformer improves performance. If DLinear beats your transformer, your transformer is not properly tuned or the dataset doesn't benefit from attention.

---

## 5. Comparison Table

| Model | Year | Key Innovation | Complexity | Covariates | Best Horizon |
|-------|------|---------------|-----------|-----------|-------------|
| **PatchTST** | 2023 | Patch tokens + channel independence | O((T/P)²) | ❌ | 96–720 |
| **TimesNet** | 2023 | 1D→2D reshape + 2D conv | O(T·k) | ❌ | Any |
| **DLinear** | 2023 | Decomposition + linear | O(T) | ❌ | 96–720 |
| **Informer** | 2021 | ProbSparse attention | O(T log T) | ✅ | 96–720 |
| **Autoformer** | 2021 | AutoCorrelation | O(T log T) | ✅ | 336–960 |
| **TFT** | 2021 | Multi-type covariates + attention | O(T²) | ✅ | 1–100 |

---

## 6. Implementation with neuralforecast

```python
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

# Build a multivariate dataset
np.random.seed(42)
n = 365 * 3
idx = pd.date_range("2021-01-01", periods=n, freq="D")
t = np.arange(n)

# Two related series
y1 = 100 + 0.1*t + 30*np.sin(2*np.pi*t/365.25) + np.random.normal(0, 3, n)
y2 = 200 + 0.2*t + 50*np.sin(2*np.pi*t/365.25 + 0.5) + np.random.normal(0, 5, n)

df = pd.concat([
    pd.DataFrame({"unique_id": "store_A", "ds": idx, "y": y1}),
    pd.DataFrame({"unique_id": "store_B", "ds": idx, "y": y2}),
])

H        = 96
train_df = df[df["ds"] < df["ds"].max() - pd.Timedelta(days=H)]
test_df  = df[df["ds"] >= df["ds"].max() - pd.Timedelta(days=H)]

model_patchtst = PatchTST(
    h=H,
    input_size=H * 2,     # lookback window

    # Patch configuration
    patch_len=16,          # each token covers 16 timesteps
    stride=8,              # overlap: 50% (S < P → overlapping)

    # Architecture
    d_model=128,
    n_heads=8,
    e_layers=3,            # encoder layers
    d_ff=256,              # feed-forward dimension
    dropout=0.1,
    revin=True,            # Reversible Instance Normalization

    # Training
    max_steps=1000,
    batch_size=32,
    learning_rate=1e-4,    # lower LR for transformers
    val_check_steps=100,
    early_stop_patience_steps=5,
    random_seed=42,
)

nf = NeuralForecast(models=[model_patchtst], freq="D")
nf.fit(df=train_df)
forecast = nf.predict()
print(forecast.head())
```

---

## 7. PatchTST from Scratch (Educational)

Complete minimal PatchTST for a univariate series (educational, not production):

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class UnivPatchTST(nn.Module):
    """Minimal univariate PatchTST for educational purposes."""
    def __init__(self, lookback=336, horizon=96, patch_len=16, stride=8,
                 d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        n_patches = (lookback - patch_len) // stride + 1

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed   = nn.Embedding(n_patches, d_model)
        self.norm_input  = nn.LayerNorm(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head    = nn.Linear(n_patches * d_model, horizon)

        self.patch_len = patch_len
        self.stride    = stride

    def forward(self, x):
        # x: (B, T) — univariate series
        # 1. Instance normalize
        mean = x.mean(dim=1, keepdim=True)
        std  = x.std(dim=1, keepdim=True).clamp(min=1e-5)
        x_n  = (x - mean) / std

        # 2. Patch
        patches = x_n.unfold(-1, self.patch_len, self.stride)  # (B, N, P)

        # 3. Embed
        tok = self.patch_embed(patches)      # (B, N, d_model)
        pos = torch.arange(tok.shape[1], device=x.device)
        tok = self.norm_input(tok + self.pos_embed(pos))

        # 4. Encode
        enc = self.encoder(tok)              # (B, N, d_model)

        # 5. Flatten + head
        out = self.head(enc.flatten(1))      # (B, H)

        # 6. Denormalize
        return out * std + mean

# Test
model = UnivPatchTST(lookback=336, horizon=96)
x     = torch.randn(8, 336)
y     = model(x)
print(f"Input: {x.shape} → Output: {y.shape}")   # (8, 96)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
```

---

*← [02 — Informer/Autoformer/FEDformer](./02_informer_autoformer_fedformer.md) | [Module README](./README.md) | Next: [04 — TimeGPT & Lag-Llama](./04_timegpt_and_lag_llama.md) →*
