# 01 — Self-Attention for Time Series

> **Module**: 06 Transformers & Foundation Models | **File**: 1 of 6
>
> The Transformer architecture revolutionized NLP in 2017. Adapting it to time series requires rethinking positional encoding, causal masking, and the meaning of "tokens" — turning raw timesteps into rich temporal representations.

---

## Table of Contents

1. [Why Attention for Time Series?](#1-why-attention-for-time-series)
2. [Tokenizing Time Series](#2-tokenizing-time-series)
3. [Positional Encoding for TS](#3-positional-encoding-for-ts)
4. [Causal (Autoregressive) Masking](#4-causal-autoregressive-masking)
5. [Scaled Dot-Product Attention](#5-scaled-dot-product-attention)
6. [Multi-Head Attention](#6-multi-head-attention)
7. [Transformer Encoder Architecture](#7-transformer-encoder-architecture)
8. [Complexity Challenges for Long Sequences](#8-complexity-challenges-for-long-sequences)
9. [Implementation Example](#9-implementation-example)

---

## 1. Why Attention for Time Series?

### 1.1 Limitations of RNNs

| Issue | Description |
|-------|-------------|
| **Sequential bottleneck** | Hidden state is a fixed-size vector — long-range information gets compressed |
| **Vanishing gradients** | Gradients decay exponentially over long sequences (partially fixed by LSTM) |
| **No parallelism** | Each timestep must wait for the previous; training is slow |
| **Long-range dependencies** | Hard to link a forecast to an event 500 steps ago |

### 1.2 What Attention Solves

Self-attention computes a **direct, weighted connection between every pair of timesteps** in a single operation:

```
RNN:  x₁ → h₁ → h₂ → h₃ → ... → hₙ   (sequential, compresses)

Attn: each position attends to ALL positions simultaneously
      attention(xᵢ, xⱼ) = learned weight for how much xᵢ depends on xⱼ
```

Benefits for time series:
- **Long-range dependencies** captured in O(1) path length
- **Parallelizable** training (no sequential dependency)
- **Interpretable** — attention maps show which timesteps matter

---

## 2. Tokenizing Time Series

In NLP, tokens are words. In time series, a "token" is one or more timesteps converted into an embedding vector.

### 2.1 Point Tokenization (One Timestep = One Token)

```python
# Shape: (batch, time_steps, 1) → (batch, time_steps, d_model)
import torch.nn as nn

class PointEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.projection = nn.Linear(1, d_model)   # scalar → d_model

    def forward(self, x):           # x: (B, T, 1)
        return self.projection(x)   # (B, T, d_model)
```

**Limitation**: Too fine-grained for long sequences (e.g., 5000 timesteps × O(T²) attention = infeasible).

### 2.2 Patch Tokenization (PatchTST Approach)

```python
# Split series into non-overlapping patches of length P
# Each patch → one token
# Reduces sequence length from T → T/P

class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.projection = nn.Linear(patch_len, d_model)

    def forward(self, x):              # x: (B, T)
        T = x.shape[-1]
        # Create patches: (B, num_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        return self.projection(x)      # (B, num_patches, d_model)
```

**PatchTST insight**: A patch of 16 timesteps contains local temporal patterns — similar to how a word contains character-level patterns. This is 16× fewer tokens = 256× less attention computation.

### 2.3 Channel Tokenization

```python
# For multivariate TS: each variable = one token
# Shape: (batch, n_vars, time_steps) → (batch, n_vars, d_model)
# Attention operates across variables, not time
```

---

## 3. Positional Encoding for TS

Self-attention is **permutation-invariant** by default — it has no notion of order. Positional encoding injects temporal information.

### 3.1 Sinusoidal Positional Encoding (Original Transformer)

```python
import torch
import numpy as np

def sinusoidal_pe(seq_len, d_model):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)  # even dims → sine
    pe[:, 1::2] = torch.cos(position * div_term)  # odd dims → cosine
    return pe   # (seq_len, d_model)

pe = sinusoidal_pe(500, 64)
print(f"Positional encoding shape: {pe.shape}")  # (500, 64)
```

**Properties**:
- Fixed (not learned) — same encoding for same position
- Each dimension oscillates at a different frequency
- Positions close together have similar encodings

### 3.2 Learnable Positional Encoding

```python
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)  # learned table

    def forward(self, x):           # x: (B, T, d_model)
        positions = torch.arange(x.shape[1], device=x.device)
        return x + self.pe(positions)
```

### 3.3 Temporal Positional Encoding (Calendar-Aware)

Real time series have rich temporal structure (hour of day, day of week, month, etc.):

```python
class TemporalEmbedding(nn.Module):
    """Embed calendar features into positional information."""
    def __init__(self, d_model):
        super().__init__()
        # Each calendar feature gets its own embedding table
        self.hour_embed   = nn.Embedding(24, d_model)
        self.weekday_embed = nn.Embedding(7, d_model)
        self.month_embed  = nn.Embedding(13, d_model)  # 1–12

    def forward(self, hour, weekday, month):  # each: (B, T)
        return self.hour_embed(hour) + self.weekday_embed(weekday) + self.month_embed(month)
```

---

## 4. Causal (Autoregressive) Masking

For **forecasting** (autoregressive generation), position t should only attend to positions ≤ t — no "looking into the future."

```python
def causal_mask(seq_len, device):
    """
    Upper triangular mask (True = masked/forbidden).
    Allows attending to current and past positions only.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask
    # mask[i, j] = True if j > i  (future position — blocked)

# Usage in attention:
# attn_weights = attn_weights.masked_fill(mask, float('-inf'))
# → softmax turns -inf into 0 weight → future positions ignored
```

**When to use causal masking**:
| Scenario | Masking | Reason |
|----------|---------|--------|
| Autoregressive generation (one step at a time) | ✅ Causal | Can't see future at inference |
| Encoder-only forecasting (all at once) | ❌ Bidirectional | Full context for representation |
| Cross-attention (encoder → decoder) | ❌ Causal on decoder only | Encoder sees all; decoder is causal |

---

## 5. Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

- **Q** (Query): "What am I looking for?"
- **K** (Key): "What do I have to offer?"
- **V** (Value): "What information do I carry?"

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, heads, seq_len, d_k)
    mask: (seq_len, seq_len) boolean — True = masked
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    # scores: (batch, heads, seq_len_q, seq_len_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    # attn_weights: (batch, heads, seq_len_q, seq_len_k)

    output = torch.matmul(attn_weights, V)
    # output: (batch, heads, seq_len_q, d_k)

    return output, attn_weights

# Why √d_k scaling?
# Without it, dot products grow large for high d_k → softmax saturates
# → gradients vanish → training stalls
# Dividing by √d_k keeps variance at ~1 regardless of d_k
```

---

## 6. Multi-Head Attention

Instead of one attention head, use **H parallel heads**, each learning different relationship patterns:

```
head_i = Attention(Q·Wᵢq, K·Wᵢk, V·Wᵢv)
MultiHead(Q,K,V) = Concat(head₁, ..., headH) · Wₒ
```

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # (B, T, d_model) → (B, n_heads, T, d_k)
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        out, weights = scaled_dot_product_attention(Q, K, V, mask)
        # out: (B, n_heads, T, d_k) → (B, T, d_model)
        B, _, T, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out), weights
```

**What different heads learn** (empirical findings in TS):
- Head 1: Short-range (yesterday → today)
- Head 2: Weekly pattern (same day last week)
- Head 3: Trend direction
- Head 4: Seasonal peak alignment

---

## 7. Transformer Encoder Architecture

```
Input: (B, T, 1) → Embedding → (B, T, d_model)
         ↓
    Positional Encoding
         ↓
    ┌─────────────────────────────────┐
    │   N × Transformer Block:        │
    │   ┌─ Multi-Head Self-Attention  │
    │   │  + Add & LayerNorm          │
    │   └─ Feed-Forward Network       │
    │      + Add & LayerNorm          │
    └─────────────────────────────────┘
         ↓
    Linear Projection → Forecast (B, H)
```

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # Sub-layer 1: Self-attention with residual
        attn_out, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + attn_out)        # Add & Norm

        # Sub-layer 2: Feed-forward with residual
        x = self.norm2(x + self.ffn(x))     # Add & Norm
        return x

class TSTransformerEncoder(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=3, ffn_dim=256,
                 horizon=24, lookback=96):
        super().__init__()
        self.embedding   = nn.Linear(1, d_model)
        self.pos_encoding = LearnablePositionalEncoding(lookback, d_model)
        self.blocks      = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model * lookback, horizon)

    def forward(self, x):            # x: (B, T, 1)
        x = self.pos_encoding(self.embedding(x))    # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        x = x.flatten(start_dim=1)   # (B, T*d_model)
        return self.head(x)           # (B, horizon)
```

---

## 8. Complexity Challenges for Long Sequences

Standard self-attention has **O(T²)** memory and time complexity:

| Sequence Length T | Attention Pairs | Challenge |
|:-----------------:|:---------------:|-----------|
| 512 | 262K | Fine (NLP-scale) |
| 2,048 | 4.2M | Manageable |
| 8,192 | 67M | Heavy |
| 96,000 | 9.2B | Out of memory |

Time series can be very long (e.g., 15-minute electricity data for 1 year = 35,040 steps).

### Solutions:

| Approach | Model | Complexity | Mechanism |
|----------|-------|-----------|-----------|
| **ProbSparse attention** | Informer | O(T log T) | Only top-k queries attend to all keys |
| **AutoCorrelation** | Autoformer | O(T log T) | FFT-based period finding |
| **Patching** | PatchTST | O((T/P)²) | Treat patches as tokens |
| **Linear attention** | FEDformer | O(T) | Random frequency components |

---

## 9. Implementation Example

```python
import torch

# Quick end-to-end test
B, T, H = 16, 96, 24           # batch, lookback, horizon
model = TSTransformerEncoder(
    d_model=64,
    n_heads=4,
    n_layers=3,
    ffn_dim=256,
    horizon=H,
    lookback=T,
)
x = torch.randn(B, T, 1)       # batch of univariate series
y_pred = model(x)              # (B, H)
print(f"Input:  {x.shape}")    # [16, 96, 1]
print(f"Output: {y_pred.shape}")  # [16, 24]

# Parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")
```

---

## Summary

| Concept | Key Idea |
|---------|----------|
| **Tokenization** | Convert timesteps or patches to embedding vectors |
| **Positional encoding** | Inject temporal order (sinusoidal or learnable or calendar) |
| **Causal mask** | Prevent attention to future timesteps in autoregressive models |
| **Scaled dot-product** | Q·Kᵀ/√d_k — learns which pairs of timesteps are related |
| **Multi-head attention** | H parallel attention heads capture different temporal patterns |
| **O(T²) complexity** | Key bottleneck for long time series — solved by sparse/patch methods |

---

*← [Module README](./README.md) | Next: [02 — Informer, Autoformer & FEDformer](./02_informer_autoformer_fedformer.md) →*
