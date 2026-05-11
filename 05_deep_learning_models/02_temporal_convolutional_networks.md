# 02 — Temporal Convolutional Networks (TCN)

> **Module**: 05 Deep Learning Models | **File**: 2 of 6
>
> TCNs replace recurrence with **dilated causal convolutions**, enabling parallel training over long sequences without the vanishing gradient problem. On many benchmarks, TCN outperforms LSTM/GRU while training 2–5× faster.

---

## Table of Contents

1. [Motivation: Limitations of RNNs](#1-motivation-limitations-of-rnns)
2. [Causal Convolutions](#2-causal-convolutions)
3. [Dilated Convolutions and Receptive Field](#3-dilated-convolutions-and-receptive-field)
4. [Residual TCN Block](#4-residual-tcn-block)
5. [Full TCN Architecture](#5-full-tcn-architecture)
6. [TCN vs. LSTM Comparison](#6-tcn-vs-lstm-comparison)

---

## 1. Motivation: Limitations of RNNs

| RNN Limitation | How TCN Solves It |
|---------------|------------------|
| Sequential computation (no parallelism) | Convolutions are fully parallel |
| Vanishing gradients over long sequences | Skip connections + dilations avoid deep chains |
| Fixed memory bottleneck (hidden state) | Receptive field grows with layers, no compression |
| Hard to train with very long lookbacks | Dilated convolutions reach thousands of steps |

---

## 2. Causal Convolutions

### 2.1 Standard vs. Causal Convolution

A **causal** convolution ensures the output at time `t` only depends on inputs at time `t` and earlier — no future leakage:

```
Standard 1D conv (NOT causal):
  y(t) = Σₖ w(k) · x(t - k + ⌊K/2⌋)   ← uses past AND future

Causal 1D conv:
  y(t) = Σₖ₌₀^{K-1} w(k) · x(t - k)   ← uses ONLY past (and present)
```

In PyTorch, causal convolution is achieved by **left-padding** the input by `kernel_size - 1`:

```python
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation   # left-pad only
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0,
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = nn.functional.pad(x, (self.pad, 0))   # pad left only
        return self.conv(x)
```

---

## 3. Dilated Convolutions and Receptive Field

### 3.1 Dilation

A dilated convolution inserts **gaps** between filter elements, allowing each output to "see" a wider input range without increasing the kernel size:

```
Kernel size K=3, dilation d:

d=1: filter taps at [t, t-1, t-2]         ← standard
d=2: filter taps at [t, t-2, t-4]         ← skip every 1
d=4: filter taps at [t, t-4, t-8]         ← skip every 3
d=8: filter taps at [t, t-8, t-16]

Receptive field with exponentially increasing dilations:
  Layer 1 (d=1): sees 3 steps
  Layer 2 (d=2): sees 7 steps   (3 + 4 new steps)
  Layer 3 (d=4): sees 15 steps
  Layer 4 (d=8): sees 31 steps
  ...
  Layer n (d=2^{n-1}): sees 2^n * (K-1) + 1 steps total
```

### 3.2 Receptive Field Calculation

```
TCN receptive field = 1 + (K-1) · Σᵢ dᵢ
                    = 1 + (K-1) · (2⁰ + 2¹ + ... + 2^{L-1})
                    = 1 + (K-1) · (2^L - 1)

Example: K=3, L=8 layers, dilations=[1,2,4,8,16,32,64,128]
  RF = 1 + (3-1) · (2^8 - 1) = 1 + 2 · 255 = 511 time steps!
```

Design rule: **ensure RF ≥ lookback window**:

```python
def compute_rf(n_layers, kernel_size, dilation_base=2):
    return 1 + (kernel_size - 1) * (dilation_base ** n_layers - 1)

# Design for 365-day lookback
for n_layers in range(4, 12):
    rf = compute_rf(n_layers, kernel_size=3)
    if rf >= 365:
        print(f"n_layers={n_layers} → RF={rf} ≥ 365 ✅")
        break
```

---

## 4. Residual TCN Block

### 4.1 Block Architecture

```
Input x ──→ CausalConv1d (d=dilation) ──→ WeightNorm ──→ ReLU ──→ Dropout
         ──→ CausalConv1d (d=dilation) ──→ WeightNorm ──→ ReLU ──→ Dropout
         + 1×1 Conv (residual, if in≠out channels)
         ──→ ReLU ──→ Output
```

### 4.2 Implementation

```python
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           dilation=dilation, padding=pad))
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                           dilation=dilation, padding=pad))
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()
        self.chomp   = lambda x: x[:, :, :-pad]   # remove right padding

        # Residual 1×1 conv (only if channels differ)
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.relu(self.chomp(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.chomp(self.conv2(out)))
        out = self.dropout(out)
        return self.relu(out + self.residual(x))
```

---

## 5. Full TCN Architecture

```python
class TCN(nn.Module):
    def __init__(self, input_size, hidden_channels, kernel_size,
                 output_size, dropout=0.2, dilation_base=2):
        super().__init__()
        layers = []
        n_levels = len(hidden_channels)

        for i in range(n_levels):
            in_ch  = input_size if i == 0 else hidden_channels[i-1]
            out_ch = hidden_channels[i]
            dil    = dilation_base ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dil, dropout))

        self.network = nn.Sequential(*layers)
        self.fc_out  = nn.Linear(hidden_channels[-1], output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size) → (batch, input_size, seq_len)
        x = x.permute(0, 2, 1)
        out = self.network(x)             # (batch, hidden, seq_len)
        out = out[:, :, -1]               # last timestep: (batch, hidden)
        return self.fc_out(out)           # (batch, output_size)


# Instantiate: 8-layer TCN, RF ≥ 511 steps
tcn_model = TCN(
    input_size=1,
    hidden_channels=[64] * 8,   # 8 layers, each 64 channels
    kernel_size=3,
    output_size=12,
    dropout=0.2,
)
print(f"TCN params: {sum(p.numel() for p in tcn_model.parameters()):,}")
```

---

## 6. TCN vs. LSTM Comparison

| Aspect | LSTM | TCN |
|--------|------|-----|
| **Training** | Sequential | Parallel (faster) |
| **Very long lookback** | Degrades | Handles naturally via dilation |
| **Gradient flow** | Gated (decent) | Direct (skip connections) |
| **Interpretability** | Low | Low |
| **Sequence length flexibility** | Fixed hidden state | Flexible RF via architecture |
| **Hyperparameter tuning** | hidden_size, num_layers | n_layers, kernel_size, channels |
| **Typical performance** | Strong | Matches or beats LSTM |

> **Rule**: Try TCN when LSTM is slow to train or when you need very long lookback windows (> 200 steps). TCN trains 2–5× faster on modern GPUs due to full parallelism.

---

*← [01 — LSTM/GRU](./01_rnn_lstm_gru.md) | [Module README](./README.md) | Next: [03 — Seq2Seq & Attention](./03_seq2seq_and_attention.md) →*
