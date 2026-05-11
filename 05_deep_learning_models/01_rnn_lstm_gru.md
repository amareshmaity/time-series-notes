# 01 — RNN, LSTM & GRU

> **Module**: 05 Deep Learning Models | **File**: 1 of 6
>
> Recurrent neural networks were the dominant deep learning architecture for time series before transformers. LSTM and GRU are production-ready, fast to train, and still widely used for short-to-medium horizon forecasting.

---

## Table of Contents

1. [The Vanishing Gradient Problem in Vanilla RNN](#1-the-vanishing-gradient-problem-in-vanilla-rnn)
2. [LSTM — Long Short-Term Memory](#2-lstm--long-short-term-memory)
3. [GRU — Gated Recurrent Unit](#3-gru--gated-recurrent-unit)
4. [Sliding Window Dataset for DL](#4-sliding-window-dataset-for-dl)
5. [LSTM/GRU for Forecasting — Implementation](#5-lstmgru-for-forecasting--implementation)
6. [Multi-Layer and Bidirectional Variants](#6-multi-layer-and-bidirectional-variants)
7. [LSTM vs. GRU — Practical Comparison](#7-lstm-vs-gru--practical-comparison)

---

## 1. The Vanishing Gradient Problem in Vanilla RNN

### 1.1 Vanilla RNN Equations

```
Forward pass at each timestep t:
  h(t) = tanh(W_h · h(t-1) + W_x · x(t) + b)
  y(t) = W_y · h(t) + b_y

Where:
  h(t) = hidden state (memory)
  x(t) = input at time t
  W_h, W_x, W_y = weight matrices
```

### 1.2 Why Gradients Vanish

During backpropagation through time (BPTT), the gradient at layer `t` depends on the product of gradients from all future layers:

```
∂L/∂h(t) = (∂L/∂h(T)) · ∏_{k=t}^{T-1} ∂h(k+1)/∂h(k)

∂h(k+1)/∂h(k) = W_h · diag(1 - tanh²(h(k)))   ← Jacobian matrix

If |eigenvalues of W_h| < 1:
  Product of T Jacobians → 0 exponentially fast
  → Gradients vanish → model cannot learn long-range dependencies

If |eigenvalues| > 1:
  → Gradients explode (solved by gradient clipping)
```

**Result**: Vanilla RNNs struggle to learn dependencies beyond ~10–20 steps.

---

## 2. LSTM — Long Short-Term Memory

### 2.1 The Cell State (Memory)

LSTM solves the vanishing gradient problem by introducing a **cell state** `C(t)` that flows through time with only **additive updates** — no multiplicative decay:

```
Cell state update: C(t) = f(t) ⊙ C(t-1) + i(t) ⊙ g(t)

Where:
  f(t) = forget gate  (what to erase from memory)
  i(t) = input gate   (what new information to write)
  g(t) = candidate    (new candidate values to write)
  ⊙    = element-wise multiplication
```

Because `C(t)` is updated **additively**, gradients can flow back through hundreds of timesteps without vanishing.

### 2.2 Full LSTM Equations

```
Given: x(t) = input, h(t-1) = previous hidden, C(t-1) = previous cell

Forget gate:  f(t) = σ(W_f · [h(t-1), x(t)] + b_f)  → (0,1)
              "What fraction of old memory to keep?"

Input gate:   i(t) = σ(W_i · [h(t-1), x(t)] + b_i)  → (0,1)
              "How much of the new candidate to write?"

Candidate:    g(t) = tanh(W_g · [h(t-1), x(t)] + b_g) → (-1,1)
              "What new memory content to potentially write?"

Output gate:  o(t) = σ(W_o · [h(t-1), x(t)] + b_o)  → (0,1)
              "What part of cell state to expose as output?"

Cell update:  C(t) = f(t) ⊙ C(t-1) + i(t) ⊙ g(t)
Hidden:       h(t) = o(t) ⊙ tanh(C(t))
```

### 2.3 Gate Intuition

| Gate | Role | Learned Behavior Example |
|------|------|--------------------------|
| **Forget `f`** | Erase old memory | "Monthly trend resets at year boundary" |
| **Input `i`** | Control new write | "Large spike → write to memory" |
| **Output `o`** | Expose memory | "Expose trend component for prediction" |

### 2.4 LSTM Implementation in PyTorch

```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, output_size: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,     # input: (batch, seq_len, input_size)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        last_out = out[:, -1, :]      # last timestep: (batch, hidden_size)
        return self.fc_out(last_out)  # (batch, output_size)

model = LSTMForecaster(input_size=1, hidden_size=128,
                       num_layers=2, output_size=12, dropout=0.2)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 3. GRU — Gated Recurrent Unit

### 3.1 GRU Equations

GRU (Cho et al., 2014) merges the forget and input gates into a **reset gate**, eliminating the separate cell state:

```
Reset gate:   r(t) = σ(W_r · [h(t-1), x(t)] + b_r)
              "How much of past hidden state to use in new candidate?"

Update gate:  z(t) = σ(W_z · [h(t-1), x(t)] + b_z)
              "How much to blend old state vs. new candidate?"

Candidate:    h̃(t) = tanh(W · [r(t) ⊙ h(t-1), x(t)] + b)

Update:       h(t) = (1-z(t)) ⊙ h(t-1) + z(t) ⊙ h̃(t)
```

### 3.2 LSTM vs. GRU

| Aspect | LSTM | GRU |
|--------|------|-----|
| **Gates** | 3 (forget, input, output) | 2 (reset, update) |
| **States** | Hidden `h` + Cell `C` | Hidden `h` only |
| **Parameters** | More | ~25% fewer |
| **Training speed** | Slower | Faster |
| **Long sequences** | Slightly better | Slightly worse |

### 3.3 GRU Implementation

```python
class GRUForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, output_size),
        )
    def forward(self, x):
        out, h_n = self.gru(x)   # no cell state
        return self.fc_out(out[:, -1, :])
```

---

## 4. Sliding Window Dataset for DL

All DL forecasting models require a supervised dataset built from a sliding window:

```
Lookback L=7, Horizon H=3:

Row 1: X=[y1,y2,y3,y4,y5,y6,y7]  → y=[y8, y9, y10]
Row 2: X=[y2,y3,y4,y5,y6,y7,y8]  → y=[y9, y10, y11]
...
```

```python
from torch.utils.data import Dataset, DataLoader
import torch

class TimeSeriesDataset(Dataset):
    def __init__(self, series, lookback, horizon):
        self.series   = torch.tensor(series, dtype=torch.float32)
        self.lookback = lookback
        self.horizon  = horizon

    def __len__(self):
        return len(self.series) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.lookback].unsqueeze(-1)  # (L, 1)
        y = self.series[idx + self.lookback : idx + self.lookback + self.horizon]
        return x, y

train_ds = TimeSeriesDataset(train_array, lookback=60, horizon=12)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
x_b, y_b = next(iter(train_dl))
print(f"X: {x_b.shape} | y: {y_b.shape}")   # X: (32,60,1) | y: (32,12)
```

---

## 5. LSTM/GRU for Forecasting — Implementation

```python
import torch.optim as optim

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = LSTMForecaster(1, 128, 2, 12).to(device)
criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

best_val, best_state, patience, ctr = float("inf"), None, 20, 0

for epoch in range(200):
    model.train()
    for X_b, y_b in train_dl:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()
    scheduler.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_b, y_b in val_dl:
            val_loss += criterion(model(X_b.to(device)), y_b.to(device)).item()
    val_loss /= len(val_dl)

    if val_loss < best_val - 1e-6:
        best_val = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        ctr = 0
    else:
        ctr += 1
    if ctr >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_state)
```

---

## 6. Multi-Layer and Bidirectional Variants

```python
# Stacked (deep) LSTM — 3 layers
model_deep = LSTMForecaster(input_size=1, hidden_size=256,
                             num_layers=3, output_size=12, dropout=0.3)

# Bidirectional LSTM — NOT for forecasting (looks into future = leakage)
# Use ONLY for: anomaly detection, imputation, classification on historical data
bilstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2,
                  batch_first=True, bidirectional=True)
# Output: (batch, seq_len, hidden_size*2) — forward + backward concatenated
```

> ⚠️ **Never use bidirectional LSTM for multi-step forecasting.** It sees future values during forward pass — this is a data leakage bug, not a feature.

---

## 7. LSTM vs. GRU — Practical Comparison

| Scenario | Prefer LSTM | Prefer GRU |
|----------|------------|-----------|
| Very long sequences (> 500 steps) | ✅ | |
| Need faster training | | ✅ |
| Memory constrained | | ✅ |
| Default starting model | ✅ | ✅ (both fine) |
| Ablation studies | | ✅ (simpler to debug) |

**Rule of thumb**: Start with GRU (fewer parameters, faster), then try LSTM if GRU underfits. The difference in performance is usually < 2% RMSE in practice.

---

*← [Module README](./README.md) | Next: [02 — TCN](./02_temporal_convolutional_networks.md) →*
