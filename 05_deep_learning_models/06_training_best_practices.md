# 06 — Training Best Practices for DL Time Series

> **Module**: 05 Deep Learning Models | **File**: 6 of 6
>
> Deep learning models are sensitive to training configuration. This file covers the complete set of best practices: normalization, gradient management, learning rate scheduling, regularization, and data scaling — all adapted for time series.

---

## Table of Contents

1. [Data Normalization for Time Series](#1-data-normalization-for-time-series)
2. [Loss Functions for TS](#2-loss-functions-for-ts)
3. [Gradient Clipping](#3-gradient-clipping)
4. [Learning Rate Scheduling](#4-learning-rate-scheduling)
5. [Regularization Techniques](#5-regularization-techniques)
6. [Early Stopping](#6-early-stopping)
7. [Complete Training Template](#7-complete-training-template)

---

## 1. Data Normalization for Time Series

### 1.1 Why Normalization is Critical

DL models are sensitive to input scale. Un-normalized inputs cause:
- Slow convergence (gradients too small or large)
- Saturated activations (tanh/sigmoid ← when |input| >> 1)
- Poor generalization across different series

### 1.2 Strategies

#### Instance Normalization (Recommended)

Normalize each sample independently:

```python
class InstanceNorm:
    """Normalize each lookback window to zero mean, unit variance."""
    def normalize(self, x: torch.Tensor):
        # x: (batch, seq_len, features)
        mean = x.mean(dim=1, keepdim=True)
        std  = x.std(dim=1, keepdim=True).clamp(min=1e-8)
        return (x - mean) / std, mean, std

    def denormalize(self, y_norm, mean, std):
        # y: (batch, horizon)
        return y_norm * std[:, 0, :] + mean[:, 0, :]

norm = InstanceNorm()
X_norm, mean, std = norm.normalize(X_batch)
y_pred_norm = model(X_norm)
y_pred = norm.denormalize(y_pred_norm, mean, std)  # back to original scale
```

#### RevIN (Reversible Instance Normalization) — Best Practice

```python
class RevIN(nn.Module):
    """Reversible Instance Normalization for non-stationary TS."""
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps     = eps
        self.affine  = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode="norm"):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std  = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.std[:, 0, :] + self.mean[:, 0, :]
        return x
```

#### Global Train Set Normalization (Simpler)

```python
# Fit scaler on TRAINING set only — prevent leakage
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()
test_scaled  = scaler.transform(test_series.values.reshape(-1, 1)).flatten()

# After forecasting, inverse transform
y_pred_original = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
```

---

## 2. Loss Functions for TS

### 2.1 Regression Losses

```python
import torch.nn as nn

# MSE — standard, penalizes outliers heavily
criterion_mse = nn.MSELoss()

# MAE (L1) — robust to outliers, but non-smooth at 0
criterion_mae = nn.L1Loss()

# Huber — MSE for small errors, MAE for large (robust)
criterion_huber = nn.HuberLoss(delta=1.0)   # delta = transition point

# MAPE equivalent — scale-invariant (be careful: undefined when y=0)
def mape_loss(y_pred, y_true, eps=1e-8):
    return ((y_true - y_pred).abs() / (y_true.abs() + eps)).mean()
```

### 2.2 Quantile Loss (for Prediction Intervals)

```python
def quantile_loss(y_pred, y_true, quantile):
    """Pinball loss for a single quantile."""
    errors = y_true - y_pred
    return torch.max(quantile * errors, (quantile - 1) * errors).mean()

# Multi-quantile loss
def mql_loss(y_pred, y_true, quantiles=[0.1, 0.5, 0.9]):
    # y_pred: (batch, horizon, n_quantiles)
    losses = []
    for i, q in enumerate(quantiles):
        losses.append(quantile_loss(y_pred[:, :, i], y_true, q))
    return torch.stack(losses).mean()
```

### 2.3 Choosing the Right Loss

| Scenario | Loss |
|----------|------|
| Standard forecasting, few outliers | MSE |
| Series with outliers or spikes | Huber (delta=1.0) |
| Scale-invariant, positive series | MAPE or SMAPE |
| Probabilistic forecasting | Quantile / MQL |
| Count data (demand) | Poisson NLL |

---

## 3. Gradient Clipping

### 3.1 The Exploding Gradient Problem

During BPTT, gradients can grow exponentially:

```
If |eigenvalues of W| > 1:
  ||∂L/∂h(0)|| ≈ C · λ_max^T   → grows exponentially with sequence length T
```

### 3.2 Gradient Clipping by Norm

```python
# After loss.backward(), before optimizer.step()
max_norm = 1.0   # clip gradient norm to this value

# Clip all gradients globally
total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

# Log to monitor: if total_norm >> max_norm consistently → LR too high or model unstable
if total_norm > max_norm * 5:
    print(f"Warning: large gradient norm {total_norm:.2f} → consider reducing LR")

optimizer.step()
```

### 3.3 Gradient Clipping Best Practices

```
max_norm = 1.0    → safe default for most LSTM/GRU models
max_norm = 0.5    → more aggressive, for unstable training
max_norm = 5.0    → permissive, for TCN (generally more stable)

Monitor: track `total_norm` per batch — 
  consistently == max_norm → clipping is active (possibly too aggressive)
  always << max_norm → safe margin, clipping is not interfering
```

---

## 4. Learning Rate Scheduling

### 4.1 Warmup + Cosine Annealing

```python
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Cosine annealing — smooth decay from lr to eta_min over T_max epochs
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# One-cycle — warmup then cosine decay (very effective for fast training)
scheduler_onecycle = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(train_dl),
    epochs=100,
    pct_start=0.3,   # 30% of training for warmup
)

# Reduce on plateau — reduce LR when val_loss stops improving
scheduler_plateau = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7
)

# SGDR — cosine annealing with warm restarts (good for escaping local minima)
scheduler_sgdr = CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)
```

### 4.2 Training Loop with Scheduler

```python
for epoch in range(n_epochs):
    train_loss = train_epoch(model, train_dl, criterion, optimizer)
    val_loss   = evaluate(model, val_dl, criterion)

    # CosineAnnealingLR: step per epoch
    scheduler_cosine.step()

    # ReduceLROnPlateau: step with validation loss
    scheduler_plateau.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1}: train={train_loss:.5f} val={val_loss:.5f} lr={current_lr:.2e}")
```

---

## 5. Regularization Techniques

### 5.1 Dropout

```python
# Standard dropout (Srivastava et al., 2014)
nn.Dropout(p=0.2)   # drops 20% of neurons randomly during training

# Variational dropout for RNNs — same mask across timesteps
nn.LSTM(input_size, hidden_size, dropout=0.2)   # applied between LSTM layers

# Recommended dropout rates:
#   LSTM/GRU:  0.1–0.3
#   TCN:       0.1–0.2 (less dropout needed — already regularized by dilation)
#   Attention: 0.0–0.1 (attention dropout)
```

### 5.2 Weight Decay (L2 Regularization)

```python
# Applied via optimizer — penalizes large weights
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,   # L2 penalty coefficient
)
# Typical values: 1e-5 to 1e-3
```

### 5.3 Layer Normalization

Normalizes activations within each layer — critical for Transformers and attention:

```python
class LSTMForecaster(nn.Module):
    def __init__(self, ...):
        ...
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last = self.layer_norm(out[:, -1, :])   # normalize before projection
        return self.fc_out(last)
```

---

## 6. Early Stopping

```python
class EarlyStopping:
    """Early stopping with model checkpoint saving."""
    def __init__(self, patience=15, min_delta=1e-6, mode="min"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_score = None
        self.counter    = 0
        self.best_state = None

    def __call__(self, score, model):
        improved = (
            (self.best_score is None) or
            (self.mode == "min" and score < self.best_score - self.min_delta) or
            (self.mode == "max" and score > self.best_score + self.min_delta)
        )
        if improved:
            self.best_score = score
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience   # True = stop training

    def restore_best(self, model):
        model.load_state_dict(self.best_state)


# Usage
es = EarlyStopping(patience=20, min_delta=1e-5)
for epoch in range(1000):
    train_loss = train_epoch(model, train_dl, criterion, optimizer)
    val_loss   = evaluate(model, val_dl, criterion)

    if es(val_loss, model):
        print(f"Early stopping at epoch {epoch+1}")
        break

es.restore_best(model)
```

---

## 7. Complete Training Template

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_dl_model(
    model:        nn.Module,
    train_dl:     DataLoader,
    val_dl:       DataLoader,
    n_epochs:     int   = 200,
    lr:           float = 1e-3,
    weight_decay: float = 1e-5,
    max_norm:     float = 1.0,
    patience:     int   = 20,
    device:       str   = "auto",
):
    """Complete DL training pipeline with best practices."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)
    es        = EarlyStopping(patience=patience)

    train_losses, val_losses = [], []

    for epoch in range(1, n_epochs + 1):
        # ── Train ──────────────────────────────────────────
        model.train()
        ep_loss = 0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()

        # ── Validate ────────────────────────────────────────
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b), y_b).item()

        train_l = ep_loss   / len(train_dl)
        val_l   = val_loss  / len(val_dl)
        train_losses.append(train_l)
        val_losses.append(val_l)

        if epoch % 20 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d} | train={train_l:.5f} | val={val_l:.5f} | lr={lr_now:.2e}")

        if es(val_l, model):
            print(f"✅ Early stopping at epoch {epoch} | best val={es.best_score:.5f}")
            break

    es.restore_best(model)
    return model, train_losses, val_losses
```

### Best Practice Summary

| Practice | Setting | Why |
|----------|---------|-----|
| **Normalization** | RevIN or instance norm | Non-stationary TS |
| **Loss** | Huber (delta=1.0) | Robust to outliers |
| **Gradient clip** | max_norm=1.0 | Prevent exploding gradients |
| **LR schedule** | Cosine annealing | Smooth decay |
| **Dropout** | 0.1–0.3 | Prevent overfitting |
| **Weight decay** | 1e-5 | L2 regularization |
| **Early stopping** | patience=15–20 | Prevent overfitting |
| **Batch size** | 32–128 | Balance speed and stability |

---

*← [05 — TFT](./05_temporal_fusion_transformer.md) | [Module README](./README.md) | Next Module: [06 — Transformers & Foundation Models](../06_transformer_and_foundation_models/README.md) →*
