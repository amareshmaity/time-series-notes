"""
03_patchtst_demo.py
====================
Module 06 — Transformers & Foundation Models
Topic   : PatchTST — Patch-Based Transformer for Time Series

Covers:
  - PatchTST architecture from scratch (educational)
  - Channel-independent patch tokenization
  - RevIN normalization
  - Training loop with early stopping
  - Comparison: PatchTST vs. LSTM vs. DLinear
"""

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
BLUE, RED, GREEN = "#2C7BB6", "#D7191C", "#1A9641"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC MULTIVARIATE DATASET
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
n_vars  = 3
n       = 2000
t       = np.arange(n)

# Three correlated channels
data = np.stack([
    100 + 0.05*t + 25*np.sin(2*np.pi*t/365) + 10*np.sin(2*np.pi*t/7) + np.random.normal(0, 4, n),
    200 + 0.10*t + 40*np.sin(2*np.pi*t/365 + 0.5) + np.random.normal(0, 6, n),
    50  + 0.02*t + 15*np.sin(2*np.pi*t/365 + 1.0) + 5*np.sin(2*np.pi*t/7) + np.random.normal(0, 2, n),
], axis=1).astype(np.float32)   # shape: (n, n_vars)

LOOKBACK = 336
HORIZON  = 96
split    = int(n * 0.8)
train_data = data[:split]
test_data  = data[split - LOOKBACK:]

print(f"Data shape: {data.shape} | Train: {split} | Test: {n - split}")
print(f"Lookback: {LOOKBACK} | Horizon: {HORIZON}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────────────────────────────────────

class MultivariateDataset(Dataset):
    def __init__(self, data: np.ndarray, lookback: int, horizon: int):
        self.data     = torch.tensor(data, dtype=torch.float32)
        self.lookback = lookback
        self.horizon  = horizon

    def __len__(self):
        return len(self.data) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.lookback]              # (L, C)
        y = self.data[idx + self.lookback : idx + self.lookback + self.horizon]  # (H, C)
        return x, y

train_ds = MultivariateDataset(train_data, LOOKBACK, HORIZON)
test_ds  = MultivariateDataset(test_data,  LOOKBACK, HORIZON)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
test_dl  = DataLoader(test_ds,  batch_size=32, shuffle=False)

x_b, y_b = next(iter(train_dl))
print(f"X batch: {x_b.shape} | y batch: {y_b.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

class RevIN(nn.Module):
    """Reversible Instance Normalization — normalizes per-instance per-channel."""
    def __init__(self, n_vars: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, 1, n_vars))
        self.beta  = nn.Parameter(torch.zeros(1, 1, n_vars))
        self.eps   = eps

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std  = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self._mean) / self._std
            return x * self.gamma + self.beta
        elif mode == "denorm":
            x = (x - self.beta) / (self.gamma + self.eps)
            return x * self._std + self._mean
        raise ValueError(f"mode must be 'norm' or 'denorm', got '{mode}'")


class PatchTST(nn.Module):
    """
    PatchTST: Patch-Based Channel-Independent Transformer.
    Each channel processed independently with shared weights.
    """
    def __init__(self, n_vars: int, lookback: int, horizon: int,
                 patch_len: int = 16, stride: int = 8,
                 d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 3, d_ff: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.n_vars   = n_vars
        self.horizon  = horizon
        self.revin    = RevIN(n_vars)
        self.patch_len = patch_len
        self.stride    = stride

        n_patches = (lookback - patch_len) // stride + 1
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed   = nn.Embedding(n_patches, d_model)
        self.dropout     = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head    = nn.Linear(n_patches * d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) → (B, H, C)"""
        x = self.revin(x, "norm")                 # instance normalize
        outputs = []
        for v in range(self.n_vars):
            x_v     = x[:, :, v]                  # (B, L)
            patches  = x_v.unfold(-1, self.patch_len, self.stride)  # (B, N, P)
            tok      = self.dropout(self.patch_embed(patches))       # (B, N, d_model)
            pos      = torch.arange(tok.shape[1], device=x.device)
            tok      = tok + self.pos_embed(pos)
            enc      = self.encoder(tok)           # (B, N, d_model)
            out_v    = self.head(enc.flatten(1))   # (B, H)
            outputs.append(out_v)

        y = torch.stack(outputs, dim=-1)           # (B, H, C)
        return self.revin(y, "denorm")


class DLinear(nn.Module):
    """DLinear: Decomposition + separate linear projections per channel."""
    def __init__(self, n_vars: int, lookback: int, horizon: int,
                 kernel_size: int = 25):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.avg   = nn.AvgPool1d(kernel_size, stride=1, padding=pad)
        self.trend = nn.Linear(lookback, horizon)
        self.seas  = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) → (B, H, C)"""
        x_t  = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)  # trend
        x_s  = x - x_t                                          # seasonal
        t_fc = self.trend(x_t.permute(0, 2, 1)).permute(0, 2, 1)
        s_fc = self.seas(x_s.permute(0, 2, 1)).permute(0, 2, 1)
        return t_fc + s_fc


class LSTMForecaster(nn.Module):
    """Multi-channel LSTM forecaster for comparison."""
    def __init__(self, n_vars: int, hidden: int, layers: int, horizon: int):
        super().__init__()
        self.lstm = nn.LSTM(n_vars, hidden, layers, batch_first=True,
                            dropout=0.2 if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_vars * horizon)
        self.horizon = horizon
        self.n_vars  = n_vars

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) → (B, H, C)"""
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).view(-1, self.horizon, self.n_vars)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, train_dl, test_dl, n_epochs=60, lr=1e-3, name="model"):
    model = model.to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    best_val, best_state, patience, ctr = float("inf"), None, 10, 0
    val_hist = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for X, y in test_dl:
                vl += criterion(model(X.to(device)), y.to(device)).item()
        vl /= len(test_dl)
        val_hist.append(vl)

        if vl < best_val - 1e-6:
            best_val   = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ctr = 0
        else:
            ctr += 1

        if epoch % 20 == 0:
            print(f"  [{name}] Epoch {epoch:3d} | val={vl:.5f}")
        if ctr >= patience:
            print(f"  [{name}] Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return val_hist


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN ALL MODELS
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Training PatchTST ---")
patchtst = PatchTST(n_vars=n_vars, lookback=LOOKBACK, horizon=HORIZON,
                    patch_len=16, stride=8, d_model=128, n_heads=8,
                    n_layers=3, d_ff=256, dropout=0.1)
ptst_hist = train_model(patchtst, train_dl, test_dl, n_epochs=80, lr=1e-3, name="PatchTST")

print("\n--- Training DLinear ---")
dlinear  = DLinear(n_vars=n_vars, lookback=LOOKBACK, horizon=HORIZON)
dlin_hist = train_model(dlinear, train_dl, test_dl, n_epochs=80, lr=1e-3, name="DLinear")

print("\n--- Training LSTM ---")
lstm_model = LSTMForecaster(n_vars=n_vars, hidden=128, layers=2, horizon=HORIZON)
lstm_hist  = train_model(lstm_model, train_dl, test_dl, n_epochs=80, lr=1e-3, name="LSTM")


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_preds(model, dl):
    model.eval()
    preds, acts = [], []
    for X, y in dl:
        preds.append(model(X.to(device)).cpu().numpy())
        acts.append(y.numpy())
    return np.concatenate(preds), np.concatenate(acts)

ptst_preds, actuals = get_preds(patchtst, test_dl)
dlin_preds, _       = get_preds(dlinear,  test_dl)
lstm_preds, _       = get_preds(lstm_model, test_dl)

rmse = lambda a, p: np.sqrt(((a - p)**2).mean())
print(f"\n{'Model':<15} {'RMSE (all channels)'}")
print("-" * 35)
print(f"{'PatchTST':<15} {rmse(actuals, ptst_preds):.5f}")
print(f"{'DLinear':<15} {rmse(actuals, dlin_preds):.5f}")
print(f"{'LSTM':<15} {rmse(actuals, lstm_preds):.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(13, 10))

# Validation loss curves
axes[0].plot(ptst_hist, color=BLUE,  linewidth=2, label="PatchTST")
axes[0].plot(dlin_hist, color=GREEN, linewidth=2, linestyle="--", label="DLinear")
axes[0].plot(lstm_hist, color=RED,   linewidth=2, linestyle=":",  label="LSTM")
axes[0].set_title("Validation Loss Curves")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Huber Loss")
axes[0].legend()

# Channel 0 forecast (first 96 test windows, horizon step 1)
n_show = min(100, len(actuals))
axes[1].plot(actuals[:n_show, 0, 0],    color="black", linewidth=2, label="Actual (ch0, h=1)")
axes[1].plot(ptst_preds[:n_show, 0, 0], color=BLUE,  linewidth=1.5, linestyle="--",
             label=f"PatchTST ({rmse(actuals, ptst_preds):.3f})")
axes[1].plot(dlin_preds[:n_show, 0, 0], color=GREEN, linewidth=1.5, linestyle=":",
             label=f"DLinear ({rmse(actuals, dlin_preds):.3f})")
axes[1].plot(lstm_preds[:n_show, 0, 0], color=RED,   linewidth=1.5, linestyle="-.",
             label=f"LSTM ({rmse(actuals, lstm_preds):.3f})")
axes[1].set_title("1-Step Ahead: Channel 0")
axes[1].legend()

plt.suptitle("PatchTST vs. DLinear vs. LSTM — Multivariate TS Forecasting",
             fontweight="bold")
plt.tight_layout()
plt.savefig("03_patchtst_comparison.png", bbox_inches="tight")
plt.show()

print("\n✅ PatchTST demo complete.")
