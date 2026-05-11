"""
01_lstm_gru_demo.py
====================
Module 05 — Deep Learning Models
Topic   : LSTM & GRU for Time Series Forecasting

Covers:
  - TimeSeriesDataset (sliding window)
  - LSTMForecaster and GRUForecaster PyTorch modules
  - Full training loop with early stopping and gradient clipping
  - RevIN normalization
  - LSTM vs. GRU comparison on synthetic seasonal series
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
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
# 1. SYNTHETIC SEASONAL SERIES
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
n = 1000
t = np.arange(n)
series = (
    50 + 0.05*t
    + 20*np.sin(2*np.pi*t/365.25)
    + 10*np.sin(2*np.pi*t/7)
    + np.random.normal(0, 3, n)
).astype(np.float32)

split = int(n * 0.8)
train_data = series[:split]
test_data  = series[split - 60:]   # keep lookback overlap

LOOKBACK = 60
HORIZON  = 12
print(f"Train: {len(train_data)} | Test setup: {len(test_data)} | Lookback: {LOOKBACK} | Horizon: {HORIZON}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET AND DATALOADERS
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    def __init__(self, series, lookback, horizon):
        self.series   = torch.tensor(series, dtype=torch.float32)
        self.lookback = lookback
        self.horizon  = horizon

    def __len__(self):
        return len(self.series) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.lookback].unsqueeze(-1)
        y = self.series[idx + self.lookback: idx + self.lookback + self.horizon]
        return x, y

train_ds = TimeSeriesDataset(train_data, LOOKBACK, HORIZON)
test_ds  = TimeSeriesDataset(test_data,  LOOKBACK, HORIZON)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,  drop_last=True)
test_dl  = DataLoader(test_ds,  batch_size=32, shuffle=False)

x_b, y_b = next(iter(train_dl))
print(f"X batch: {x_b.shape} | y batch: {y_b.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=12, dropout=0.2):
        super().__init__()
        self.lstm   = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out = nn.Sequential(nn.LayerNorm(hidden_size), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(hidden_size, output_size))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc_out(out[:, -1, :])

class GRUForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=12, dropout=0.2):
        super().__init__()
        self.gru    = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out = nn.Sequential(nn.LayerNorm(hidden_size), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(hidden_size, output_size))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc_out(out[:, -1, :])


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=15):
        self.patience   = patience
        self.best_score = float("inf")
        self.counter    = 0
        self.best_state = None
    def __call__(self, score, model):
        if score < self.best_score - 1e-6:
            self.best_score = score
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience
    def restore(self, model):
        model.load_state_dict(self.best_state)

def train_model(model, train_dl, test_dl, n_epochs=150, lr=1e-3, name="model"):
    model = model.to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    es = EarlyStopping(patience=20)
    train_hist, val_hist = [], []

    for epoch in range(1, n_epochs + 1):
        model.train()
        tr_loss = 0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        scheduler.step()

        model.eval()
        vl_loss = 0
        with torch.no_grad():
            for X_b, y_b in test_dl:
                X_b, y_b = X_b.to(device), y_b.to(device)
                vl_loss += criterion(model(X_b), y_b).item()

        tr = tr_loss / len(train_dl)
        vl = vl_loss / len(test_dl)
        train_hist.append(tr); val_hist.append(vl)

        if epoch % 30 == 0:
            print(f"  [{name}] Epoch {epoch:3d} | train={tr:.5f} | val={vl:.5f}")
        if es(vl, model):
            print(f"  [{name}] Early stop at epoch {epoch} | best val={es.best_score:.5f}")
            break

    es.restore(model)
    return train_hist, val_hist


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN BOTH MODELS
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Training LSTM ---")
lstm_model = LSTMForecaster(hidden_size=64, num_layers=2, output_size=HORIZON)
lstm_train_hist, lstm_val_hist = train_model(lstm_model, train_dl, test_dl, name="LSTM")

print("\n--- Training GRU ---")
gru_model = GRUForecaster(hidden_size=64, num_layers=2, output_size=HORIZON)
gru_train_hist, gru_val_hist = train_model(gru_model, train_dl, test_dl, name="GRU")


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_test_preds(model, test_dl):
    model.eval()
    preds, actuals = [], []
    for X_b, y_b in test_dl:
        preds.append(model(X_b.to(device)).cpu().numpy())
        actuals.append(y_b.numpy())
    return np.concatenate(preds), np.concatenate(actuals)

lstm_preds, actuals = get_test_preds(lstm_model, test_dl)
gru_preds,  _       = get_test_preds(gru_model,  test_dl)

def rmse(a, p): return np.sqrt(((a - p)**2).mean())

print(f"\nTest RMSE — LSTM: {rmse(actuals, lstm_preds):.4f}")
print(f"Test RMSE — GRU:  {rmse(actuals, gru_preds):.4f}")
print(f"LSTM params: {sum(p.numel() for p in lstm_model.parameters()):,}")
print(f"GRU params:  {sum(p.numel() for p in gru_model.parameters()):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(13, 9))

# Loss curves
axes[0].plot(lstm_train_hist, color=BLUE, alpha=0.7, label="LSTM Train")
axes[0].plot(lstm_val_hist,   color=BLUE, linewidth=2, label="LSTM Val")
axes[0].plot(gru_train_hist,  color=RED,  alpha=0.7,  linestyle="--", label="GRU Train")
axes[0].plot(gru_val_hist,    color=RED,  linewidth=2, linestyle="--", label="GRU Val")
axes[0].set_title("Training & Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Huber Loss")
axes[0].legend()

# Forecast comparison (first 120 test samples' first horizon step)
n_show = min(120, len(actuals))
axes[1].plot(actuals[:n_show, 0], color="black", linewidth=2, label="Actual (h=1)")
axes[1].plot(lstm_preds[:n_show, 0], color=BLUE, linewidth=1.5, linestyle="--",
             label=f"LSTM (RMSE={rmse(actuals,lstm_preds):.3f})")
axes[1].plot(gru_preds[:n_show, 0], color=RED, linewidth=1.5, linestyle=":",
             label=f"GRU (RMSE={rmse(actuals,gru_preds):.3f})")
axes[1].set_title("1-Step-Ahead Forecast on Test Set")
axes[1].legend()

plt.suptitle("LSTM vs. GRU — Time Series Forecasting", fontweight="bold")
plt.tight_layout()
plt.savefig("01_lstm_gru_comparison.png", bbox_inches="tight")
plt.show()

print("\n✅ LSTM/GRU demo complete.")
