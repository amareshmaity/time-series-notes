"""
02_tcn_demo.py
==============
Module 05 — Deep Learning Models
Topic   : Temporal Convolutional Network (TCN)

Covers:
  - CausalConv1d with proper left-padding
  - TCNBlock with weight normalization and residual connection
  - Full TCN architecture with exponential dilation
  - Receptive field calculation
  - TCN vs. LSTM training speed and accuracy comparison
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt
import time

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
BLUE, RED, GREEN = "#2C7BB6", "#D7191C", "#1A9641"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
n = 2000   # longer series to test TCN's long-range advantage
ts_t = np.arange(n)   # named ts_t to avoid shadowing the imported `time` module
series = (
    50 + 0.05*ts_t
    + 25*np.sin(2*np.pi*ts_t/365.25)
    + 12*np.sin(2*np.pi*ts_t/7)
    + np.random.normal(0, 4, n)
).astype(np.float32)

LOOKBACK, HORIZON = 120, 12
split = int(n * 0.8)
train_data, test_data = series[:split], series[split - LOOKBACK:]

class TimeSeriesDataset(Dataset):
    def __init__(self, s, L, H):
        s = torch.tensor(s, dtype=torch.float32)
        self.X = torch.stack([s[i:i+L] for i in range(len(s)-L-H+1)]).unsqueeze(-1)
        self.y = torch.stack([s[i+L:i+L+H] for i in range(len(s)-L-H+1)])
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_dl = DataLoader(TimeSeriesDataset(train_data, LOOKBACK, HORIZON),
                      batch_size=32, shuffle=True, drop_last=True)
test_dl  = DataLoader(TimeSeriesDataset(test_data, LOOKBACK, HORIZON),
                      batch_size=32, shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TCN IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1   = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad))
        self.conv2   = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=pad))
        self.chomp   = lambda x, p: x[:, :, :-p] if p > 0 else x
        self.pad     = pad
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()
        self.skip    = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.relu(self.chomp(self.conv1(x), self.pad))
        out = self.dropout(out)
        out = self.relu(self.chomp(self.conv2(out), self.pad))
        out = self.dropout(out)
        return self.relu(out + self.skip(x))

class TCN(nn.Module):
    def __init__(self, input_size, channels, kernel_size, output_size, dropout=0.2):
        super().__init__()
        layers = []
        n_levels = len(channels)
        for i in range(n_levels):
            in_ch = input_size if i == 0 else channels[i-1]
            layers.append(TCNBlock(in_ch, channels[i], kernel_size, 2**i, dropout))
        self.network = nn.Sequential(*layers)
        self.fc_out  = nn.Linear(channels[-1], output_size)

    def forward(self, x):
        # x: (batch, seq, features) → (batch, features, seq)
        x = x.permute(0, 2, 1)
        out = self.network(x)          # (batch, channels, seq)
        return self.fc_out(out[:, :, -1])   # last timestep


# ─────────────────────────────────────────────────────────────────────────────
# 3. RECEPTIVE FIELD ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("TCN Receptive Field Analysis:")
print(f"{'N Layers':>10} {'RF':>8} {'>= Lookback':>14}")
print("-" * 36)
for n_layers in range(4, 11):
    rf = 1 + (3 - 1) * (2 ** n_layers - 1)
    ok = "✅" if rf >= LOOKBACK else "❌"
    print(f"{n_layers:>10} {rf:>8} {ok:>14}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. LSTM MODEL FOR COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden=64, layers=2, out=12, drop=0.2):
        super().__init__()
        self.lstm   = nn.LSTM(input_size, hidden, layers, batch_first=True,
                              dropout=drop if layers > 1 else 0.0)
        self.fc_out = nn.Sequential(nn.LayerNorm(hidden), nn.ReLU(),
                                    nn.Dropout(drop), nn.Linear(hidden, out))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc_out(out[:, -1, :])


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, train_dl, test_dl, n_epochs=100, lr=1e-3, name=""):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    criterion = nn.HuberLoss(delta=1.0)
    best_val, best_state, patience, ctr = float("inf"), None, 15, 0
    val_hist = []

    t_start = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        for X_b, y_b in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(X_b.to(device)), y_b.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        vl = 0
        with torch.no_grad():
            for X_b, y_b in test_dl:
                vl += criterion(model(X_b.to(device)), y_b.to(device)).item()
        vl /= len(test_dl)
        val_hist.append(vl)

        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ctr = 0
        else:
            ctr += 1

        if epoch % 25 == 0:
            print(f"  [{name}] Epoch {epoch:3d} | val={vl:.5f}")
        if ctr >= patience:
            print(f"  [{name}] Early stop at {epoch}")
            break

    train_time = time.time() - t_start
    model.load_state_dict(best_state)
    return val_hist, train_time

# n_layers = 7 → RF = 255 > LOOKBACK=120
tcn_model  = TCN(input_size=1, channels=[64]*7, kernel_size=3, output_size=HORIZON)
lstm_model = LSTMForecaster(hidden=64, layers=2, out=HORIZON)

print(f"\nTCN params:  {sum(p.numel() for p in tcn_model.parameters()):,}")
print(f"LSTM params: {sum(p.numel() for p in lstm_model.parameters()):,}")

print("\n--- Training TCN ---")
tcn_hist, tcn_time = train_model(tcn_model, train_dl, test_dl, name="TCN")

print("\n--- Training LSTM ---")
lstm_hist, lstm_time = train_model(lstm_model, train_dl, test_dl, name="LSTM")


# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_preds(model, dl):
    model.eval()
    p, a = [], []
    for X_b, y_b in dl:
        p.append(model(X_b.to(device)).cpu().numpy())
        a.append(y_b.numpy())
    return np.concatenate(p), np.concatenate(a)

tcn_preds,  actuals = get_preds(tcn_model,  test_dl)
lstm_preds, _       = get_preds(lstm_model, test_dl)

rmse = lambda a, p: np.sqrt(((a - p)**2).mean())
print(f"\n{'Model':<10} {'RMSE':>8} {'Train Time (s)':>16}")
print("-" * 38)
print(f"{'TCN':<10} {rmse(actuals, tcn_preds):>8.4f} {tcn_time:>16.1f}")
print(f"{'LSTM':<10} {rmse(actuals, lstm_preds):>8.4f} {lstm_time:>16.1f}")
print(f"\nTCN speed advantage: {lstm_time / tcn_time:.1f}× faster")


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(13, 9))
axes[0].plot(tcn_hist,  color=BLUE, linewidth=2, label="TCN Val Loss")
axes[0].plot(lstm_hist, color=RED,  linewidth=2, linestyle="--", label="LSTM Val Loss")
axes[0].set_title("Validation Loss Curves")
axes[0].legend()

n_show = 150
axes[1].plot(actuals[:n_show, 0],  color="black", linewidth=2, label="Actual")
axes[1].plot(tcn_preds[:n_show, 0], color=BLUE, linewidth=1.5, linestyle="--",
             label=f"TCN (RMSE={rmse(actuals,tcn_preds):.3f})")
axes[1].plot(lstm_preds[:n_show, 0], color=RED, linewidth=1.5, linestyle=":",
             label=f"LSTM (RMSE={rmse(actuals,lstm_preds):.3f})")
axes[1].set_title("1-Step Ahead Forecast: TCN vs. LSTM")
axes[1].legend()

plt.suptitle("Temporal Convolutional Network vs. LSTM", fontweight="bold")
plt.tight_layout()
plt.savefig("02_tcn_vs_lstm.png", bbox_inches="tight")
plt.show()

print("\n✅ TCN demo complete.")
