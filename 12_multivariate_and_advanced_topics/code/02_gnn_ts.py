"""
code/02_gnn_ts.py
===================
Module 12 — Multivariate & Advanced Topics
Practical: Spatial-temporal GNN for multivariate forecasting.

Demonstrates:
  - Correlation-based adjacency matrix construction
  - DCRNN-style GRU with graph convolution (pure PyTorch)
  - Spatial-temporal dataset generation and DataLoader
  - Training loop with evaluation
  - Comparison vs. flat LSTM baseline
  - Forecast visualization per node
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error

torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Spatial-Temporal Dataset
# ─────────────────────────────────────────────────────────────────────────────

N_NODES = 12      # number of sensors/nodes
T_STEPS = 400     # time steps
IN_STEPS = 12     # context window
HORIZON  = 6      # forecast horizon
HIDDEN   = 32
N_LAYERS = 2
EPOCHS   = 60
LR       = 1e-3

def make_spatial_ts(N=N_NODES, T=T_STEPS):
    """
    Generate correlated spatial TS with known graph structure.
    Nodes are arranged in a ring; adjacent nodes influence each other.
    """
    t   = np.linspace(0, 8*np.pi, T)
    X   = np.zeros((T, N))

    for i in range(N):
        phase = 2*np.pi * i / N       # nodes have different phases
        freq  = 1 + 0.3 * (i % 3)    # vary frequency slightly
        amp   = 1.0 + 0.2 * (i % 4)
        X[:, i] = amp * np.sin(freq * t + phase)

    # Add spatial influence: each node affected by left neighbor (lag 1)
    X_influenced = X.copy()
    for t_ in range(1, T):
        for i in range(N):
            left = (i - 1) % N
            X_influenced[t_, i] += 0.3 * X[t_-1, left]

    noise = np.random.normal(0, 0.1, X_influenced.shape)
    return (X_influenced + noise).astype(np.float32)

raw = make_spatial_ts()
print(f"Spatial-temporal dataset: {raw.shape} (T={T_STEPS}, N={N_NODES})")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Adjacency Matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_correlation_adj(data, threshold=0.4):
    """Pearson correlation adjacency, row-normalized."""
    df   = pd.DataFrame(data)
    corr = df.corr().abs().values
    A    = np.where(corr > threshold, corr, 0.0)
    np.fill_diagonal(A, 0.0)
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return (A / row_sums).astype(np.float32)

A_np     = build_correlation_adj(raw)
A_tensor = torch.tensor(A_np)
n_edges  = int((A_np > 0).sum())
print(f"Adjacency: {n_edges} edges (threshold=0.4)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset and DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class STDataset(Dataset):
    """Spatial-Temporal sliding window dataset."""
    def __init__(self, data, in_steps, horizon):
        self.X  = torch.tensor(data)   # (T, N)
        self.I  = in_steps
        self.H  = horizon

    def __len__(self):
        return len(self.X) - self.I - self.H + 1

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.I]             # (in_steps, N)
        y = self.X[idx + self.I : idx + self.I + self.H]  # (horizon, N)
        return x.unsqueeze(-1), y   # x: (I, N, 1), y: (H, N)

split   = int(0.8 * T_STEPS)
train_d = STDataset(raw[:split],  IN_STEPS, HORIZON)
val_d   = STDataset(raw[split:],  IN_STEPS, HORIZON)

train_loader = DataLoader(train_d, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_d,   batch_size=16, shuffle=False)
print(f"Train: {len(train_d)} samples | Val: {len(val_d)} samples")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Graph-GRU Model
# ─────────────────────────────────────────────────────────────────────────────

class GraphConv(nn.Module):
    """Simple graph convolution: Y = A X W + b."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.W = nn.Linear(in_ch, out_ch, bias=True)

    def forward(self, X, A):
        """X: (B, N, C), A: (N, N)."""
        return self.W(torch.bmm(A.unsqueeze(0).expand(X.size(0),-1,-1), X))


class GraphGRUCell(nn.Module):
    """GRU cell with graph convolution replacing matrix-vector products."""
    def __init__(self, in_ch, hidden):
        super().__init__()
        self.reset  = GraphConv(in_ch + hidden, hidden)
        self.update = GraphConv(in_ch + hidden, hidden)
        self.cand   = GraphConv(in_ch + hidden, hidden)

    def forward(self, X, H, A):
        XH = torch.cat([X, H], dim=-1)
        r  = torch.sigmoid(self.reset(XH, A))
        u  = torch.sigmoid(self.update(XH, A))
        XH_r = torch.cat([X, r * H], dim=-1)
        c  = torch.tanh(self.cand(XH_r, A))
        return (1 - u) * H + u * c


class GraphTSForecaster(nn.Module):
    """
    Encoder-Decoder with Graph-GRU.
    Input:  (B, T_in, N, 1)
    Output: (B, H, N)
    """
    def __init__(self, N, in_ch=1, hidden=HIDDEN, n_layers=N_LAYERS, horizon=HORIZON):
        super().__init__()
        self.N        = N
        self.hidden   = hidden
        self.n_layers = n_layers
        self.horizon  = horizon

        self.enc = nn.ModuleList([
            GraphGRUCell(in_ch if i == 0 else hidden, hidden)
            for i in range(n_layers)
        ])
        self.dec = nn.ModuleList([
            GraphGRUCell(in_ch if i == 0 else hidden, hidden)
            for i in range(n_layers)
        ])
        self.out = nn.Linear(hidden, 1)

    def forward(self, X, A):
        """X: (B, T_in, N, 1), A: (N, N)."""
        B, T, N, _ = X.shape
        H = [torch.zeros(B, N, self.hidden, device=X.device)
             for _ in range(self.n_layers)]

        # Encode
        for t in range(T):
            x_t = X[:, t]       # (B, N, 1)
            for l, cell in enumerate(self.enc):
                H[l] = cell(x_t, H[l], A)
                x_t  = H[l]

        # Decode
        preds = []
        x_t   = X[:, -1]
        for _ in range(self.horizon):
            for l, cell in enumerate(self.dec):
                H[l] = cell(x_t, H[l], A)
                x_t  = H[l]
            y_hat = self.out(H[-1]).squeeze(-1)  # (B, N)
            preds.append(y_hat)
            x_t = y_hat.unsqueeze(-1)

        return torch.stack(preds, dim=1)   # (B, H, N)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Baseline LSTM (Flat — no graph)
# ─────────────────────────────────────────────────────────────────────────────

class FlatLSTM(nn.Module):
    """Treats N nodes as N independent LSTM channels."""
    def __init__(self, N, in_ch=1, hidden=HIDDEN, horizon=HORIZON):
        super().__init__()
        self.lstm  = nn.LSTM(in_ch, hidden, batch_first=True)
        self.out   = nn.Linear(hidden, horizon)
        self.N     = N

    def forward(self, X, A=None):
        """X: (B, T_in, N, 1) → flatten nodes → LSTM → (B, H, N)."""
        B, T, N, D = X.shape
        x_flat  = X.permute(0, 2, 1, 3).reshape(B*N, T, D)  # (B*N, T, 1)
        h, _    = self.lstm(x_flat)
        y_hat   = self.out(h[:, -1, :])    # (B*N, H)
        return y_hat.reshape(B, N, -1).permute(0, 2, 1)  # (B, H, N)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Training
# ─────────────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
A_dev  = A_tensor.to(device)

def train_model(model, name, epochs=EPOCHS, lr=LR):
    model  = model.to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist   = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            y_hat = model(X_b, A_dev)
            loss  = F.mse_loss(y_hat, y_b)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item()

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                vl_loss += F.mse_loss(model(X_b, A_dev), y_b).item()

        sched.step()
        tr = tr_loss / len(train_loader)
        vl = vl_loss / len(val_loader)
        hist["train"].append(tr); hist["val"].append(vl)

        if (epoch+1) % 10 == 0:
            print(f"  [{name}] Epoch {epoch+1}/{epochs} | Train: {tr:.4f} | Val: {vl:.4f}")

    return model, hist

print(f"\nTraining on {device}")
print("Graph-GRU:")
gnn_model, gnn_hist = train_model(
    GraphTSForecaster(N_NODES, in_ch=1, hidden=HIDDEN, n_layers=N_LAYERS, horizon=HORIZON),
    "GNN"
)

print("\nFlat LSTM:")
lstm_model, lstm_hist = train_model(
    FlatLSTM(N_NODES, in_ch=1, hidden=HIDDEN, horizon=HORIZON),
    "LSTM"
)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_model(model, loader):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            y_hat = model(X_b.to(device), A_dev).cpu().numpy()
            all_true.append(y_b.numpy()); all_pred.append(y_hat)
    yt = np.concatenate(all_true);  yp = np.concatenate(all_pred)
    return {"mae": mean_absolute_error(yt.flatten(), yp.flatten()),
            "rmse": float(np.sqrt(((yt - yp)**2).mean())),
            "yt": yt, "yp": yp}

gnn_eval  = eval_model(gnn_model, val_loader)
lstm_eval = eval_model(lstm_model, val_loader)

print(f"\n{'='*50}")
print(f"{'Model':<15} {'MAE':>8} {'RMSE':>8}")
print(f"{'─'*35}")
print(f"{'Graph-GRU':<15} {gnn_eval['mae']:>8.4f} {gnn_eval['rmse']:>8.4f}")
print(f"{'Flat LSTM':<15} {lstm_eval['mae']:>8.4f} {lstm_eval['rmse']:>8.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Visualization
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Training curves
ax = axes[0, 0]
ax.plot(gnn_hist["train"], color="#2196F3", label="GNN train")
ax.plot(gnn_hist["val"],   color="#2196F3", linestyle="--", label="GNN val")
ax.plot(lstm_hist["train"], color="#FF5722", label="LSTM train")
ax.plot(lstm_hist["val"],   color="#FF5722", linestyle="--", label="LSTM val")
ax.set_title("Training Curves", fontsize=11)
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 2: Adjacency heatmap
ax = axes[0, 1]
im = ax.imshow(A_np, cmap="Blues", aspect="auto")
ax.set_title("Adjacency Matrix (correlation-based)", fontsize=11)
ax.set_xlabel("Node"); ax.set_ylabel("Node")
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel 3: MAE comparison bar
ax = axes[1, 0]
names = ["Graph-GRU", "Flat LSTM"]
maes  = [gnn_eval["mae"], lstm_eval["mae"]]
colors = ["#2196F3", "#FF5722"]
bars   = ax.bar(names, maes, color=colors, width=0.4)
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=11)
ax.set_ylabel("Val MAE"); ax.set_title("Model Comparison — Val MAE", fontsize=11)
ax.grid(alpha=0.3, axis="y")

# Panel 4: Per-node forecasts (GNN)
ax = axes[1, 1]
yt = gnn_eval["yt"][0]   # (H, N) — first sample
yp = gnn_eval["yp"][0]
for node in range(min(4, N_NODES)):
    ax.plot(yt[:, node], color=plt.cm.tab10(node/N_NODES),
            linewidth=1.5, label=f"Node {node} (true)")
    ax.plot(yp[:, node], color=plt.cm.tab10(node/N_NODES),
            linewidth=1.5, linestyle="--")
ax.set_title("GNN Forecast vs. True (4 nodes, 1 sample)", fontsize=11)
ax.set_xlabel("Horizon step"); ax.set_ylabel("Value")
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)

plt.suptitle(f"Spatial-Temporal GNN — N={N_NODES} nodes, H={HORIZON} steps",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("gnn_ts_forecasting.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: gnn_ts_forecasting.png")
