# 04 — Graph Neural Networks for Time Series

> **Module**: 12 Multivariate & Advanced Topics | **File**: 4 of 6
>
> When time series come from sensors arranged in a network — traffic sensors on roads, electricity substations on a grid, weather stations across a region — the spatial relationships carry predictive signal that temporal models ignore. Graph Neural Networks encode this topology explicitly, propagating information along edges to improve forecasts.

---

## Table of Contents

1. [Why Graphs for Time Series?](#1-why-graphs-for-time-series)
2. [Graph Representation of TS Networks](#2-graph-representation-of-ts-networks)
3. [DCRNN — Diffusion Convolutional RNN](#3-dcrnn--diffusion-convolutional-rnn)
4. [WaveNet on Graphs (Graph WaveNet)](#4-wavenet-on-graphs-graph-wavenet)
5. [STGCN — Spatio-Temporal GCN](#5-stgcn--spatio-temporal-gcn)
6. [Graph Construction Strategies](#6-graph-construction-strategies)
7. [PyG Temporal Implementation](#7-pyg-temporal-implementation)

---

## 1. Why Graphs for Time Series?

### 1.1 The Spatial Dependency Problem

```
Standard multivariate models (VAR, TFT, Crossformer):
  Treat all D series as a flat feature vector.
  Implicitly learn pairwise relationships from data.
  Scale as O(D²) — impractical for D > 100.

GNN-based models:
  Encode spatial structure as a graph G = (V, E, A).
  V = sensors/nodes; E = spatial connections; A = adjacency matrix.
  Message passing along edges: only connected nodes exchange information.
  Scale as O(|E|) — sparse graphs → efficient for D = 207, 1000+.

Example: Traffic forecasting on METR-LA
  207 road sensors (nodes)
  Edges: sensors within 400m of each other
  Adjacency: road network distance (not Euclidean!)
  
  Standard LSTM: treats 207 sensors as 207 independent streams.
  DCRNN: propagates traffic information along road topology.
  → 15% lower MAE on 60-min forecast horizon.
```

---

## 2. Graph Representation of TS Networks

### 2.1 Graph Components

```
Graph G = (V, E, A)
  V = {v₁, ..., vᴺ}   — N nodes (sensors/entities)
  E ⊆ V × V            — edges (physical or learned connections)
  A ∈ ℝᴺˣᴺ            — adjacency matrix (A_ij = weight of edge i→j)

Node features at time t:
  X_t ∈ ℝᴺˣᴰ         — each node has D features at time t

Spatial-temporal input:
  X ∈ ℝᴺˣᵀˣᴰ         — N nodes × T time steps × D features

Prediction target:
  Ŷ ∈ ℝᴺˣᴴ           — forecast next H steps for each node
```

### 2.2 Adjacency Matrix Construction

```python
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def build_distance_adjacency(
    locations: np.ndarray,   # (N, 2) lat/lon or x/y coordinates
    threshold_km: float = 2.0,
    sigma: float = 0.5,
) -> np.ndarray:
    """
    Build adjacency matrix from spatial distances.

    A_ij = exp(-d(i,j)² / σ²) if d(i,j) ≤ threshold else 0

    Parameters
    ----------
    locations    : (N, 2) array of coordinates
    threshold_km : maximum connection distance
    sigma        : decay parameter for Gaussian kernel

    Returns
    -------
    A : (N, N) weighted adjacency matrix (row-normalized)
    """
    D = cdist(locations, locations, metric="euclidean")
    A = np.exp(-D**2 / sigma**2)
    A[D > threshold_km] = 0.0
    np.fill_diagonal(A, 0.0)   # no self-loops

    # Row-normalize: D^{-1} A
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return A / row_sums


def build_correlation_adjacency(
    series_df: pd.DataFrame,
    threshold: float = 0.5,
    method: str = "pearson",
) -> np.ndarray:
    """
    Build adjacency matrix from temporal correlations.
    Useful when no spatial information is available.

    A_ij = |corr(sᵢ, sⱼ)| if > threshold else 0
    """
    corr = series_df.corr(method=method).abs().values
    A    = np.where(corr > threshold, corr, 0.0)
    np.fill_diagonal(A, 0.0)
    return A


def add_learned_adjacency(N: int, device: str = "cpu") -> "nn.Parameter":
    """
    Learnable adjacency matrix — initialized randomly, trained end-to-end.
    As in Graph WaveNet: A_adaptive = softmax(E₁ · E₂ᵀ).

    E₁, E₂: (N, d_e) embedding matrices (trainable parameters).
    """
    import torch
    import torch.nn as nn
    E1 = nn.Parameter(torch.randn(N, 10))   # node embedding 1
    E2 = nn.Parameter(torch.randn(10, N))   # node embedding 2
    return E1, E2

def compute_adaptive_adj(E1, E2, topk: int = 10):
    """Compute adaptive adjacency from learnable embeddings."""
    import torch
    import torch.nn.functional as F
    A_raw = F.relu(torch.mm(E1, E2))   # (N, N), non-negative
    # Top-k sparsification
    vals, _ = A_raw.topk(topk, dim=1)
    thresh  = vals[:, -1:].detach()
    A       = F.softmax(torch.where(A_raw >= thresh, A_raw, torch.full_like(A_raw, -1e9)), dim=1)
    return A
```

---

## 3. DCRNN — Diffusion Convolutional RNN

### 3.1 Architecture

```
DCRNN (Li et al., 2018):
  Key idea: Replace standard RNN matrix multiplications (WH, WX)
            with graph diffusion convolutions.

  Diffusion process on graph:
    Σₖ Θₖ (D^{-1}A)ᵏ X     (forward diffusion, k-hop)
    + Σₖ Θₖ (D^{-T}Aᵀ)ᵏ X  (backward diffusion)

  DCGRU cell:
    r_t = σ(DiffConv(X_t, H_{t-1}; θᵣ))         [reset gate]
    u_t = σ(DiffConv(X_t, H_{t-1}; θᵤ))         [update gate]
    c_t = tanh(DiffConv(X_t, r_t ⊙ H_{t-1}; θ_c)) [candidate]
    H_t = (1 - u_t) ⊙ H_{t-1} + u_t ⊙ c_t

  Encoder-Decoder:
    Encoder: process input sequence → final hidden states
    Decoder: autoregressively generates H-step forecast
    Scheduled sampling: mix ground truth and predictions during training
```

### 3.2 Simplified DCGRU Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionConv(nn.Module):
    """
    Diffusion convolution on a graph.
    Approximates graph diffusion as sum of K-hop powers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 2,       # diffusion steps
        bias: bool = True,
    ):
        super().__init__()
        self.K        = K
        # 2K+1 weight matrices: K forward + K backward + 1 identity
        self.weights  = nn.Parameter(torch.randn(2*K+1, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        X : (batch, N, in_channels)
        A : (N, N) normalized adjacency (forward direction)

        Returns (batch, N, out_channels)
        """
        B, N, _ = X.shape
        A_T      = A.T   # backward direction

        out = torch.zeros(B, N, self.weights.shape[-1], device=X.device)

        # Identity (k=0)
        out += X @ self.weights[0]

        X_fwd = X
        X_bwd = X
        for k in range(1, self.K + 1):
            X_fwd = torch.einsum("bn,bnc->bnc", A.unsqueeze(0).expand(B,-1,-1).reshape(B*N,N).mean(0), X_fwd)
            # Simplified: direct matrix multiply
            X_fwd  = (A.unsqueeze(0) @ X_fwd)   # (B, N, C)
            X_bwd  = (A_T.unsqueeze(0) @ X_bwd)
            out   += X_fwd @ self.weights[k]
            out   += X_bwd @ self.weights[K + k]

        if self.bias is not None:
            out += self.bias
        return out


class DCGRUCell(nn.Module):
    """Single DCGRU cell — GRU with graph diffusion convolution."""

    def __init__(self, in_channels: int, hidden_channels: int, K: int = 2):
        super().__init__()
        self.hidden_channels = hidden_channels
        # Reset gate
        self.reset  = DiffusionConv(in_channels + hidden_channels, hidden_channels, K)
        # Update gate
        self.update = DiffusionConv(in_channels + hidden_channels, hidden_channels, K)
        # Candidate
        self.cand   = DiffusionConv(in_channels + hidden_channels, hidden_channels, K)

    def forward(self, X: torch.Tensor, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        X : (batch, N, in_channels)
        H : (batch, N, hidden_channels)
        A : (N, N) normalized adjacency
        """
        XH = torch.cat([X, H], dim=-1)
        r  = torch.sigmoid(self.reset(XH, A))
        u  = torch.sigmoid(self.update(XH, A))
        XH_r = torch.cat([X, r * H], dim=-1)
        c  = torch.tanh(self.cand(XH_r, A))
        H  = (1 - u) * H + u * c
        return H


class DCRNN(nn.Module):
    """
    DCRNN encoder-decoder for spatial-temporal forecasting.

    Input:  (batch, T, N, D) — T steps, N nodes, D features
    Output: (batch, H, N, 1) — H-step forecast per node
    """

    def __init__(
        self,
        n_nodes:  int,
        in_ch:    int = 1,
        hidden:   int = 64,
        n_layers: int = 2,
        horizon:  int = 12,
        K:        int = 2,
    ):
        super().__init__()
        self.horizon  = horizon
        self.n_layers = n_layers
        self.hidden   = hidden

        # Encoder
        self.enc_cells = nn.ModuleList([
            DCGRUCell(in_ch if i==0 else hidden, hidden, K)
            for i in range(n_layers)
        ])
        # Decoder
        self.dec_cells = nn.ModuleList([
            DCGRUCell(in_ch if i==0 else hidden, hidden, K)
            for i in range(n_layers)
        ])
        self.output = nn.Linear(hidden, 1)

    def forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        B, T, N, D = X.shape

        # Encode
        H = [torch.zeros(B, N, self.hidden, device=X.device)
             for _ in range(self.n_layers)]
        for t in range(T):
            x_t = X[:, t]    # (B, N, D)
            for l, cell in enumerate(self.enc_cells):
                H[l] = cell(x_t, H[l], A)
                x_t  = H[l]

        # Decode (autoregressive)
        preds  = []
        x_t    = X[:, -1, :, :1]   # use last known value as first input
        for h in range(self.horizon):
            for l, cell in enumerate(self.dec_cells):
                H[l] = cell(x_t, H[l], A)
                x_t  = H[l]
            y_hat = self.output(H[-1])   # (B, N, 1)
            preds.append(y_hat)
            x_t = y_hat

        return torch.stack(preds, dim=1)   # (B, H, N, 1)
```

---

## 4. WaveNet on Graphs (Graph WaveNet)

### 4.1 Architecture

```
Graph WaveNet (Wu et al., 2019):
  Key innovations:
    1. Adaptive adjacency matrix (learned from data — no prior graph needed)
    2. Dilated causal convolutions (WaveNet-style) for temporal dependencies
    3. Bidirectional graph diffusion

  Temporal component:
    Gated dilated causal convolutions:
      (tanh × σ) activation at multiple dilation rates [1,2,4,8,...,512]
    → Very long temporal receptive fields with few parameters

  Spatial component:
    Graph convolution with adaptive + fixed adjacency:
      h = Σₖ θₖ (D⁻¹A)ᵏ X + θ_ad (A_adapt)ᵏ X
    Adaptive adjacency: A_adapt = softmax(E₁ · E₂ᵀ)  (learnable)

  Why it works:
    Dilated convolutions → efficient long-range temporal patterns
    Learned adjacency → discovers spatial relationships from data
    → State-of-the-art on METR-LA and PEMS-BAY benchmarks
```

---

## 5. STGCN — Spatio-Temporal GCN

### 5.1 Architecture

```
STGCN (Yu et al., 2018) interleaves:
  1. Graph convolution: spatial (node) dimension
  2. 1D convolution:    temporal dimension

ST-Conv Block:
  Input X → Temporal Conv → Graph Conv → Temporal Conv → Output
  
  Each temporal conv: gated activation (tanh × σ)
  Graph conv: Chebyshev spectral approximation (K-order polynomial)

Advantage over DCRNN:
  ✅ All-convolutional (no recurrence → parallelizable across time)
  ✅ Faster training than DCRNN
  ✅ Easier to scale

Disadvantage:
  ❌ Fixed temporal receptive field (no adaptive attention)
  ❌ Less flexible than DCRNN for irregular sampling
```

---

## 6. Graph Construction Strategies

```python
def build_graph_from_correlation(
    df: pd.DataFrame,
    window: int = None,
    threshold: float = 0.6,
    topk: int = None,
) -> np.ndarray:
    """
    Build dynamic or static correlation-based adjacency matrix.

    Parameters
    ----------
    df        : (T, N) DataFrame of N time series
    window    : if not None, use rolling correlation over this window
    threshold : minimum correlation for edge inclusion
    topk      : if set, keep only top-k edges per node

    Returns
    -------
    A : (N, N) adjacency matrix
    """
    if window:
        # Dynamic: use last `window` steps
        corr = df.iloc[-window:].corr().abs().values
    else:
        corr = df.corr().abs().values

    A = corr.copy()
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 0.0)

    if topk:
        for i in range(len(A)):
            row    = A[i].copy()
            sorted_idx = np.argsort(row)[::-1]
            mask   = np.zeros_like(row)
            mask[sorted_idx[:topk]] = 1.0
            A[i] *= mask

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    d     = A.sum(axis=1)
    d_inv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv = np.diag(d_inv)
    return D_inv @ A @ D_inv
```

---

## 7. PyG Temporal Implementation

```python
def stgnn_example():
    """
    Minimal spatial-temporal GNN using PyTorch Geometric Temporal.
    pip install torch-geometric-temporal
    """
    example_code = """
from torch_geometric_temporal.nn.recurrent import A3TGCN

# A3TGCN: Attention Temporal Graph Convolutional Network
model = A3TGCN(
    in_channels=2,       # features per node per time step
    out_channels=32,     # hidden size
    periods=12,          # input sequence length
)

# Input: (batch, nodes, features, time_steps)
X  = torch.randn(4, 207, 2, 12)
edge_index = torch.tensor(...)   # (2, num_edges) COO format
edge_attr  = torch.ones(num_edges, 1)   # edge weights

out = model(X, edge_index, edge_attr)  # (batch, nodes, out_channels)
forecast = nn.Linear(32, 12)(out)      # (batch, nodes, 12) → 12-step forecast
"""
    print(example_code)


def train_gnn_epoch(model, loader, optimizer, A_tensor, device):
    """Standard GNN training loop for spatial-temporal data."""
    import torch
    model.train()
    total_loss = 0.0
    for batch in loader:
        X, y    = batch
        X, y    = X.to(device), y.to(device)
        y_hat   = model(X, A_tensor)
        loss    = torch.nn.functional.mse_loss(y_hat, y)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    import torch
    # Quick DCRNN sanity check on synthetic data
    N   = 10   # 10 nodes
    T   = 12   # 12 input steps
    H   = 6    # 6-step forecast
    B   = 4    # batch size

    # Synthetic correlation-based adjacency
    np.random.seed(42)
    A = np.random.rand(N, N) * 0.5
    np.fill_diagonal(A, 0); A = (A + A.T) / 2
    row_sums = A.sum(axis=1, keepdims=True) + 1e-8
    A_norm   = A / row_sums
    A_tensor = torch.tensor(A_norm, dtype=torch.float32)

    X_dummy  = torch.randn(B, T, N, 1)
    model    = DCRNN(n_nodes=N, in_ch=1, hidden=32, n_layers=2, horizon=H, K=2)
    y_hat    = model(X_dummy, A_tensor)
    print(f"DCRNN output shape: {y_hat.shape}")  # (4, 6, 10, 1)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

*← [03 — DTW Advanced](./03_dynamic_time_warping_advanced.md) | [Module README](./README.md) | Next: [05 — Diffusion Models](./05_diffusion_models_for_ts.md) →*
