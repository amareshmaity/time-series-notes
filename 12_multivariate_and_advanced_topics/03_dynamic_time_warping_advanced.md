# 03 — Dynamic Time Warping: Advanced Topics

> **Module**: 12 Multivariate & Advanced Topics | **File**: 3 of 6
>
> DTW extends beyond simple distance computation to multivariate alignment, centroid computation, and differentiable relaxations used in modern deep learning. This note covers multivariate DTW, DTW Barycenter Averaging (DBA), Soft-DTW, and their applications in clustering and gradient-based optimization.

---

## Table of Contents

1. [Multivariate DTW](#1-multivariate-dtw)
2. [DTW Barycenter Averaging (DBA)](#2-dtw-barycenter-averaging-dba)
3. [Soft-DTW — Differentiable Relaxation](#3-soft-dtw--differentiable-relaxation)
4. [Global Alignment Kernel (GAK)](#4-global-alignment-kernel-gak)
5. [DTW for Hierarchical Series Alignment](#5-dtw-for-hierarchical-series-alignment)
6. [Implementation](#6-implementation)

---

## 1. Multivariate DTW

### 1.1 Extension to D Dimensions

```
Univariate DTW:
  d(xᵢ, yⱼ) = (xᵢ - yⱼ)²  (scalar distance)

Multivariate DTW:
  xₜ ∈ ℝᴰ,  yₜ ∈ ℝᴰ
  Local distance: d(xᵢ, yⱼ) = ||xᵢ - yⱼ||²  (Euclidean in D dims)

Two strategies:
  1. Dependent (D-DTW):
     Single warping path W applied simultaneously to all D dimensions.
     d(xᵢ, yⱼ) = Σₖ (xᵢₖ - yⱼₖ)²  (sum across channels)
     
     Use when: channels are correlated and should be aligned together.
     Example: IMU sensor — all 3 axes belong to same motion gesture.

  2. Independent (I-DTW):
     Separate warping paths W₁, ..., Wᴰ for each dimension.
     DTW_total = Σₖ DTW(Xₖ, Yₖ)  (sum per-channel DTW distances)
     
     Use when: channels are independent or have different patterns.
     Example: Multivariate biosignals with different frequencies.

Empirically: D-DTW generally outperforms I-DTW for classification.
```

### 1.2 Implementation

```python
import numpy as np

def multivariate_dtw(
    X: np.ndarray,
    Y: np.ndarray,
    window: int = None,
    dependent: bool = True,
) -> float:
    """
    Multivariate DTW distance.

    Parameters
    ----------
    X, Y      : (T, D) arrays — sequences of D-dimensional vectors
    window    : Sakoe-Chiba band width
    dependent : True = D-DTW (single path), False = I-DTW (per-channel)

    Returns
    -------
    dtw_distance : float
    """
    if X.ndim == 1: X = X[:, np.newaxis]
    if Y.ndim == 1: Y = Y[:, np.newaxis]

    Tx, D = X.shape
    Ty    = Y.shape[0]

    if window is None:
        window = max(Tx, Ty)

    if not dependent:
        # I-DTW: sum of per-channel DTW
        total = 0.0
        for d in range(D):
            total += _dtw_1d(X[:, d], Y[:, d], window)
        return float(total)

    # D-DTW: single warp path, multivariate local distance
    D_mat = np.full((Tx + 1, Ty + 1), np.inf)
    D_mat[0, 0] = 0.0

    for i in range(1, Tx + 1):
        for j in range(max(1, i - window), min(Ty, i + window) + 1):
            cost = float(np.sum((X[i-1] - Y[j-1])**2))
            D_mat[i, j] = cost + min(D_mat[i-1, j], D_mat[i, j-1], D_mat[i-1, j-1])

    return float(np.sqrt(D_mat[Tx, Ty]))


def _dtw_1d(x: np.ndarray, y: np.ndarray, window: int) -> float:
    """Scalar DTW — returns squared distance (not sqrt)."""
    Tx, Ty = len(x), len(y)
    D = np.full((Tx + 1, Ty + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, Tx + 1):
        for j in range(max(1, i - window), min(Ty, i + window) + 1):
            D[i, j] = (x[i-1] - y[j-1])**2 + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(D[Tx, Ty])


def multivariate_dtw_matrix(
    X: np.ndarray,
    window_pct: float = 0.1,
    dependent: bool = True,
) -> np.ndarray:
    """
    Compute pairwise D-DTW distance matrix for (N, T, D) array.

    Returns
    -------
    D_mat : (N, N) symmetric distance matrix
    """
    N = len(X)
    T = X.shape[1]
    w = int(T * window_pct)
    D = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            d = multivariate_dtw(X[i], X[j], window=w, dependent=dependent)
            D[i, j] = D[j, i] = d

    return D
```

---

## 2. DTW Barycenter Averaging (DBA)

### 2.1 Algorithm

```
DBA (Petitjean et al., 2011) computes the Fréchet mean under DTW:
  c* = argmin_c Σᵢ DTW(c, xᵢ)²

Algorithm (iterative):
  Initialize c = random member of the set or mean of all series
  
  Repeat until convergence:
    1. For each series xᵢ, compute optimal warp path W_i
       between c and xᵢ  (i.e., DTW alignment)
    
    2. For each position j in c:
         C[j] = mean of all xᵢ points aligned to c[j] via W_i
         
       (Each centroid position is updated to the mean of its assigned points)
  
  Return c.

DBA is to DTW what sample mean is to Euclidean distance.
Required for k-Means with DTW (DBA centroids replace arithmetic mean).
```

### 2.2 DBA Implementation

```python
def dtw_warp_path(x: np.ndarray, y: np.ndarray) -> list:
    """Return optimal DTW warp path as list of (i,j) index pairs."""
    Tx, Ty = len(x), len(y)
    D = np.full((Tx+1, Ty+1), np.inf); D[0,0] = 0.0
    for i in range(1, Tx+1):
        for j in range(1, Ty+1):
            D[i,j] = (x[i-1]-y[j-1])**2 + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    path = [(Tx, Ty)]
    i, j = Tx, Ty
    while i > 1 or j > 1:
        cands = {(i-1,j-1): D[i-1,j-1], (i-1,j): D[i-1,j], (i,j-1): D[i,j-1]}
        i, j  = min(cands, key=cands.get)
        path.append((i, j))
    path.reverse()
    return [(i-1, j-1) for i, j in path]


def dba(
    series_list: list,
    n_iter: int = 20,
    init: str = "random_member",
) -> np.ndarray:
    """
    DTW Barycenter Averaging — compute the mean series under DTW.

    Parameters
    ----------
    series_list : list of 1D numpy arrays (equal length T)
    n_iter      : number of DBA iterations
    init        : initialization ('random_member' or 'mean')

    Returns
    -------
    centroid : 1D array of length T — the DBA average
    """
    T   = len(series_list[0])
    N   = len(series_list)

    # Initialize centroid
    if init == "mean":
        centroid = np.mean(series_list, axis=0)
    else:
        centroid = series_list[np.random.randint(N)].copy()

    for iteration in range(n_iter):
        # Accumulate aligned values for each centroid position
        assign  = [[] for _ in range(T)]

        for s in series_list:
            path = dtw_warp_path(centroid, s)
            for (ci, si) in path:
                assign[ci].append(s[si])

        new_centroid = np.array([np.mean(a) if a else centroid[j]
                                  for j, a in enumerate(assign)])

        # Check convergence
        change = float(np.abs(new_centroid - centroid).max())
        centroid = new_centroid

        if change < 1e-6:
            print(f"DBA converged at iteration {iteration+1}")
            break

    return centroid


def dba_multivariate(
    series_list: list,
    n_iter: int = 20,
) -> np.ndarray:
    """
    DBA for multivariate series: (N, T, D) → centroid of shape (T, D).
    Applies DBA per channel independently.
    """
    D         = series_list[0].shape[1]
    centroids = []
    for d in range(D):
        channel_series = [s[:, d] for s in series_list]
        centroids.append(dba(channel_series, n_iter=n_iter))
    return np.column_stack(centroids)
```

---

## 3. Soft-DTW — Differentiable Relaxation

### 3.1 Motivation

```
Problem with hard DTW:
  DTW(X, Y) is NOT differentiable w.r.t. X or Y.
  → Cannot use DTW as a loss function in gradient-based learning.

Soft-DTW (Cuturi & Blondel, 2017):
  Replaces hard min with soft minimum:
    soft-min_γ(a₁, ..., aₙ) = -γ log Σᵢ exp(-aᵢ/γ)

  Soft-DTW_γ(X, Y) = ⟨A*, Δ(X,Y)⟩
    where A* = argmin_{A ∈ A(T_x,T_y)} soft-min of alignments

  γ → 0:   hard DTW
  γ → ∞:   sum of all pairwise distances (no alignment)
  γ = 1.0: good default for normalized series

Applications:
  ✅ DTW as training loss for neural networks
  ✅ Differentiable k-Means with DTW centroids
  ✅ Gradient-based DBA (faster convergence than iterative DBA)
```

### 3.2 Implementation

```python
def soft_dtw(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """
    Soft-DTW distance — differentiable relaxation of DTW.

    Parameters
    ----------
    X, Y  : 1D arrays of equal or different lengths
    gamma : smoothing parameter (> 0)

    Returns
    -------
    Soft-DTW value (float) — differentiable w.r.t. X and Y
    """
    def soft_min(a, b, c, g):
        """Numerically stable soft minimum of three values."""
        vals = np.array([-a/g, -b/g, -c/g])
        m    = vals.max()
        return -g * (m + np.log(np.exp(vals - m).sum()))

    Tx, Ty = len(X), len(Y)
    R      = np.full((Tx + 1, Ty + 1), np.inf)
    R[0, 0] = 0.0

    for i in range(1, Tx + 1):
        for j in range(1, Ty + 1):
            cost   = (X[i-1] - Y[j-1])**2
            R[i,j] = cost - soft_min(R[i-1,j], R[i,j-1], R[i-1,j-1], gamma)

    return float(R[Tx, Ty])


def soft_dtw_loss_example():
    """
    Example: using soft-DTW as a training loss with PyTorch.
    pip install tslearn (provides SoftDTW in torch)
    """
    example_code = """
import torch
from tslearn.metrics import SoftDTWLossPyTorch

# Batch: (batch_size, T, D)
X = torch.randn(32, 50, 1, requires_grad=True)  # predictions
Y = torch.randn(32, 50, 1)                       # targets

sdtw_loss = SoftDTWLossPyTorch(gamma=1.0, normalize=True)
loss = sdtw_loss(X, Y).mean()
loss.backward()  # ← works because Soft-DTW is differentiable!
print(f"Loss: {loss.item():.4f}, Grad norm: {X.grad.norm().item():.4f}")
"""
    print(example_code)
```

---

## 4. Global Alignment Kernel (GAK)

### 4.1 Theory

```
Global Alignment Kernel (Cuturi, 2011):
  K(X, Y) = Σ_{A ∈ A(T,T)} exp(-⟨A, Δ(X,Y)⟩ / σ²)

  Summing over ALL valid alignment paths (vs. DTW which takes the min).
  Properly positive-definite → can be used in kernel SVM, kernel PCA.

Properties:
  ✅ Positive-definite kernel (DTW is NOT a valid kernel)
  ✅ Can be plugged into any kernel method (SVM, Gaussian Process)
  ✅ Invariant to local time warping
  ✅ σ controls smoothness (analogous to γ in Soft-DTW)

Use case:
  Kernel SVM for TSC when you want DTW-like similarity but need a PD kernel.
```

```python
def gak(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """
    Global Alignment Kernel between two 1D sequences.
    Uses tslearn's implementation for numerical stability.

    pip install tslearn
    """
    try:
        from tslearn.metrics import cdist_gak
        X = x.reshape(1, -1, 1)
        Y = y.reshape(1, -1, 1)
        K = cdist_gak(X, Y, sigma=sigma)
        return float(K[0, 0])
    except ImportError:
        # Approximate via Soft-DTW
        sdtw = soft_dtw(x, y, gamma=sigma**2)
        return float(np.exp(-sdtw / sigma**2))
```

---

## 5. DTW for Hierarchical Series Alignment

### 5.1 Hierarchical DTW Use Case

```
Hierarchical alignment problem:
  Given: N series grouped into K clusters.
  Goal:  Align all series within each cluster to a common template.
  
  Applications:
    - ECG beat alignment across patients
    - Gesture alignment for training recognition models
    - Multi-site clinical trial data harmonization

Pipeline:
  1. For each cluster, compute DBA centroid c_k
  2. Warp each series Xᵢ to align with c_k via DTW
  3. Use aligned series for downstream analysis
```

```python
def align_series_to_template(
    series_list: list,
    template: np.ndarray,
) -> list:
    """
    Warp all series to align with a common template using DTW.

    Parameters
    ----------
    series_list : list of 1D arrays (may have different lengths)
    template    : 1D reference array

    Returns
    -------
    aligned : list of arrays with same length as template,
              each warped to best match the template
    """
    T       = len(template)
    aligned = []

    for s in series_list:
        path = dtw_warp_path(template, s)

        # Map each template position to its aligned series values
        template_to_s = [[] for _ in range(T)]
        for (ti, si) in path:
            template_to_s[ti].append(s[si])

        aligned_s = np.array([np.mean(vals) if vals else 0.0
                               for vals in template_to_s])
        aligned.append(aligned_s)

    return aligned
```

---

## 6. Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_dba():
    """
    Show DBA averaging vs. simple mean on a set of phase-shifted sinusoids.
    """
    np.random.seed(42)
    T = 80
    t = np.linspace(0, 4*np.pi, T)

    series_list = []
    for _ in range(15):
        phase = np.random.uniform(-np.pi/3, np.pi/3)
        amp   = np.random.uniform(0.8, 1.2)
        noise = np.random.normal(0, 0.1, T)
        series_list.append(amp * np.sin(t + phase) + noise)

    X = np.array(series_list)

    # Simple mean (ignores phase — produces flat, attenuated result)
    simple_mean = X.mean(axis=0)

    # DBA mean (respects temporal patterns despite phase shift)
    dba_mean = dba(series_list, n_iter=20)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for s in series_list:
        axes[0].plot(s, color="#2196F3", alpha=0.3, linewidth=1)
    axes[0].plot(simple_mean, "r-", linewidth=2.5, label="Simple mean (flat!)")
    axes[0].plot(dba_mean, "g-", linewidth=2.5, label="DBA mean (preserves shape)")
    axes[0].set_title("DBA vs. Simple Mean on Phase-Shifted Sinusoids", fontsize=11)
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # Soft-DTW vs. DTW distance comparison
    test_pairs = [(X[0], X[1]), (X[0], X[7]), (X[0], X[14])]
    gammas     = [0.1, 1.0, 5.0, 10.0]
    labels     = [f"Pair {i+1}" for i in range(3)]

    ax2 = axes[1]
    for i, (xa, xb) in enumerate(test_pairs):
        dtw_d  = _dtw_1d(xa, xb, window=T)
        sdtw_d = [soft_dtw(xa, xb, g) for g in gammas]
        ax2.plot(gammas, sdtw_d, "o-", linewidth=1.5, label=f"{labels[i]} (DTW={dtw_d:.1f})")
        ax2.axhline(dtw_d, linestyle="--", alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_xlabel("γ (Soft-DTW smoothing)"); ax2.set_ylabel("Distance")
    ax2.set_title("Soft-DTW vs. γ (→ Hard DTW as γ → 0)", fontsize=11)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("dtw_advanced_demo.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved: dtw_advanced_demo.png")


if __name__ == "__main__":
    demonstrate_dba()
```

---

*← [02 — Granger Causality](./02_granger_causality.md) | [Module README](./README.md) | Next: [04 — GNN for TS](./04_graph_neural_networks_for_ts.md) →*
