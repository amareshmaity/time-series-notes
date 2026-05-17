# 02 — Distance-Based Methods and DTW

> **Module**: 10 Classification & Clustering | **File**: 2 of 5
>
> The simplest and most robust time series classifiers are nearest-neighbor methods. Their performance depends entirely on the distance function. Dynamic Time Warping (DTW) — which aligns two series by warping the time axis — remains one of the most competitive distance measures 25 years after its introduction.

---

## Table of Contents

1. [Euclidean Distance — Baseline](#1-euclidean-distance--baseline)
2. [Dynamic Time Warping (DTW)](#2-dynamic-time-warping-dtw)
3. [Warping Constraints — Sakoe-Chiba Band](#3-warping-constraints--sakoe-chiba-band)
4. [DTW Lower Bounds — LB_Keogh](#4-dtw-lower-bounds--lb_keogh)
5. [Weighted DTW (WDTW)](#5-weighted-dtw-wdtw)
6. [Other Elastic Distances — LCSS, ERP, EDR](#6-other-elastic-distances--lcss-erp-edr)
7. [kNN-DTW Classifier](#7-knn-dtw-classifier)
8. [FastDTW and Scalable Variants](#8-fastdtw-and-scalable-variants)
9. [Implementation](#9-implementation)

---

## 1. Euclidean Distance — Baseline

### 1.1 Definition

```
ED(X, Y) = √(Σᵢ (xᵢ - yᵢ)²)   for sequences of EQUAL length T

Problems:
  1. Requires equal-length sequences
  2. Point-to-point alignment — xᵢ matched to yᵢ only
     → Two identical shapes but slightly shifted → large distance

Example failure:
  X = [0, 1, 0, 0, 0]   (single spike at position 2)
  Y = [0, 0, 1, 0, 0]   (single spike at position 3)
  ED(X, Y) = √2 ≈ 1.41  ← large, despite identical shape

DTW correctly identifies these as similar (distance ≈ 0 with warping).
```

---

## 2. Dynamic Time Warping (DTW)

### 2.1 Intuition

```
DTW finds the optimal alignment between two sequences by "warping" the time axis.
It matches each point in X to one or more points in Y, finding the alignment
that minimises total distance.

Visual:
  X: ___/\___
  Y: ___/\___  (shifted by 2 steps)

Euclidean: aligns step-by-step → mismatches at peak
DTW:       stretches/compresses to align peaks → correct

Warp path W = {(i₁,j₁), (i₂,j₂), ..., (iₖ,jₖ)}:
  - Monotonic: i and j are non-decreasing
  - Continuity: each step moves at most 1 in each direction
  - Boundary: starts at (1,1), ends at (T,T)
```

### 2.2 Dynamic Programming Formulation

```
DTW(X, Y) = min over all valid warp paths W of: Σ d(xᵢₜ, yⱼₜ)

Computed via:
  D[i, j] = d(xᵢ, yⱼ) + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

Where d(·,·) = local distance (typically |xᵢ - yⱼ|² or |xᵢ - yⱼ|)

DTW(X, Y) = √D[T, T]    (or just D[T, T] for squared-DTW)

Time complexity:  O(T²)
Space complexity: O(T²) or O(T) with striped computation
```

### 2.3 Pure Python Implementation

```python
import numpy as np

def dtw_distance(x: np.ndarray, y: np.ndarray, window: int = None) -> float:
    """
    Compute DTW distance between two 1D sequences.

    Parameters
    ----------
    x, y   : 1D arrays (length Tx, Ty)
    window : Sakoe-Chiba band width (None = no constraint)

    Returns
    -------
    DTW distance (float)
    """
    Tx, Ty = len(x), len(y)

    if window is None:
        window = max(Tx, Ty)   # no constraint

    # Initialize cost matrix
    D = np.full((Tx + 1, Ty + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, Tx + 1):
        # Sakoe-Chiba band
        j_lo = max(1, i - window)
        j_hi = min(Ty, i + window)
        for j in range(j_lo, j_hi + 1):
            cost    = (x[i-1] - y[j-1])**2
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    return float(np.sqrt(D[Tx, Ty]))


def dtw_distance_matrix(X: np.ndarray, Y: np.ndarray = None, window: int = None) -> np.ndarray:
    """
    Compute pairwise DTW distance matrix.

    X: (n, T)  — query sequences
    Y: (m, T)  — reference sequences (None = symmetric X vs X)

    Returns
    -------
    D: (n, m) distance matrix
    """
    if Y is None:
        Y         = X
        symmetric = True
    else:
        symmetric = False

    n, m = len(X), len(Y)
    D    = np.zeros((n, m))

    for i in range(n):
        j_start = i + 1 if symmetric else 0
        for j in range(j_start, m):
            d = dtw_distance(X[i], Y[j], window)
            D[i, j] = d
            if symmetric:
                D[j, i] = d

    return D
```

### 2.4 Warp Path Retrieval

```python
def dtw_warp_path(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Compute DTW distance and return the optimal warp path.

    Returns
    -------
    (distance, path) where path = list of (i, j) index pairs
    """
    Tx, Ty = len(x), len(y)
    D      = np.full((Tx + 1, Ty + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, Tx + 1):
        for j in range(1, Ty + 1):
            cost    = (x[i-1] - y[j-1])**2
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    # Backtrack
    path = [(Tx, Ty)]
    i, j = Tx, Ty
    while i > 1 or j > 1:
        candidates = {
            (i-1, j-1): D[i-1, j-1],
            (i-1, j):   D[i-1, j],
            (i,   j-1): D[i,   j-1],
        }
        i, j = min(candidates, key=candidates.get)
        path.append((i, j))

    path.reverse()
    return float(np.sqrt(D[Tx, Ty])), [(i-1, j-1) for i, j in path]
```

---

## 3. Warping Constraints — Sakoe-Chiba Band

### 3.1 Why Constrain?

```
Unconstrained DTW allows arbitrarily large warpings:
  - A single point in X can match all T points in Y
  - This makes similar but different sequences identical
  - Also slows computation (O(T²) without constraint)

Sakoe-Chiba Band: |i - j| ≤ r
  - Limits warping to within r steps of the diagonal
  - r = 0: reduces to Euclidean distance
  - r = T: unconstrained DTW

Itakura Parallelogram:
  - Alternative constraint: slope of warp path ≤ max_slope
  - Less commonly used

Best practice:
  - Tune r on training data via cross-validation
  - Typical best values: r = 10–20% of T (i.e., window = 0.1 * T)
```

```python
def best_warping_window(X_train: np.ndarray, y_train: np.ndarray,
                        candidates: list = None) -> int:
    """
    Find optimal Sakoe-Chiba band width via leave-one-out CV on training set.

    Parameters
    ----------
    X_train    : (n, T) training series (z-normalized)
    y_train    : (n,) class labels
    candidates : list of window sizes to try (default: 0% to 20% of T)

    Returns
    -------
    best_window : int — optimal warping window
    """
    T = X_train.shape[1]
    if candidates is None:
        candidates = [int(T * r) for r in np.arange(0, 0.21, 0.02)]

    best_window, best_acc = 0, 0.0
    n = len(X_train)

    for w in candidates:
        correct = 0
        for i in range(n):
            # LOO: exclude sample i from reference
            X_ref = np.delete(X_train, i, axis=0)
            y_ref = np.delete(y_train, i)
            dists = np.array([dtw_distance(X_train[i], X_ref[j], window=w)
                              for j in range(len(X_ref))])
            nn_class = y_ref[np.argmin(dists)]
            correct += (nn_class == y_train[i])

        acc = correct / n
        if acc > best_acc:
            best_acc, best_window = acc, w
        print(f"  w={w:3d}: LOO-acc={acc:.4f}")

    print(f"\nBest window: {best_window} (LOO accuracy: {best_acc:.4f})")
    return best_window
```

---

## 4. DTW Lower Bounds — LB_Keogh

### 4.1 Why Lower Bounds?

```
Problem: computing DTW(X, Y) for all Y in a dataset of size N
         costs O(N·T²) — prohibitive for large N or T.

Lower Bound trick:
  If LB(X, Y) > current_best_distance:
    → Skip exact DTW computation (pruning)

LB_Keogh: fastest lower bound for DTW
  1. Create bounding envelope from query X with warping window r:
     U_i = max(x_{i-r}, ..., x_{i+r})  (upper envelope)
     L_i = min(x_{i-r}, ..., x_{i+r})  (lower envelope)

  2. LB_Keogh(X, Y) = √(Σᵢ max(yᵢ - Uᵢ, Lᵢ - yᵢ, 0)²)

Property: LB_Keogh(X, Y) ≤ DTW(X, Y) always
           Fast to compute: O(T)
           Typical pruning: 50–99% of DTW computations skipped
```

```python
def lb_keogh(x: np.ndarray, y: np.ndarray, window: int) -> float:
    """
    LB_Keogh lower bound for DTW distance.

    Parameters
    ----------
    x      : query sequence (1D)
    y      : candidate sequence (1D)
    window : Sakoe-Chiba band width

    Returns
    -------
    lower bound ≤ DTW(x, y)
    """
    T  = len(x)
    lb = 0.0
    for i in range(T):
        lo  = max(0, i - window)
        hi  = min(T - 1, i + window)
        env_max = x[lo:hi+1].max()
        env_min = x[lo:hi+1].min()
        if y[i] > env_max:
            lb += (y[i] - env_max)**2
        elif y[i] < env_min:
            lb += (env_min - y[i])**2
    return float(np.sqrt(lb))
```

---

## 5. Weighted DTW (WDTW)

### 5.1 Motivation

Standard DTW penalizes all warpings equally. WDTW assigns higher penalty to large warpings (diagonal matching is preferred):

```
WDTW(X, Y) = min_path Σ w(|i-j|) · d(xᵢ, yⱼ)

Where w(m) = 1 / (1 + exp(-g · (m - T/2)))  — logistic weight
g = penalty factor (higher → prefer diagonal, less warping)
```

```python
def wdtw_distance(x: np.ndarray, y: np.ndarray, g: float = 0.05) -> float:
    """
    Weighted DTW — penalises off-diagonal warpings.

    g : penalty factor (0.0 = standard DTW, larger → diagonal preferred)
    """
    Tx, Ty = len(x), len(y)
    T_max  = max(Tx, Ty)

    # Penalty weight for warping by m steps
    w = np.array([1 / (1 + np.exp(-g * (m - T_max/2))) for m in range(T_max + 1)])

    D = np.full((Tx + 1, Ty + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, Tx + 1):
        for j in range(1, Ty + 1):
            cost    = w[abs(i - j)] * (x[i-1] - y[j-1])**2
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    return float(np.sqrt(D[Tx, Ty]))
```

---

## 6. Other Elastic Distances — LCSS, ERP, EDR

### 6.1 LCSS — Longest Common Subsequence

```
LCSS is robust to noise and outliers by allowing points to be SKIPPED.
A match between xᵢ and yⱼ only counts if |xᵢ - yⱼ| ≤ ε (threshold).

LCSS(X, Y; ε, δ):
  δ = window constraint (like DTW)
  ε = matching threshold (ignore differences smaller than ε)

  L[i, j] = L[i-1, j-1] + 1          if |xᵢ - yⱼ| ≤ ε and |i-j| ≤ δ
             max(L[i-1, j], L[i, j-1])  otherwise

LCSS distance = 1 - L[Tx, Ty] / min(Tx, Ty)

Use case: noisy IoT data where occasional large values should be ignored.
```

### 6.2 EDR — Edit Distance on Real Sequences

```
EDR counts the minimum number of edit operations needed to transform X into Y.
Operations: insert, delete, substitute (with threshold ε).

EDR(X, Y) = edit_operations / max(Tx, Ty)

Advantage: metric (satisfies triangle inequality, unlike DTW)
Use case: when you need a proper metric for indexing or kernel methods.
```

### 6.3 ERP — Edit distance with Real Penalty

```
ERP uses a fixed reference value g (often 0.0) instead of ε.
Gaps (insertions/deletions) are penalized by |xᵢ - g| rather than ε.

ERP(X, Y; g):
  D[i, j] = min(
    D[i-1, j-1] + |xᵢ - yⱼ|,      # match
    D[i-1, j]   + |xᵢ - g|,         # delete from X
    D[i, j-1]   + |yⱼ - g|,         # delete from Y
  )

ERP is a metric (unlike DTW/LCSS) and reduces to L1 distance when g = xᵢ = yⱼ.
```

---

## 7. kNN-DTW Classifier

### 7.1 1-NN DTW — The Gold Standard Baseline

```
The 1-NN classifier with DTW distance:
  1. Z-normalize all training and test series
  2. For each test series q:
       Find train sample x* = argmin_{x ∈ train} DTW(q, x)
       Assign label y* = label(x*)

Performance:
  - Competitive with state-of-the-art on many UCR datasets
  - Best on small datasets (no hyperparameters to tune)
  - Slow at test time: O(N·T²) per query

The DTW warping window r is the ONLY hyperparameter.
Tune with leave-one-out CV on training set.
```

### 7.2 Implementation with tslearn

```python
import numpy as np
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def knn_dtw_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_neighbors: int = 1,
    window_pct: float = 0.1,
) -> np.ndarray:
    """
    kNN classifier with DTW distance (via tslearn).

    Parameters
    ----------
    X_train     : (n_train, T) or (n_train, T, 1) array
    y_train     : (n_train,) class labels
    X_test      : (n_test, T) or (n_test, T, 1) array
    n_neighbors : k for kNN (1-NN is the standard baseline)
    window_pct  : Sakoe-Chiba band as fraction of T (e.g., 0.1 = 10%)

    Returns
    -------
    y_pred : (n_test,) predicted labels
    """
    # tslearn expects (n_samples, T, n_channels)
    if X_train.ndim == 2:
        X_train = X_train[:, :, np.newaxis]
        X_test  = X_test[:,  :, np.newaxis]

    # Z-normalization (per series)
    scaler  = TimeSeriesScalerMeanVariance()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    T      = X_train.shape[1]
    window = int(T * window_pct)

    clf = KNeighborsTimeSeriesClassifier(
        n_neighbors=n_neighbors,
        metric="dtw",
        metric_params={
            "global_constraint": "sakoe_chiba",
            "sakoe_chiba_radius": window,
        },
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_test)
```

---

## 8. FastDTW and Scalable Variants

### 8.1 FastDTW

FastDTW (Salvador & Chan, 2007) computes an approximate DTW in **O(T)** time using a multilevel coarsening approach:

```python
def fast_dtw_distance(x: np.ndarray, y: np.ndarray, radius: int = 1) -> float:
    """
    FastDTW approximation — O(T·radius) time.

    Parameters
    ----------
    radius : accuracy-speed tradeoff (higher = more accurate, slower)
             radius=1 is usually indistinguishable from exact DTW

    Uses the fastdtw package (pip install fastdtw).
    """
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        dist, _ = fastdtw(x, y, radius=radius, dist=euclidean)
        return float(dist)
    except ImportError:
        print("Install fastdtw: pip install fastdtw")
        return dtw_distance(x, y)   # fallback to exact
```

### 8.2 Batch DTW with tslearn (Optimized)

```python
def batch_dtw_1nn(X_train, y_train, X_test, window_pct=0.1):
    """
    Vectorized 1-NN DTW using tslearn's cdist_dtw (Cython/C++ backend).
    Much faster than pure-Python loops.
    """
    try:
        from tslearn.metrics import cdist_dtw
        from tslearn.preprocessing import TimeSeriesScalerMeanVariance

        scaler  = TimeSeriesScalerMeanVariance()
        Xtr     = scaler.fit_transform(X_train[:, :, np.newaxis])
        Xte     = scaler.transform(X_test[:,  :, np.newaxis])
        T       = Xtr.shape[1]
        window  = int(T * window_pct)

        D        = cdist_dtw(Xte, Xtr, sakoe_chiba_radius=window)
        nn_idx   = D.argmin(axis=1)
        return y_train[nn_idx]
    except ImportError:
        print("Install tslearn: pip install tslearn")
        return None
```

---

## 9. Implementation

### 9.1 DTW Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_dtw_alignment(x: np.ndarray, y: np.ndarray, window: int = None):
    """
    Visualize DTW warp path between two sequences.
    """
    dist, path = dtw_warp_path(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: aligned sequences with path connections
    ax = axes[0]
    ax.plot(x, "b-o", markersize=4, linewidth=1.5, label="X")
    ax.plot(y, "r-o", markersize=4, linewidth=1.5, label="Y")
    for (i, j) in path[::max(1, len(path)//20)]:   # subsample path for clarity
        ax.plot([i, j], [x[i], y[j]], "k-", alpha=0.2, linewidth=0.8)
    ax.set_title(f"DTW Alignment (distance = {dist:.4f})", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: cost matrix with warp path
    Tx, Ty = len(x), len(y)
    D      = np.full((Tx, Ty), np.inf)
    D[0, 0] = (x[0] - y[0])**2
    for i in range(Tx):
        for j in range(Ty):
            c = (x[i] - y[j])**2
            pred = np.inf
            if i > 0 and j > 0: pred = min(D[i-1,j-1], pred)
            if i > 0:            pred = min(D[i-1,j],   pred)
            if j > 0:            pred = min(D[i,j-1],   pred)
            D[i, j] = c + (pred if pred < np.inf else 0.0)

    ax = axes[1]
    ax.imshow(D, origin="lower", aspect="auto", cmap="viridis")
    path_i = [p[0] for p in path]
    path_j = [p[1] for p in path]
    ax.plot(path_j, path_i, "w-", linewidth=2, label="Warp path")
    ax.set_title("DTW Cost Matrix with Warp Path", fontsize=12)
    ax.set_xlabel("Y index"); ax.set_ylabel("X index")
    ax.legend()

    plt.tight_layout()
    plt.savefig("dtw_alignment.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    # Two sinusoids with phase shift
    T = 50
    t = np.linspace(0, 4*np.pi, T)
    x = np.sin(t)
    y = np.sin(t + np.pi/4)   # phase-shifted

    dist_ed  = np.sqrt(((x - y)**2).sum())
    dist_dtw = dtw_distance(x, y)

    print(f"Euclidean distance: {dist_ed:.4f}")
    print(f"DTW distance:       {dist_dtw:.4f}  (smaller → correctly identified as similar)")

    plot_dtw_alignment(x, y)
```

---

*← [01 — TSC Overview](./01_ts_classification_overview.md) | [Module README](./README.md) | Next: [03 — Feature-Based Classification](./03_feature_based_classification.md) →*
