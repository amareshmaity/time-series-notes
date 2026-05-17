# 05 — Time Series Clustering Methods

> **Module**: 10 Classification & Clustering | **File**: 5 of 5
>
> Clustering groups time series by similarity without labels. The right distance measure and algorithm can reveal hidden structure in sensor arrays, user behavior cohorts, or financial instruments. This note covers k-Shape, DTW-based k-Means, hierarchical clustering, HDBSCAN, and evaluation methodology.

---

## Table of Contents

1. [TS Clustering Problem Formulation](#1-ts-clustering-problem-formulation)
2. [Distance Measures for Clustering](#2-distance-measures-for-clustering)
3. [k-Shape — Shape-Based Clustering](#3-k-shape--shape-based-clustering)
4. [k-Means with DTW](#4-k-means-with-dtw)
5. [Hierarchical Clustering](#5-hierarchical-clustering)
6. [HDBSCAN for Time Series](#6-hdbscan-for-time-series)
7. [Clustering Evaluation](#7-clustering-evaluation)
8. [Production Pipeline](#8-production-pipeline)

---

## 1. TS Clustering Problem Formulation

### 1.1 Definition

```
Given:
  D = {X₁, X₂, ..., Xₙ}  — n unlabeled time series

Find:
  Partition {C₁, C₂, ..., Cₖ} such that:
    - Series in the same cluster are similar
    - Series in different clusters are dissimilar

Key challenge: what does "similar" mean for time series?
  - Same amplitude? → Euclidean distance
  - Same shape regardless of scaling? → Pearson-corr distance
  - Same pattern regardless of speed? → DTW
  - Same pattern regardless of phase? → Shape-based distance (SBD)
```

### 1.2 Clustering Approaches

| Approach                       | Distance Measure       | Centroid           | Best For                    |
|-------------------------------|------------------------|--------------------|-----------------------------|
| k-Means + Euclidean           | L2                     | Sample mean        | Very fast, phase-aligned     |
| k-Means + DTW                 | DTW                    | DBA average        | Phase-shifted patterns       |
| k-Shape                       | SBD (cross-correlation)| Shape centroid     | Shape similarity             |
| Hierarchical + DTW            | DTW                    | No centroid        | Dendrogram exploration       |
| HDBSCAN + DTW                 | DTW                    | No centroid        | Arbitrary shape clusters     |
| Autoencoder + k-Means         | Learned                | Embedding centroid | Complex patterns             |

### 1.3 Common Applications

```
IoT/Sensor data:       Group sensors with similar failure patterns
Finance:               Cluster stocks with similar price movements
Healthcare:            Group patients by vital sign trajectories
Energy:                Cluster buildings by consumption profiles
E-commerce:            Segment customers by purchase frequency patterns
```

---

## 2. Distance Measures for Clustering

### 2.1 Shape-Based Distance (SBD)

**SBD** (Paparrizos & Gravano, 2015) is the distance measure used by k-Shape. It is based on normalized cross-correlation:

```
SBD(X, Y) = 1 - max_s [CC_w(X, Y) / √(||X||² · ||Y||²)]

Where:
  CC_w(X, Y) is the phase-invariant cross-correlation
    (uses FFT for efficient computation)

  s is the optimal phase shift that maximizes correlation

SBD = 0: perfectly shape-similar (up to shift and scaling)
SBD = 1: completely dissimilar
SBD = 2: opposite shapes

Advantages:
  ✅ Invariant to shift (finds best alignment)
  ✅ Scale-invariant (normalized)
  ✅ Fast: O(T log T) via FFT
  ✅ Metric properties
```

```python
import numpy as np

def sbd(x: np.ndarray, y: np.ndarray) -> float:
    """
    Shape-Based Distance (SBD) using normalized cross-correlation.
    Used internally by k-Shape.

    Parameters
    ----------
    x, y : 1D arrays (equal length T)

    Returns
    -------
    sbd_distance in [0, 2]
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    T = len(x)

    # Normalize
    x_norm = (x - x.mean()) / (x.std() + 1e-12)
    y_norm = (y - y.mean()) / (y.std() + 1e-12)

    # Cross-correlation via FFT
    xf    = np.fft.rfft(x_norm, n=2*T)
    yf    = np.fft.rfft(y_norm, n=2*T)
    cc    = np.fft.irfft(xf * np.conj(yf))
    cc    = np.concatenate([cc[-(T-1):], cc[:T]])   # shift for max

    # Normalized cross-correlation
    ncc_max = cc.max() / (np.linalg.norm(x_norm) * np.linalg.norm(y_norm) + 1e-12)

    return float(1 - ncc_max)
```

### 2.2 Soft-DTW

**Soft-DTW** (Cuturi & Blondel, 2017) is a differentiable relaxation of DTW that enables gradient-based centroid computation:

```
Soft-DTW_γ(X, Y) = min_A{γ} ⟨A, Δ(X, Y)⟩

Where:
  γ > 0: temperature (γ→0 = hard DTW, γ→∞ = sum of all alignments)
  Δ(X, Y): cost matrix (element-wise squared distances)

Differentiable → can optimize centroids via gradient descent
Widely used in DBA (DTW Barycenter Averaging) and tslearn
```

---

## 3. k-Shape — Shape-Based Clustering

### 3.1 Algorithm

```
k-Shape (Paparrizos & Gravano, 2015):
  Initialize: random cluster assignments

  Repeat until convergence:
    E-step (assignment):
      For each series Xᵢ, assign to cluster Cₖ with minimum SBD:
        k* = argmin_k SBD(Xᵢ, μₖ)

    M-step (update centroid):
      For each cluster Cₖ:
        μₖ = argmax_μ Σ_{Xᵢ ∈ Cₖ} (CC_w(Xᵢ, μ) / (||Xᵢ|| · ||μ||))
        = eigenvector of M̃ (SBD centroid problem, solved via eigendecomposition)

Convergence: guaranteed (objective decreases monotonically)
Complexity:  O(k·N·T·log(T)) per iteration
```

### 3.2 Implementation with tslearn

```python
import numpy as np
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def kshape_cluster(
    X: np.ndarray,
    k: int,
    n_init: int = 10,
    max_iter: int = 100,
    random_state: int = 42,
) -> tuple:
    """
    k-Shape clustering for time series.

    Parameters
    ----------
    X           : (n_samples, T) array of time series
    k           : number of clusters
    n_init      : number of random restarts (pick best)
    max_iter    : maximum iterations per run
    random_state: reproducibility

    Returns
    -------
    labels      : (n_samples,) cluster assignments
    centroids   : (k, T) cluster centroids (shape representatives)
    inertia     : float — within-cluster SBD sum
    """
    # tslearn expects (n_samples, T, 1)
    X_3d = X[:, :, np.newaxis] if X.ndim == 2 else X

    # Z-normalize (required for k-Shape)
    scaler = TimeSeriesScalerMeanVariance()
    X_norm = scaler.fit_transform(X_3d)

    model = KShape(
        n_clusters=k,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=False,
    )
    labels    = model.fit_predict(X_norm)
    centroids = model.cluster_centers_[:, :, 0]   # (k, T)

    return labels, centroids, model.inertia_


def kshape_elbow(X: np.ndarray, k_range: range = range(2, 11)) -> dict:
    """
    Run k-Shape for multiple k values to find elbow point.

    Returns
    -------
    dict with k_range, inertias, silhouette_scores
    """
    from sklearn.metrics import silhouette_score

    inertias   = []
    silhouettes = []

    for k in k_range:
        labels, _, inertia = kshape_cluster(X, k)
        inertias.append(inertia)

        # Silhouette requires a distance matrix
        try:
            from tslearn.metrics import cdist_soft_dtw_normalized as cdist_fn
            X_3d = X[:, :, np.newaxis] if X.ndim == 2 else X
            D    = cdist_fn(X_3d, X_3d)
            sil  = silhouette_score(D, labels, metric="precomputed")
        except Exception:
            sil = np.nan
        silhouettes.append(sil)
        print(f"  k={k}: inertia={inertia:.4f}, silhouette={sil:.4f}")

    return {"k_range": list(k_range), "inertia": inertias, "silhouette": silhouettes}
```

---

## 4. k-Means with DTW

### 4.1 DTW Barycenter Averaging (DBA)

Standard k-Means centroid = sample mean (coordinate-wise). For DTW, the centroid must be computed with **DBA** (Petitjean et al., 2011):

```
DBA algorithm:
  1. Initialize centroid c (e.g., random member)
  2. Repeat:
       a. DTW-align each series xᵢ to current c → optimal warp path Wᵢ
       b. For each position j in c:
            new_c[j] = mean of all xᵢ points that are aligned to c[j] via Wᵢ
  3. Until centroid doesn't change

DBA minimizes: Σᵢ DTW(xᵢ, c)²
DBA is to DTW what sample mean is to Euclidean distance.
```

### 4.2 Implementation with tslearn

```python
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def dtw_kmeans(
    X: np.ndarray,
    k: int,
    metric: str = "dtw",
    window_pct: float = 0.1,
    n_init: int = 3,
    max_iter: int = 50,
    random_state: int = 42,
) -> tuple:
    """
    k-Means with DTW or soft-DTW distance.

    Parameters
    ----------
    X           : (n_samples, T) array
    k           : number of clusters
    metric      : 'dtw' (DBA centroids) or 'softdtw' (smooth centroids)
    window_pct  : Sakoe-Chiba band (for 'dtw' metric)
    n_init      : number of random restarts
    max_iter    : maximum iterations

    Returns
    -------
    labels : (n_samples,) cluster assignments
    model  : fitted TimeSeriesKMeans
    """
    X_3d = X[:, :, np.newaxis] if X.ndim == 2 else X
    scaler = TimeSeriesScalerMeanVariance()
    X_norm = scaler.fit_transform(X_3d)

    T      = X.shape[1]
    window = int(T * window_pct)

    metric_params = {}
    if metric == "dtw":
        metric_params = {"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": window}

    model  = TimeSeriesKMeans(
        n_clusters=k,
        metric=metric,
        metric_params=metric_params,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=False,
        n_jobs=-1,
    )
    labels = model.fit_predict(X_norm)
    return labels, model
```

---

## 5. Hierarchical Clustering

### 5.1 Agglomerative Clustering with DTW

```
Agglomerative clustering (bottom-up):
  1. Start: each series = its own cluster
  2. Merge the two most similar clusters at each step
  3. Continue until k clusters remain

Linkage criteria (how to measure cluster-cluster distance):
  - Single:   d(A, B) = min_{a∈A, b∈B} d(a, b)  (chaining problem)
  - Complete: d(A, B) = max_{a∈A, b∈B} d(a, b)  (compact clusters)
  - Average:  d(A, B) = mean_{a∈A, b∈B} d(a, b)  (balanced, robust)
  - Ward:     minimizes within-cluster variance  (requires Euclidean)

For DTW: use Average or Complete linkage
  (Ward requires Euclidean geometry)
```

### 5.2 Implementation

```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

def hierarchical_dtw_clustering(
    X: np.ndarray,
    k: int = 4,
    window_pct: float = 0.1,
    linkage_method: str = "average",
) -> tuple:
    """
    Hierarchical clustering with DTW distance matrix.

    Parameters
    ----------
    X               : (n_samples, T) array (z-normalized)
    k               : number of clusters to cut
    window_pct      : Sakoe-Chiba band width
    linkage_method  : 'single', 'complete', 'average', 'ward'

    Returns
    -------
    labels   : (n_samples,) cluster assignments
    Z        : linkage matrix (for dendrogram plotting)
    dist_mat : pairwise DTW distance matrix
    """
    # Import our DTW implementation from note 02
    from scipy.spatial.distance import pdist

    n = len(X)
    T = X.shape[1]
    window = int(T * window_pct)

    # Pure DTW via our implementation (or use tslearn for speed)
    def dtw_dist(u, v):
        from numpy import inf, sqrt
        D = np.full((len(u)+1, len(v)+1), inf)
        D[0, 0] = 0.0
        for i in range(1, len(u)+1):
            jlo = max(1, i - window); jhi = min(len(v), i + window)
            for j in range(jlo, jhi+1):
                D[i,j] = (u[i-1]-v[j-1])**2 + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
        return float(sqrt(D[len(u), len(v)]))

    print(f"Computing {n*(n-1)//2} pairwise DTW distances...")
    dist_vec = pdist(X, metric=dtw_dist)
    dist_mat = squareform(dist_vec)

    Z      = linkage(dist_vec, method=linkage_method)
    labels = fcluster(Z, k, criterion="maxclust") - 1   # 0-indexed

    return labels, Z, dist_mat


def plot_dendrogram(Z, labels=None, k=None, figsize=(12, 5)):
    """Plot hierarchical clustering dendrogram with optional cut line."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    dend    = dendrogram(Z, ax=ax, leaf_font_size=8, leaf_rotation=90,
                          labels=labels, color_threshold=0 if k is None else None)

    if k is not None:
        # Draw horizontal cut line at height corresponding to k clusters
        heights = sorted(Z[:, 2])
        cut_h   = heights[-(k-1)] if k > 1 else heights[-1]
        ax.axhline(cut_h, color="red", linestyle="--", linewidth=1.5,
                   label=f"Cut for k={k}")
        ax.legend()

    ax.set_title("Hierarchical Clustering Dendrogram", fontsize=12)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("DTW distance")
    plt.tight_layout()
    plt.savefig("dendrogram.png", dpi=150, bbox_inches="tight")
    plt.show()
```

---

## 6. HDBSCAN for Time Series

### 6.1 Why HDBSCAN?

```
k-Means / k-Shape weaknesses:
  - Must specify k in advance
  - Assumes convex, roughly equal-sized clusters
  - Sensitive to outliers

HDBSCAN (Campello et al., 2013):
  ✅ Automatically determines number of clusters
  ✅ Handles arbitrary cluster shapes and sizes
  ✅ Explicitly identifies outliers (label = -1)
  ✅ More stable than DBSCAN (no ε parameter)

Works with DTW pre-computed distance matrix.
```

```python
import numpy as np
import hdbscan

def hdbscan_dtw_clustering(
    X: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    window_pct: float = 0.1,
) -> tuple:
    """
    HDBSCAN clustering on DTW distance matrix.

    Parameters
    ----------
    X                 : (n_samples, T) array (z-normalized)
    min_cluster_size  : minimum size of a valid cluster
    min_samples       : HDBSCAN core distance parameter
    window_pct        : DTW Sakoe-Chiba band width

    Returns
    -------
    labels            : (n_samples,) cluster assignments (-1 = noise)
    probabilities     : (n_samples,) soft cluster membership
    clusterer         : fitted HDBSCAN object
    """
    try:
        from tslearn.metrics import cdist_dtw
        from tslearn.preprocessing import TimeSeriesScalerMeanVariance

        X_3d   = X[:, :, np.newaxis]
        scaler = TimeSeriesScalerMeanVariance()
        X_norm = scaler.fit_transform(X_3d)
        T      = X.shape[1]
        window = int(T * window_pct)

        print("Computing DTW distance matrix (this may take a moment)...")
        D = cdist_dtw(X_norm, sakoe_chiba_radius=window)

    except ImportError:
        print("tslearn not available — using Euclidean distance")
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(X)

    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="precomputed",
            cluster_selection_method="eom",   # "eom" (default) or "leaf"
        )
        labels        = clusterer.fit_predict(D)
        probabilities = clusterer.probabilities_

    except ImportError:
        print("Install hdbscan: pip install hdbscan")
        return None, None, None

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"Found {n_clusters} clusters, {n_noise} noise points")

    return labels, probabilities, clusterer
```

---

## 7. Clustering Evaluation

### 7.1 Internal Validation Metrics

```python
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    dist_matrix: np.ndarray = None,
) -> dict:
    """
    Comprehensive clustering evaluation.

    Parameters
    ----------
    X           : (n_samples, T) feature matrix (or flattened series)
    labels      : (n_samples,) cluster assignments
    dist_matrix : (n_samples, n_samples) precomputed distances (for DTW-based)

    Returns
    -------
    dict with silhouette, Davies-Bouldin, Calinski-Harabasz scores
    """
    # Remove noise points (label = -1) for silhouette computation
    valid = labels != -1
    if valid.sum() < 2 or len(set(labels[valid])) < 2:
        return {"error": "Too few valid clusters for evaluation"}

    X_valid  = X[valid]
    lb_valid = labels[valid]

    results = {}

    # Silhouette (higher = better, range [-1, 1])
    if dist_matrix is not None:
        D_valid = dist_matrix[np.ix_(valid, valid)]
        results["silhouette"] = float(silhouette_score(D_valid, lb_valid, metric="precomputed"))
    else:
        results["silhouette"] = float(silhouette_score(X_valid, lb_valid))

    # Davies-Bouldin (lower = better, range [0, ∞))
    results["davies_bouldin"] = float(davies_bouldin_score(X_valid, lb_valid))

    # Calinski-Harabász (higher = better)
    results["calinski_harabasz"] = float(calinski_harabasz_score(X_valid, lb_valid))

    results["n_clusters"] = int(len(set(lb_valid)))
    results["n_noise"]    = int((labels == -1).sum())
    results["n_total"]    = int(len(labels))

    return results


def silhouette_per_cluster(X, labels, dist_matrix=None):
    """Per-cluster silhouette scores to identify weak/strong clusters."""
    from sklearn.metrics import silhouette_samples
    valid = labels != -1
    X_v   = X[valid]
    lb_v  = labels[valid]

    if dist_matrix is not None:
        D_v   = dist_matrix[np.ix_(valid, valid)]
        s_vals = silhouette_samples(D_v, lb_v, metric="precomputed")
    else:
        s_vals = silhouette_samples(X_v, lb_v)

    results = {}
    for cluster in sorted(set(lb_v)):
        mask = lb_v == cluster
        results[cluster] = {
            "mean":   round(float(s_vals[mask].mean()), 4),
            "min":    round(float(s_vals[mask].min()),  4),
            "size":   int(mask.sum()),
        }
    return results
```

### 7.2 External Validation (if labels available)

```python
def external_clustering_eval(y_true, y_pred):
    """
    Evaluate clustering against ground truth labels.
    Only meaningful when ground truth is available (e.g., UCR benchmark).
    """
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score,
        adjusted_mutual_info_score, homogeneity_completeness_v_measure
    )
    ari  = adjusted_rand_score(y_true, y_pred)
    nmi  = normalized_mutual_info_score(y_true, y_pred)
    ami  = adjusted_mutual_info_score(y_true, y_pred)
    h, c, v = homogeneity_completeness_v_measure(y_true, y_pred)
    return {
        "ARI":         round(ari, 4),    # Adjusted Rand Index (0 = random, 1 = perfect)
        "NMI":         round(nmi, 4),    # Normalized Mutual Info
        "AMI":         round(ami, 4),    # Adjusted Mutual Info
        "Homogeneity": round(h, 4),      # All samples in cluster from same class
        "Completeness":round(c, 4),      # All same-class samples in same cluster
        "V-measure":   round(v, 4),      # Harmonic mean of H and C
    }
```

---

## 8. Production Pipeline

```python
import numpy as np
import pandas as pd

class TSClusteringPipeline:
    """
    Production time series clustering pipeline.
    Supports k-Shape (default), DTW k-Means, and hierarchical clustering.
    Includes automatic k selection via silhouette elbow.
    """

    def __init__(
        self,
        method: str = "kshape",
        k: int = None,
        k_range: range = range(2, 11),
        window_pct: float = 0.1,
        n_init: int = 5,
        random_state: int = 42,
    ):
        self.method       = method
        self.k            = k
        self.k_range      = k_range
        self.window_pct   = window_pct
        self.n_init       = n_init
        self.random_state = random_state
        self._fitted      = False

    def _normalize(self, X):
        mu    = X.mean(axis=1, keepdims=True)
        sigma = X.std(axis=1, keepdims=True) + 1e-8
        return (X - mu) / sigma

    def auto_select_k(self, X: np.ndarray) -> int:
        """Select k using silhouette elbow on the given k_range."""
        from sklearn.metrics import silhouette_score
        X_norm = self._normalize(X)
        best_k, best_sil = self.k_range[0], -1

        for k in self.k_range:
            try:
                if self.method == "kshape":
                    labels, _, _ = kshape_cluster(X_norm, k, n_init=3,
                                                   random_state=self.random_state)
                else:
                    labels, _ = dtw_kmeans(X_norm, k, n_init=3,
                                            random_state=self.random_state)

                if len(set(labels)) < 2:
                    continue
                sil = silhouette_score(X_norm, labels)
                print(f"  k={k}: silhouette={sil:.4f}")
                if sil > best_sil:
                    best_sil, best_k = sil, k
            except Exception as e:
                print(f"  k={k}: failed ({e})")

        print(f"\nAuto-selected k={best_k} (silhouette={best_sil:.4f})")
        return best_k

    def fit(self, X: np.ndarray) -> "TSClusteringPipeline":
        """Fit clustering model on X."""
        X_norm = self._normalize(X)

        if self.k is None:
            self.k = self.auto_select_k(X)

        if self.method == "kshape":
            self.labels_, self.centroids_, self.inertia_ = \
                kshape_cluster(X_norm, self.k, n_init=self.n_init,
                               random_state=self.random_state)
        elif self.method == "dtw_kmeans":
            self.labels_, self._model = \
                dtw_kmeans(X_norm, self.k, n_init=self.n_init,
                           random_state=self.random_state)
            self.centroids_ = self._model.cluster_centers_[:, :, 0]
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._X_norm = X_norm
        self._fitted  = True
        return self

    def get_cluster_summary(self) -> pd.DataFrame:
        """Summary of each cluster: size, centroid stats."""
        assert self._fitted
        rows = []
        for c in sorted(set(self.labels_)):
            mask     = self.labels_ == c
            centroid = self.centroids_[c]
            rows.append({
                "cluster":        c,
                "size":           int(mask.sum()),
                "pct":            round(100 * mask.mean(), 1),
                "centroid_mean":  round(centroid.mean(), 4),
                "centroid_std":   round(centroid.std(), 4),
                "centroid_range": round(centroid.max() - centroid.min(), 4),
            })
        return pd.DataFrame(rows)
```

---

*← [04 — Deep Learning](./04_deep_learning_classification.md) | [Module README](./README.md)*
