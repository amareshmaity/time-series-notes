"""
code/03_ts_clustering.py
==========================
Module 10 — Classification & Clustering
Practical: k-Shape and hierarchical TS clustering.

Demonstrates:
  - k-Shape clustering with silhouette elbow selection
  - DTW k-Means (via tslearn)
  - Hierarchical clustering with dendrogram
  - Silhouette score evaluation and per-cluster analysis
  - Centroid visualization
  - Comparison with ground truth labels (ARI, NMI)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic TS Dataset — 4 Distinct Shapes
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(0)
T       = 64
N_PER   = 40     # samples per cluster
N_CLS   = 4

def make_cluster_data():
    """Generate 4 shapely clusters with variation."""
    t   = np.linspace(0, 4*np.pi, T)
    X, y = [], []

    for _ in range(N_PER):
        X.append(np.sin(t + np.random.uniform(-0.4, 0.4)) + np.random.normal(0, 0.15, T))
        y.append(0)
    for _ in range(N_PER):
        X.append(np.sin(2*t + np.random.uniform(-0.4, 0.4)) * np.random.uniform(0.8, 1.2) + np.random.normal(0, 0.15, T))
        y.append(1)
    for _ in range(N_PER):
        X.append(np.where(t < 2*np.pi, t/(2*np.pi), 2 - t/(2*np.pi)) * np.random.uniform(0.8, 1.2) + np.random.normal(0, 0.1, T))
        y.append(2)
    for _ in range(N_PER):
        X.append(np.concatenate([np.zeros(T//2), np.ones(T//2)]) * np.random.uniform(0.8, 1.2) + np.random.normal(0, 0.08, T))
        y.append(3)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

X_all, y_true = make_cluster_data()
N = len(X_all)

# Z-normalize per series
def z_norm(X):
    mu = X.mean(axis=1, keepdims=True); s = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / s

X_norm = z_norm(X_all)
print(f"Dataset: {N} series, T={T}, {N_CLS} ground-truth clusters")


# ─────────────────────────────────────────────────────────────────────────────
# 2. k-Shape Clustering (tslearn)
# ─────────────────────────────────────────────────────────────────────────────

def run_kshape(X, k, n_init=5, random_state=42):
    """k-Shape via tslearn."""
    try:
        from tslearn.clustering import KShape
        from tslearn.preprocessing import TimeSeriesScalerMeanVariance

        X_3d   = X[:, :, np.newaxis]
        scaler = TimeSeriesScalerMeanVariance()
        X_s    = scaler.fit_transform(X_3d)

        model  = KShape(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = model.fit_predict(X_s)
        centroids = model.cluster_centers_[:, :, 0]
        return labels, centroids, model.inertia_

    except ImportError:
        print("tslearn not installed — using sklearn KMeans on raw series (Euclidean fallback)")
        from sklearn.cluster import KMeans
        model  = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = model.fit_predict(X)
        centroids = model.cluster_centers_
        return labels, centroids, float(model.inertia_)


# Elbow selection
print("\nk-Shape elbow curve (k=2..7):")
k_range     = range(2, 8)
inertias    = []
silhouettes = []
ks_labels_dict = {}

for k in k_range:
    labels, centroids, inertia = run_kshape(X_norm, k)
    ks_labels_dict[k] = labels
    inertias.append(inertia)
    sil = silhouette_score(X_norm, labels) if len(set(labels)) > 1 else np.nan
    silhouettes.append(sil)
    ari = adjusted_rand_score(y_true, labels)
    print(f"  k={k}: inertia={inertia:.2f}, silhouette={sil:.4f}, ARI={ari:.4f}")

best_k  = k_range.start + np.argmax(silhouettes)
print(f"\nAuto-selected k={best_k} (silhouette={max(silhouettes):.4f})")

labels_kshape, centroids_kshape, _ = run_kshape(X_norm, best_k)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DTW k-Means
# ─────────────────────────────────────────────────────────────────────────────

def run_dtw_kmeans(X, k, n_init=3, random_state=42):
    """DTW k-Means via tslearn."""
    try:
        from tslearn.clustering import TimeSeriesKMeans
        from tslearn.preprocessing import TimeSeriesScalerMeanVariance

        X_3d   = X[:, :, np.newaxis]
        scaler = TimeSeriesScalerMeanVariance()
        X_s    = scaler.fit_transform(X_3d)

        model  = TimeSeriesKMeans(
            n_clusters=k, metric="dtw", n_init=n_init,
            max_iter=20, random_state=random_state, verbose=False, n_jobs=-1,
        )
        labels    = model.fit_predict(X_s)
        centroids = model.cluster_centers_[:, :, 0]
        return labels, centroids

    except ImportError:
        print("tslearn not available — returning k-Shape labels")
        return labels_kshape, centroids_kshape

print(f"\nDTW k-Means (k={best_k})...")
labels_dtw, centroids_dtw = run_dtw_kmeans(X_norm, best_k)
print(f"  ARI={adjusted_rand_score(y_true, labels_dtw):.4f}, "
      f"silhouette={silhouette_score(X_norm, labels_dtw):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hierarchical Clustering
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nHierarchical clustering (average linkage, Euclidean)...")
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

# Use Euclidean for speed (DTW would require precomputed matrix)
dist_vec    = pdist(X_norm, metric="euclidean")
Z           = linkage(dist_vec, method="average")
labels_hier = fcluster(Z, best_k, criterion="maxclust") - 1   # 0-indexed

print(f"  ARI={adjusted_rand_score(y_true, labels_hier):.4f}, "
      f"silhouette={silhouette_score(X_norm, labels_hier):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluation Summary
# ─────────────────────────────────────────────────────────────────────────────

def eval_clustering(labels, name):
    valid = labels != -1
    sil = silhouette_score(X_norm[valid], labels[valid]) if valid.sum() > 1 else np.nan
    db  = davies_bouldin_score(X_norm[valid], labels[valid]) if valid.sum() > 1 else np.nan
    ari = adjusted_rand_score(y_true[valid], labels[valid])
    nmi = normalized_mutual_info_score(y_true[valid], labels[valid])
    return {"Method": name, "k": len(set(labels[valid])),
            "Silhouette↑": round(sil, 4), "Davies-Bouldin↓": round(db, 4),
            "ARI↑": round(ari, 4), "NMI↑": round(nmi, 4),
            "Noise": int((labels == -1).sum())}

eval_df = pd.DataFrame([
    eval_clustering(labels_kshape, f"k-Shape (k={best_k})"),
    eval_clustering(labels_dtw,    f"DTW k-Means (k={best_k})"),
    eval_clustering(labels_hier,   f"Hierarchical (k={best_k})"),
]).set_index("Method")

print("\n" + "="*65)
print("CLUSTERING EVALUATION")
print("="*65)
print(eval_df.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualization
# ─────────────────────────────────────────────────────────────────────────────

colors = plt.cm.tab10(np.linspace(0, 0.5, best_k))
fig, axes = plt.subplots(3, 2, figsize=(15, 14))

# Panel 1: Ground truth
ax = axes[0, 0]
gt_colors = plt.cm.tab10(np.linspace(0, 0.5, N_CLS))
for c in range(N_CLS):
    mask = y_true == c
    for series in X_norm[mask][:5]:   # 5 samples per class
        ax.plot(series, color=gt_colors[c], alpha=0.5, linewidth=1)
    ax.plot([], color=gt_colors[c], linewidth=2, label=f"Cluster {c}")
ax.set_title("Ground Truth Classes (5 samples each)", fontsize=11)
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)

# Panel 2: k-Shape cluster centroids
ax = axes[0, 1]
for c in range(best_k):
    ax.plot(centroids_kshape[c], color=colors[c], linewidth=2.5, label=f"Centroid {c}")
ax.set_title(f"k-Shape Cluster Centroids (k={best_k})", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 3: k-Shape assignments
ax = axes[1, 0]
for c in range(best_k):
    mask = labels_kshape == c
    for series in X_norm[mask][:6]:
        ax.plot(series, color=colors[c], alpha=0.35, linewidth=1.0)
    ax.plot(centroids_kshape[c], color=colors[c], linewidth=2.5, label=f"k={c} (n={mask.sum()})")
ax.set_title("k-Shape Clusters with Centroids", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 4: Elbow curve
ax = axes[1, 1]
ks = list(k_range)
ax2 = ax.twinx()
ax.plot(ks, inertias, "o-", color="#2196F3", linewidth=2, markersize=7, label="Inertia")
ax2.plot(ks, silhouettes, "s--", color="#FF5722", linewidth=2, markersize=7, label="Silhouette")
ax.axvline(best_k, color="gray", linestyle=":", linewidth=1.5, label=f"Best k={best_k}")
ax.set_xlabel("k"); ax.set_ylabel("Inertia", color="#2196F3")
ax2.set_ylabel("Silhouette Score", color="#FF5722")
ax.set_title("k-Shape Elbow + Silhouette Curve", fontsize=11)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
ax.grid(alpha=0.3)

# Panel 5: Dendrogram (subset for readability)
ax = axes[2, 0]
subset   = np.random.choice(len(X_norm), 40, replace=False)
dv_sub   = pdist(X_norm[subset], metric="euclidean")
Z_sub    = linkage(dv_sub, method="average")
dendrogram(Z_sub, ax=ax, leaf_font_size=7, color_threshold=Z_sub[-best_k, 2])
ax.axhline(Z_sub[-best_k, 2], color="red", linestyle="--", linewidth=1.5,
           label=f"Cut for k={best_k}")
ax.set_title("Hierarchical Clustering Dendrogram (subset, n=40)", fontsize=11)
ax.legend(fontsize=9); ax.set_xlabel("Sample index"); ax.set_ylabel("Distance")

# Panel 6: ARI comparison bar
ax = axes[2, 1]
method_names = eval_df.index.tolist()
ari_vals     = eval_df["ARI↑"].values
nmi_vals     = eval_df["NMI↑"].values
x_pos        = np.arange(len(method_names))
w            = 0.35
bars1 = ax.bar(x_pos - w/2, ari_vals, w, label="ARI", color="#2196F3")
bars2 = ax.bar(x_pos + w/2, nmi_vals, w, label="NMI", color="#4CAF50")
ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=9)
ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=9)
ax.set_xticks(x_pos); ax.set_xticklabels(method_names, rotation=20, ha="right", fontsize=9)
ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
ax.set_title("External Validation: ARI vs. NMI", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

plt.suptitle("Time Series Clustering: k-Shape, DTW k-Means, Hierarchical",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("ts_clustering_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: ts_clustering_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Per-Cluster Silhouette Analysis
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.metrics import silhouette_samples
s_vals = silhouette_samples(X_norm, labels_kshape)

print(f"\nPer-cluster silhouette (k-Shape, k={best_k}):")
for c in sorted(set(labels_kshape)):
    mask = labels_kshape == c
    print(f"  Cluster {c} (n={mask.sum():3d}): mean={s_vals[mask].mean():.4f}, "
          f"min={s_vals[mask].min():.4f}")
