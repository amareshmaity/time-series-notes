"""
code/01_dtw_classification.py
================================
Module 10 — Classification & Clustering
Practical: DTW distance and kNN-DTW classification.

Demonstrates:
  - Pure Python DTW implementation with Sakoe-Chiba band
  - LB_Keogh lower bound for fast pruning
  - Warp path visualization
  - 1-NN DTW classifier on synthetic TS
  - Warping window tuning via LOO cross-validation
  - Comparison: Euclidean vs. DTW kNN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────────────────────────────────────
# 1. DTW Core Implementations
# ─────────────────────────────────────────────────────────────────────────────

def dtw_distance(x, y, window=None):
    """DTW distance with optional Sakoe-Chiba band."""
    Tx, Ty = len(x), len(y)
    if window is None:
        window = max(Tx, Ty)
    D = np.full((Tx + 1, Ty + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, Tx + 1):
        for j in range(max(1, i - window), min(Ty, i + window) + 1):
            cost    = (x[i-1] - y[j-1])**2
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    return float(np.sqrt(D[Tx, Ty]))


def dtw_warp_path(x, y):
    """DTW distance + optimal warp path (for visualization)."""
    Tx, Ty = len(x), len(y)
    D = np.full((Tx + 1, Ty + 1), np.inf); D[0, 0] = 0.0
    for i in range(1, Tx + 1):
        for j in range(1, Ty + 1):
            D[i, j] = (x[i-1]-y[j-1])**2 + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    path = [(Tx, Ty)]
    i, j = Tx, Ty
    while i > 1 or j > 1:
        candidates = {(i-1,j-1): D[i-1,j-1], (i-1,j): D[i-1,j], (i,j-1): D[i,j-1]}
        i, j = min(candidates, key=candidates.get)
        path.append((i, j))
    path.reverse()
    return float(np.sqrt(D[Tx, Ty])), [(i-1, j-1) for i, j in path]


def lb_keogh(x, y, window):
    """LB_Keogh lower bound — O(T), much faster than DTW."""
    T  = len(x); lb = 0.0
    for i in range(T):
        lo   = max(0, i - window); hi = min(T-1, i + window)
        emax = x[lo:hi+1].max();  emin = x[lo:hi+1].min()
        if y[i] > emax:   lb += (y[i] - emax)**2
        elif y[i] < emin: lb += (emin - y[i])**2
    return float(np.sqrt(lb))


def z_normalize(X):
    """Z-normalize each series in (n, T) array independently."""
    mu    = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sigma


# ─────────────────────────────────────────────────────────────────────────────
# 2. Synthetic Dataset — Two Classes with Phase Variation
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
N_PER_CLASS = 50
T           = 80

def make_class(freq, phase_var, amp_var, n=N_PER_CLASS):
    """Generate sinusoid samples with phase and amplitude variation."""
    X = []
    for _ in range(n):
        t     = np.linspace(0, 4*np.pi, T)
        phase = np.random.uniform(-phase_var, phase_var)
        amp   = 1.0 + np.random.uniform(-amp_var, amp_var)
        noise = np.random.normal(0, 0.15, T)
        X.append(amp * np.sin(freq * t + phase) + noise)
    return np.array(X)

# Class 0: slow sinusoid (freq=1), Class 1: fast sinusoid (freq=2)
X0     = make_class(1.0, 0.5, 0.2)
X1     = make_class(2.0, 0.5, 0.2)
X_full = np.vstack([X0, X1])
y_full = np.array([0]*N_PER_CLASS + [1]*N_PER_CLASS)

# Z-normalize
X_norm = z_normalize(X_full)

# Train/test split (temporal-safe: first 40 from each class = train)
train_mask = np.concatenate([np.arange(40), np.arange(50, 90)])
test_mask  = np.concatenate([np.arange(40, 50), np.arange(90, 100)])
X_train, y_train = X_norm[train_mask], y_full[train_mask]
X_test,  y_test  = X_norm[test_mask],  y_full[test_mask]

print(f"Dataset: N={len(X_full)}, T={T}, 2 classes")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. 1-NN Classifiers (Euclidean vs. DTW)
# ─────────────────────────────────────────────────────────────────────────────

def knn_predict(X_train, y_train, X_test, dist_fn):
    """1-NN classifier with custom distance function."""
    y_pred = []
    for q in X_test:
        dists = [dist_fn(q, X_train[j]) for j in range(len(X_train))]
        y_pred.append(y_train[np.argmin(dists)])
    return np.array(y_pred)


print("\n1-NN Euclidean:")
y_pred_ed = knn_predict(X_train, y_train, X_test,
                         lambda a, b: float(np.sqrt(((a-b)**2).sum())))
print(classification_report(y_test, y_pred_ed, target_names=["Class 0", "Class 1"]))

print("1-NN DTW (window=8, 10% of T):")
WINDOW = int(T * 0.10)
y_pred_dtw = knn_predict(X_train, y_train, X_test,
                          lambda a, b: dtw_distance(a, b, window=WINDOW))
print(classification_report(y_test, y_pred_dtw, target_names=["Class 0", "Class 1"]))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Warping Window Tuning via LOO-CV
# ─────────────────────────────────────────────────────────────────────────────

print("Tuning warping window via LOO-CV on training set...")
window_candidates = [0, 4, 8, 12, 16, 20, 30]
loo_accuracies    = []

for w in window_candidates:
    correct = 0
    for i in range(len(X_train)):
        X_ref = np.delete(X_train, i, 0)
        y_ref = np.delete(y_train, i)
        dists = [dtw_distance(X_train[i], X_ref[j], window=w) for j in range(len(X_ref))]
        correct += (y_ref[np.argmin(dists)] == y_train[i])
    loo_acc = correct / len(X_train)
    loo_accuracies.append(loo_acc)
    print(f"  w={w:3d}: LOO accuracy = {loo_acc:.4f}")

best_window = window_candidates[np.argmax(loo_accuracies)]
print(f"\nBest window: {best_window}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. LB_Keogh Pruning Speedup Demo
# ─────────────────────────────────────────────────────────────────────────────

print("\nLB_Keogh pruning demonstration...")
query       = X_test[0]
n_dtw_calls = 0; n_lb_prune = 0
best_dist   = np.inf

for j in range(len(X_train)):
    lb = lb_keogh(query, X_train[j], WINDOW)
    if lb < best_dist:
        d = dtw_distance(query, X_train[j], WINDOW)
        n_dtw_calls += 1
        if d < best_dist:
            best_dist = d
    else:
        n_lb_prune += 1

total     = len(X_train)
print(f"  Total reference series: {total}")
print(f"  Full DTW computed:      {n_dtw_calls}")
print(f"  Pruned by LB_Keogh:     {n_lb_prune} ({100*n_lb_prune/total:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualization
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Class examples
ax = axes[0, 0]
for i in range(5):
    ax.plot(X0[i], color="#2196F3", alpha=0.5, linewidth=1.2)
    ax.plot(X1[i], color="#F44336", alpha=0.5, linewidth=1.2)
ax.plot([], color="#2196F3", label="Class 0 (slow sinusoid)")
ax.plot([], color="#F44336", label="Class 1 (fast sinusoid)")
ax.set_title("Sample Time Series per Class", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 2: DTW warp path
ax = axes[0, 1]
x_ex = X_norm[0]; y_ex = X_norm[N_PER_CLASS]   # one from each class
dist, path = dtw_warp_path(x_ex[:30], y_ex[:30])   # short segment for clarity
ax.plot(x_ex[:30], "b-o", markersize=4, linewidth=1.5, label="Class 0")
ax.plot(y_ex[:30], "r-o", markersize=4, linewidth=1.5, label="Class 1")
for (i, j) in path[::3]:
    ax.plot([i, j], [x_ex[i], y_ex[j]], "k-", alpha=0.25, linewidth=0.8)
ax.set_title(f"DTW Warp Path (distance={dist:.3f})", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 3: Window tuning curve
ax = axes[1, 0]
ax.plot(window_candidates, loo_accuracies, "o-", color="#4CAF50", linewidth=2, markersize=8)
ax.axvline(best_window, color="red", linestyle="--", linewidth=1.5,
           label=f"Best: w={best_window}")
ax.set_xlabel("Warping Window"); ax.set_ylabel("LOO Accuracy")
ax.set_title("Warping Window Tuning (LOO-CV)", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 4: Accuracy comparison
methods  = ["1-NN Euclidean", "1-NN DTW (w=8)", f"1-NN DTW (w={best_window})"]
best_pred = knn_predict(X_train, y_train, X_test,
                         lambda a, b: dtw_distance(a, b, window=best_window))
accs     = [accuracy_score(y_test, y_pred_ed),
            accuracy_score(y_test, y_pred_dtw),
            accuracy_score(y_test, best_pred)]
colors_  = ["#9E9E9E", "#FF9800", "#4CAF50"]
bars     = axes[1, 1].bar(methods, accs, color=colors_, width=0.5)
axes[1, 1].bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
axes[1, 1].set_ylim(0, 1.1); axes[1, 1].set_ylabel("Test Accuracy")
axes[1, 1].set_title("Method Comparison", fontsize=11)
axes[1, 1].tick_params(axis="x", rotation=15)
axes[1, 1].grid(alpha=0.3, axis="y")

plt.suptitle("DTW-Based Time Series Classification", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("dtw_classification.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: dtw_classification.png")
