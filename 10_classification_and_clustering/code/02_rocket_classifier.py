"""
code/02_rocket_classifier.py
==============================
Module 10 — Classification & Clustering
Practical: ROCKET and feature-based classifiers on UCR-style datasets.

Demonstrates:
  - ROCKET pipeline (sktime) — the speed-accuracy champion
  - tsfresh feature extraction (with fallback to catch22 / manual features)
  - catch22 lightweight features
  - Multi-method benchmark with F1 and training-time comparison
  - Visualizing ROCKET kernel outputs
  - Confusion matrix comparison
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Multi-Class Dataset
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
N_PER  = 80     # samples per class
T      = 100    # series length
N_CLS  = 4

def make_dataset(n_per, T, n_cls, noise=0.2):
    """Generate 4 distinct pattern classes with noise."""
    X, y = [], []
    t = np.linspace(0, 4*np.pi, T)
    patterns = [
        lambda t: np.sin(t),                      # Class 0: sine
        lambda t: np.sin(2*t),                    # Class 1: double freq
        lambda t: np.sign(np.sin(t)),              # Class 2: square wave
        lambda t: np.where(t < 2*np.pi, t/(2*np.pi), 2 - t/(2*np.pi)),  # Class 3: triangle
    ]
    for c, pat in enumerate(patterns):
        for _ in range(n_per):
            scale = np.random.uniform(0.8, 1.2)
            shift = np.random.uniform(-0.3, 0.3)
            x = scale * pat(t + shift) + np.random.normal(0, noise, T)
            X.append(x); y.append(c)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

X_all, y_all = make_dataset(N_PER, T, N_CLS)
split        = int(0.75 * len(X_all))
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

# Z-normalize
def z_norm(X):
    mu = X.mean(axis=1, keepdims=True); s = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / s

X_train_n = z_norm(X_train)
X_test_n  = z_norm(X_test)

print(f"Dataset: {N_PER*N_CLS} samples, T={T}, {N_CLS} classes")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROCKET via sktime
# ─────────────────────────────────────────────────────────────────────────────

def run_rocket(X_train, y_train, X_test, n_kernels=10_000):
    """ROCKET + RidgeClassifierCV pipeline."""
    try:
        from sktime.transformations.panel.rocket import Rocket

        Xtr = X_train[:, np.newaxis, :]   # (n, 1, T)
        Xte = X_test[:,  np.newaxis, :]

        t0     = time.time()
        rocket = Rocket(num_kernels=n_kernels, random_state=42, n_jobs=-1)
        Xtr_f  = rocket.fit_transform(Xtr)
        Xte_f  = rocket.transform(Xte)

        clf    = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        clf.fit(Xtr_f, y_train)
        y_pred = clf.predict(Xte_f)
        elapsed = time.time() - t0

        return y_pred, elapsed, "ROCKET"

    except ImportError:
        print("sktime not available — ROCKET skipped")
        return None, None, "ROCKET"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Manual Feature Extraction (catch22 + statistics fallback)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_manual(X):
    """Extract statistical + spectral features from (n, T) array."""
    feats = []
    for s in X:
        f = {
            "mean":      s.mean(),
            "std":       s.std(),
            "min":       s.min(),
            "max":       s.max(),
            "range":     s.max() - s.min(),
            "skew":      float(pd.Series(s).skew()),
            "kurt":      float(pd.Series(s).kurt()),
            "rms":       float(np.sqrt((s**2).mean())),
            "zcr":       float((np.diff(np.sign(s)) != 0).mean()),
            "n_peaks":   int((np.diff(np.sign(np.diff(s))) < 0).sum()),
            "acf1":      float(pd.Series(s).autocorr(1) or 0.0),
            "acf5":      float(pd.Series(s).autocorr(5) or 0.0),
            "q25":       float(np.percentile(s, 25)),
            "q75":       float(np.percentile(s, 75)),
            "energy":    float((s**2).sum()),
            # FFT-based
            "fft_e1":   float(np.abs(np.fft.rfft(s)[1])),
            "fft_e2":   float(np.abs(np.fft.rfft(s)[2])),
            "fft_e3":   float(np.abs(np.fft.rfft(s)[3])),
            "spectral_entropy": float(-np.sum(np.abs(np.fft.rfft(s))**2 / (((np.abs(np.fft.rfft(s))**2).sum()+1e-12))) *
                                      np.log(np.abs(np.fft.rfft(s))**2 / ((np.abs(np.fft.rfft(s))**2).sum()+1e-12) + 1e-12)),
        }
        # Try catch22
        try:
            import pycatch22
            result = pycatch22.catch22_all(list(s))
            for name, val in zip(result["names"], result["values"]):
                f[f"c22_{name}"] = float(val) if np.isfinite(float(val)) else 0.0
        except ImportError:
            pass
        feats.append(f)
    return pd.DataFrame(feats)


def run_feature_rf(X_train, y_train, X_test, name="FeatureRF"):
    """Feature extraction + Random Forest classifier."""
    t0 = time.time()
    F_tr = extract_features_manual(X_train)
    F_te = extract_features_manual(X_test)

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("rf",      RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
    ])
    clf.fit(F_tr.values, y_train)
    y_pred  = clf.predict(F_te.values)
    elapsed = time.time() - t0
    return y_pred, elapsed, name


# ─────────────────────────────────────────────────────────────────────────────
# 4. 1-NN DTW Baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_dtw_knn(X_train, y_train, X_test, window=10):
    """Fast 1-NN DTW using tslearn (C backend)."""
    try:
        from tslearn.neighbors import KNeighborsTimeSeriesClassifier
        from tslearn.preprocessing import TimeSeriesScalerMeanVariance

        Xtr = X_train[:, :, np.newaxis]
        Xte = X_test[:,  :, np.newaxis]
        scaler = TimeSeriesScalerMeanVariance()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        t0  = time.time()
        clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw",
                                              metric_params={"sakoe_chiba_radius": window},
                                              n_jobs=-1)
        clf.fit(Xtr, y_train)
        y_pred  = clf.predict(Xte)
        elapsed = time.time() - t0
        return y_pred, elapsed, "1-NN DTW"
    except ImportError:
        print("tslearn not available — DTW kNN skipped")
        return None, None, "1-NN DTW"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Run Benchmark
# ─────────────────────────────────────────────────────────────────────────────

print("\nRunning classifier benchmark...")
results = []

# ROCKET
y_pred_rocket, t_rocket, name_r = run_rocket(X_train_n, y_train, X_test_n)
if y_pred_rocket is not None:
    results.append({"Method": name_r, "Accuracy": accuracy_score(y_test, y_pred_rocket),
                    "Macro-F1": f1_score(y_test, y_pred_rocket, average="macro"),
                    "Time(s)": round(t_rocket, 2)})
    print(f"  {name_r}: acc={results[-1]['Accuracy']:.4f}, time={t_rocket:.2f}s")

# Feature + RF
y_pred_rf, t_rf, name_f = run_feature_rf(X_train_n, y_train, X_test_n)
results.append({"Method": name_f, "Accuracy": accuracy_score(y_test, y_pred_rf),
                "Macro-F1": f1_score(y_test, y_pred_rf, average="macro"),
                "Time(s)": round(t_rf, 2)})
print(f"  {name_f}: acc={results[-1]['Accuracy']:.4f}, time={t_rf:.2f}s")

# 1-NN DTW
y_pred_dtw, t_dtw, name_d = run_dtw_knn(X_train_n, y_train, X_test_n, window=10)
if y_pred_dtw is not None:
    results.append({"Method": name_d, "Accuracy": accuracy_score(y_test, y_pred_dtw),
                    "Macro-F1": f1_score(y_test, y_pred_dtw, average="macro"),
                    "Time(s)": round(t_dtw, 2)})
    print(f"  {name_d}: acc={results[-1]['Accuracy']:.4f}, time={t_dtw:.2f}s")

result_df = pd.DataFrame(results).set_index("Method")
print("\n" + "="*55)
print("BENCHMARK RESULTS")
print("="*55)
print(result_df.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualization
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
cls_names = [f"Class {c}" for c in range(N_CLS)]
colors    = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

# Panel 1: Class prototypes
ax = axes[0, 0]
t_ = np.linspace(0, 4*np.pi, T)
proto = [np.sin(t_), np.sin(2*t_), np.sign(np.sin(t_)),
         np.where(t_ < 2*np.pi, t_/(2*np.pi), 2 - t_/(2*np.pi))]
for c, p in enumerate(proto):
    ax.plot(p, color=colors[c], linewidth=2, label=cls_names[c])
ax.set_title("Class Prototypes", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 2: Accuracy bar chart
ax = axes[0, 1]
if results:
    method_names = result_df.index.tolist()
    accs         = result_df["Accuracy"].values
    bars         = ax.bar(method_names, accs, color=["#2196F3","#4CAF50","#FF9800"][:len(method_names)])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Test Accuracy")
    ax.set_title("Classifier Accuracy Comparison", fontsize=11)
    ax.tick_params(axis="x", rotation=15); ax.grid(alpha=0.3, axis="y")

# Panel 3: Training time comparison
ax = axes[1, 0]
if results:
    times = result_df["Time(s)"].values
    bars  = ax.bar(method_names, times, color=["#2196F3","#4CAF50","#FF9800"][:len(method_names)])
    ax.bar_label(bars, fmt="%.2fs", padding=3, fontsize=10)
    ax.set_ylabel("Training + Inference Time (s)")
    ax.set_title("Speed Comparison", fontsize=11)
    ax.tick_params(axis="x", rotation=15); ax.grid(alpha=0.3, axis="y")

# Panel 4: Confusion matrix of best method
best_pred = y_pred_rocket if y_pred_rocket is not None else y_pred_rf
ax = axes[1, 1]
cm = confusion_matrix(y_test, best_pred)
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(N_CLS)); ax.set_xticklabels(cls_names, rotation=45, ha="right")
ax.set_yticks(range(N_CLS)); ax.set_yticklabels(cls_names)
for i in range(N_CLS):
    for j in range(N_CLS):
        ax.text(j, i, cm[i,j], ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=12)
ax.set_title("Confusion Matrix (Best Method)", fontsize=11)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("ROCKET & Feature-Based Time Series Classification",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("rocket_classifier_benchmark.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: rocket_classifier_benchmark.png")
