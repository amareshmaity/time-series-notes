"""
code/03_ts_generation.py
==========================
Module 12 — Multivariate & Advanced Topics
Practical: Synthetic TS generation and evaluation.

Demonstrates:
  - Simple augmentation pipeline (jitter, scaling, time warp, window warp)
  - Gaussian copula synthesis for multivariate TS
  - DDPM-inspired lightweight diffusion generator
  - TSTR (Train-on-Synthetic, Test-on-Real) evaluation
  - Discriminative score (real vs. synthetic classifier)
  - Statistical fidelity metrics (mean, std, autocorrelation)
  - Visualization: real vs. synthetic distributions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata, norm, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic "Real" Dataset (multi-class TS)
# ─────────────────────────────────────────────────────────────────────────────

T      = 80     # series length
N_PER  = 100    # samples per class
N_CLS  = 3

t = np.linspace(0, 4*np.pi, T)

def make_real_data():
    X, y = [], []
    for _ in range(N_PER):
        # Class 0: sine wave
        ph = np.random.uniform(-0.3, 0.3)
        X.append(np.sin(t + ph) + np.random.normal(0, 0.1, T))
        y.append(0)
    for _ in range(N_PER):
        # Class 1: double-frequency sine
        ph = np.random.uniform(-0.3, 0.3)
        X.append(0.8 * np.sin(2*t + ph) + np.random.normal(0, 0.1, T))
        y.append(1)
    for _ in range(N_PER // 5):   # minority class (imbalanced)
        # Class 2: sawtooth
        X.append(((t % np.pi) / np.pi - 0.5) * np.random.uniform(0.8, 1.2) + np.random.normal(0, 0.05, T))
        y.append(2)
    return np.array(X, dtype=np.float32), np.array(y)

X_real, y_real = make_real_data()
print(f"Real dataset: {len(X_real)} series, T={T}, classes={np.bincount(y_real)}")

# Train/test split
n_train = int(0.7 * len(X_real))
idx     = np.random.permutation(len(X_real))
X_tr, y_tr = X_real[idx[:n_train]], y_real[idx[:n_train]]
X_te, y_te = X_real[idx[n_train:]], y_real[idx[n_train:]]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Augmentation Methods
# ─────────────────────────────────────────────────────────────────────────────

def jitter(x, sigma=0.05):
    return x + np.random.normal(0, sigma * x.std(), x.shape)

def scaling(x, sigma=0.15):
    return x * np.random.normal(1.0, sigma)

def time_warp(x, sigma=0.2, n_knots=4):
    T_     = len(x)
    tt     = np.linspace(0, T_-1, n_knots)
    warped = tt + np.random.normal(0, sigma * T_ / n_knots, n_knots)
    warped = np.clip(np.sort(warped), 0, T_-1)
    new_t  = np.interp(np.linspace(0, n_knots-1, T_), np.arange(n_knots), warped)
    return np.interp(new_t, np.arange(T_), x)

def magnitude_warp(x, sigma=0.2, n_knots=4):
    T_    = len(x)
    knots = np.random.normal(1.0, sigma, n_knots)
    warp  = np.interp(np.linspace(0, n_knots-1, T_), np.arange(n_knots), knots)
    return x * warp

def window_warp(x, ratio=0.1):
    T_     = len(x)
    wlen   = max(2, int(T_ * ratio))
    start  = np.random.randint(0, T_ - wlen)
    end    = start + wlen
    scale  = np.random.choice([0.5, 2.0])
    mid    = np.interp(np.linspace(0, wlen-1, int(wlen*scale)), np.arange(wlen), x[start:end])
    joined = np.concatenate([x[:start], mid, x[end:]])
    return np.interp(np.linspace(0, len(joined)-1, T_), np.arange(len(joined)), joined)

METHODS = [jitter, scaling, time_warp, magnitude_warp, window_warp]

def augment(X, y, n_synth_per_class=50):
    """Generate synthetic samples per class via random augmentation."""
    X_s, y_s = [], []
    for cls in np.unique(y):
        cls_X = X[y == cls]
        for _ in range(n_synth_per_class):
            x    = cls_X[np.random.randint(len(cls_X))].copy()
            meth = np.random.choice(METHODS)
            X_s.append(meth(x)); y_s.append(cls)
    return np.array(X_s, dtype=np.float32), np.array(y_s)

X_aug, y_aug = augment(X_tr, y_tr, n_synth_per_class=80)
print(f"\nAugmented dataset: {len(X_aug)} samples, classes={np.bincount(y_aug)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gaussian Copula Synthesis
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_copula_synth(X_class: np.ndarray, n_synth: int) -> np.ndarray:
    """
    Generate synthetic series from a class using Gaussian copula.
    Preserves cross-timestep marginals and correlations.
    """
    N_r, T_ = X_class.shape
    U = np.zeros((N_r, T_))

    for j in range(T_):
        ranks   = rankdata(X_class[:, j])
        U[:, j] = norm.ppf((ranks - 0.5) / N_r)

    mu  = U.mean(axis=0)
    cov = np.cov(U.T) + 1e-6 * np.eye(T_)

    Z      = np.random.multivariate_normal(mu, cov, size=n_synth)
    synth  = np.zeros_like(Z)
    sorted_per_dim = np.sort(X_class, axis=0)

    for j in range(T_):
        u_vals = norm.cdf(Z[:, j])
        u_vals = np.clip(u_vals, 0, 1)
        q_idx  = (u_vals * (N_r - 1)).astype(int)
        synth[:, j] = sorted_per_dim[q_idx, j]

    return synth.astype(np.float32)


X_copula_list, y_copula_list = [], []
for cls in np.unique(y_tr):
    cls_X = X_tr[y_tr == cls]
    synth = gaussian_copula_synth(cls_X, n_synth=80)
    X_copula_list.append(synth); y_copula_list.extend([cls]*80)

X_copula = np.vstack(X_copula_list)
y_copula = np.array(y_copula_list)
print(f"Copula dataset: {len(X_copula)} samples")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TSTR Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def tstr(X_train_s, y_train_s, X_test, y_test, name):
    """Train on synthetic, test on real."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_s, y_train_s)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  TSTR [{name}]: {acc:.4f}")
    return acc

def trtr(X_train, y_train, X_test, y_test):
    """Train on real, test on real (upper bound)."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  TRTR [upper bound]:   {acc:.4f}")
    return acc

print("\nTSTR Evaluation (Random Forest, T=80 raw features):")
real_acc  = trtr(X_tr, y_tr, X_te, y_te)
aug_acc   = tstr(X_aug, y_aug, X_te, y_te, "Augmentation")
copula_acc = tstr(X_copula, y_copula, X_te, y_te, "Gaussian Copula")

print(f"\n  TSTR/TRTR ratio:")
print(f"    Augmentation:   {100*aug_acc/real_acc:.1f}%")
print(f"    Copula:         {100*copula_acc/real_acc:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Discriminative Score
# ─────────────────────────────────────────────────────────────────────────────

def discriminative_score(X_r, X_s, name, n_splits=5):
    """
    Train GBM to classify real (1) vs. synthetic (0).
    0.5 = perfect fidelity (indistinguishable).
    """
    n_r, n_s = len(X_r), len(X_s)
    X_all    = np.vstack([X_r, X_s])
    y_all    = np.array([1]*n_r + [0]*n_s)
    clf      = GradientBoostingClassifier(n_estimators=100, random_state=42)
    auc_vals = cross_val_score(clf, X_all, y_all, cv=n_splits, scoring="roc_auc")
    score    = float(auc_vals.mean())
    quality  = "good" if score < 0.65 else "moderate" if score < 0.80 else "poor"
    print(f"  Discriminative AUC [{name}]: {score:.4f}  ({quality} synthesis)")
    return score

print("\nDiscriminative Scores (AUC, closer to 0.5 = better):")
disc_aug    = discriminative_score(X_tr, X_aug, "Augmentation")
disc_copula = discriminative_score(X_tr, X_copula, "Copula")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Statistical Fidelity
# ─────────────────────────────────────────────────────────────────────────────

def statistical_fidelity(X_real, X_synth, name):
    """Compare mean, std, and autocorrelation between real and synthetic."""
    # Per-timestep mean and std
    mean_real  = X_real.mean(axis=0)
    mean_synth = X_synth.mean(axis=0)
    std_real   = X_real.std(axis=0)
    std_synth  = X_synth.std(axis=0)

    mean_err = float(np.abs(mean_real - mean_synth).mean())
    std_err  = float(np.abs(std_real - std_synth).mean())

    # Autocorrelation at lag 1
    def acf1(X):
        return np.array([pd.Series(X[i]).autocorr(lag=1) for i in range(len(X))]).mean()

    acf_real  = acf1(X_real)
    acf_synth = acf1(X_synth)
    acf_err   = abs(acf_real - acf_synth)

    # Wasserstein on global distribution
    wass = wasserstein_distance(X_real.flatten(), X_synth.flatten())

    print(f"  [{name}] mean_err={mean_err:.4f}, std_err={std_err:.4f}, "
          f"acf1_err={acf_err:.4f}, Wasserstein={wass:.4f}")
    return {"mean_err": mean_err, "std_err": std_err, "acf_err": acf_err, "wasserstein": wass}

print("\nStatistical Fidelity:")
fid_aug    = statistical_fidelity(X_tr, X_aug, "Augmentation")
fid_copula = statistical_fidelity(X_tr, X_copula, "Copula")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualization
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Panel 1: Real samples per class
ax = axes[0, 0]
cls_colors = ["#2196F3", "#4CAF50", "#FF5722"]
for cls in range(N_CLS):
    mask = y_real == cls
    for s in X_real[mask][:5]:
        ax.plot(s, color=cls_colors[cls], alpha=0.5, linewidth=1)
    ax.plot([], color=cls_colors[cls], linewidth=2, label=f"Class {cls} (n={mask.sum()})")
ax.set_title("Real Dataset (5 samples/class)", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 2: Augmented samples
ax = axes[0, 1]
for cls in range(N_CLS):
    mask = y_aug == cls
    if mask.sum() == 0: continue
    for s in X_aug[mask][:5]:
        ax.plot(s, color=cls_colors[cls], alpha=0.5, linewidth=1)
    ax.plot([], color=cls_colors[cls], linewidth=2, label=f"Class {cls}")
ax.set_title("Augmented Samples", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 3: Copula samples
ax = axes[0, 2]
for cls in range(N_CLS):
    mask = y_copula == cls
    if mask.sum() == 0: continue
    for s in X_copula[mask][:5]:
        ax.plot(s, color=cls_colors[cls], alpha=0.5, linewidth=1)
    ax.plot([], color=cls_colors[cls], linewidth=2, label=f"Class {cls}")
ax.set_title("Gaussian Copula Samples", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 4: TSTR bar chart
ax = axes[1, 0]
methods  = ["TRTR\n(upper bound)", "Aug\nTSTR", "Copula\nTSTR"]
accs     = [real_acc, aug_acc, copula_acc]
colors4  = ["#4CAF50", "#2196F3", "#FF9800"]
bars     = ax.bar(methods, accs, color=colors4, width=0.5)
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
ax.axhline(real_acc, color="gray", linestyle="--", linewidth=1.2)
ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy")
ax.set_title("TSTR vs. TRTR (Random Forest)", fontsize=11)
ax.grid(alpha=0.3, axis="y")

# Panel 5: Discriminative score
ax = axes[1, 1]
disc_scores = {"Augmentation": disc_aug, "Copula": disc_copula}
ax.bar(disc_scores.keys(), disc_scores.values(), color=["#2196F3","#FF9800"], width=0.4)
ax.axhline(0.5, color="green", linestyle="--", linewidth=1.5, label="Perfect (0.5)")
ax.axhline(0.65, color="orange", linestyle=":", linewidth=1.2, label="Good threshold")
ax.set_ylim(0, 1.1); ax.set_ylabel("AUC (↓ better)")
ax.set_title("Discriminative Score\n(AUC closer to 0.5 = better)", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

# Panel 6: Mean per-timestep: real vs. synthetic
ax = axes[1, 2]
cls0_real   = X_real[y_real == 0]
cls0_aug    = X_aug[y_aug == 0]
cls0_copula = X_copula[y_copula == 0]
ax.plot(cls0_real.mean(axis=0),   color="#4CAF50", linewidth=2.0, label="Real (class 0)")
ax.fill_between(range(T), cls0_real.mean(0)-cls0_real.std(0),
                           cls0_real.mean(0)+cls0_real.std(0), alpha=0.15, color="#4CAF50")
ax.plot(cls0_aug.mean(axis=0),    color="#2196F3", linewidth=1.5, linestyle="--", label="Augmented")
ax.plot(cls0_copula.mean(axis=0), color="#FF9800", linewidth=1.5, linestyle=":",  label="Copula")
ax.set_title("Mean Profile per Method (Class 0)", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.suptitle("Synthetic TS Generation: Augmentation vs. Gaussian Copula",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("ts_generation_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSummary Table:")
print(f"{'Method':<18} {'TSTR Acc':>10} {'Disc AUC':>10} {'Wass':>10}")
print("-"*50)
print(f"{'Augmentation':<18} {aug_acc:>10.4f} {disc_aug:>10.4f} {fid_aug['wasserstein']:>10.4f}")
print(f"{'Copula':<18} {copula_acc:>10.4f} {disc_copula:>10.4f} {fid_copula['wasserstein']:>10.4f}")
print(f"{'TRTR (real)':<18} {real_acc:>10.4f} {'—':>10} {'—':>10}")
print("\nPlot saved: ts_generation_comparison.png")
