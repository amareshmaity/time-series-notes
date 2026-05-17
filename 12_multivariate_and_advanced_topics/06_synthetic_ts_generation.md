# 06 — Synthetic Time Series Generation

> **Module**: 12 Multivariate & Advanced Topics | **File**: 6 of 6
>
> Synthetic TS generation addresses data scarcity, class imbalance, and privacy constraints. This note covers TimeGAN, augmentation strategies, statistical generators, and the train-on-synthetic/test-on-real (TSTR) evaluation protocol.

---

## Table of Contents

1. [Why Generate Synthetic TS?](#1-why-generate-synthetic-ts)
2. [Augmentation Strategies](#2-augmentation-strategies)
3. [TimeGAN Architecture](#3-timegan-architecture)
4. [Statistical Generators](#4-statistical-generators)
5. [TSTR Evaluation Protocol](#5-tstr-evaluation-protocol)
6. [Privacy-Preserving Synthesis](#6-privacy-preserving-synthesis)
7. [Implementation](#7-implementation)

---

## 1. Why Generate Synthetic TS?

```
USE CASES:

  DATA SCARCITY:
    Rare events: equipment failure, fraud, disease onset.
    Sensor outages: gaps in historical data.
    New product launch: no historical demand.

  CLASS IMBALANCE:
    Medical: 99% normal ECGs, 1% arrhythmias.
    Generating synthetic minority-class samples → better classifiers.

  PRIVACY / DATA PROTECTION:
    GDPR: cannot share raw patient data.
    Synthetic TS preserves statistical properties without PII.

  DATA AUGMENTATION:
    More diverse training data → reduced overfitting for deep models.
    Standard augmentation (jitter, scaling, window slicing) is cheap.
    Generative augmentation (TimeGAN, diffusion) captures complex patterns.

  SIMULATION:
    Generate scenarios for stress testing:
    "What if demand spikes 3x next quarter?"
```

---

## 2. Augmentation Strategies

### 2.1 Simple Transformations

```python
import numpy as np

def jitter(x: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """Add Gaussian noise. Preserves shape; σ = fraction of std."""
    return x + np.random.normal(0, sigma * x.std(), x.shape)

def scaling(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """Randomly scale amplitude. σ controls variation."""
    factor = np.random.normal(1.0, sigma)
    return x * factor

def window_slice(x: np.ndarray, reduce_ratio: float = 0.9) -> np.ndarray:
    """Crop a random contiguous window; stretch back to original length."""
    T     = len(x)
    start = np.random.randint(0, T - int(T * reduce_ratio))
    end   = start + int(T * reduce_ratio)
    return np.interp(np.linspace(0, len(x[start:end])-1, T),
                     np.arange(len(x[start:end])), x[start:end])

def time_warp(x: np.ndarray, sigma: float = 0.2, n_knots: int = 4) -> np.ndarray:
    """Random non-linear time stretching via smooth warp path."""
    T      = len(x)
    tt     = np.linspace(0, T-1, n_knots)
    warped = tt + np.random.normal(0, sigma * T / n_knots, n_knots)
    warped = np.clip(np.sort(warped), 0, T-1)
    new_t  = np.interp(np.linspace(0, n_knots-1, T), np.arange(n_knots), warped)
    return np.interp(new_t, np.arange(T), x)

def magnitude_warp(x: np.ndarray, sigma: float = 0.2, n_knots: int = 4) -> np.ndarray:
    """Random smooth multiplicative scaling along time axis."""
    T       = len(x)
    knots   = np.random.normal(1.0, sigma, n_knots)
    warp    = np.interp(np.linspace(0, n_knots-1, T), np.arange(n_knots), knots)
    return x * warp

def augment_dataset(
    X: np.ndarray,
    n_augmented: int,
    methods: list = None,
) -> np.ndarray:
    """
    Generate n_augmented synthetic samples by applying random augmentations.

    Parameters
    ----------
    X            : (N, T) original dataset
    n_augmented  : number of synthetic samples to generate
    methods      : list of augmentation functions to apply
    """
    if methods is None:
        methods = [jitter, scaling, time_warp, magnitude_warp]

    N  = len(X)
    aug_samples = []
    for _ in range(n_augmented):
        # Pick random original + random method
        x      = X[np.random.randint(N)].copy()
        method = np.random.choice(methods)
        aug_samples.append(method(x))

    return np.array(aug_samples)
```

### 2.2 Window Warping (WWarp)

```python
def window_warp(x: np.ndarray, window_ratio: float = 0.1, scales=(0.5, 2.0)) -> np.ndarray:
    """
    Select a random window; warp its speed (compress or stretch).
    Preserves global structure while introducing local speed variation.
    """
    T          = len(x)
    wlen       = max(2, int(T * window_ratio))
    start      = np.random.randint(0, T - wlen)
    end        = start + wlen
    scale      = np.random.choice(scales)

    left   = x[:start]
    middle = np.interp(np.linspace(0, wlen-1, int(wlen * scale)),
                        np.arange(wlen), x[start:end])
    right  = x[end:]

    warped = np.concatenate([left, middle, right])
    # Resize back to T
    return np.interp(np.linspace(0, len(warped)-1, T), np.arange(len(warped)), warped)
```

---

## 3. TimeGAN Architecture

### 3.1 Components

```
TimeGAN (Yoon et al., NeurIPS 2019):
  Addresses GAN instability for sequential data via 4-module design.

  Modules:
    1. Embedder (E):    X → H  (map real data to latent space)
    2. Recovery (R):    H → X̂  (map latent → reconstructed data)
    3. Generator (G):   Z → Ĥ  (map random noise to latent space)
    4. Discriminator (D): H → [0,1] (real vs. generated in latent space)

  Training objectives:
    Autoencoding loss:  ||X - R(E(X))||²  (trains E and R)
    GAN loss:          D(Ĥ) → 1, D(E(X)) → 1  (trains D and G)
    Supervised loss:   ||Ĥ_t - G(Ĥ_{t-1})||²  (step-wise prediction in latent)
    
  Key insight:
    Supervised loss forces the generator to learn TEMPORAL dynamics
    (each latent step must predict the next), preventing mode collapse.

  Loss combination:
    L_total = λ_ae · L_ae + L_gan + λ_sup · L_sup
```

```python
import torch
import torch.nn as nn

class TimeGANEmbedder(nn.Module):
    """Embedder: maps real TS to latent space."""
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers=3):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x):
        h, _ = self.rnn(x)
        return torch.sigmoid(self.out(h))

class TimeGANRecovery(nn.Module):
    """Recovery: maps latent space back to original TS dimension."""
    def __init__(self, latent_dim, hidden_dim, output_dim, n_layers=3):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)
    def forward(self, h):
        x, _ = self.rnn(h)
        return self.out(x)

class TimeGANGenerator(nn.Module):
    """Generator: maps noise Z to latent space Ĥ."""
    def __init__(self, noise_dim, hidden_dim, latent_dim, n_layers=3):
        super().__init__()
        self.rnn = nn.GRU(noise_dim, hidden_dim, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, latent_dim)
    def forward(self, z):
        h, _ = self.rnn(z)
        return torch.sigmoid(self.out(h))

class TimeGANDiscriminator(nn.Module):
    """Discriminator: real vs. generated in LATENT space."""
    def __init__(self, latent_dim, hidden_dim, n_layers=3):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)
    def forward(self, h):
        x, _ = self.rnn(h)
        return self.out(x).squeeze(-1)
```

### 3.2 Training Loop (Sketch)

```python
def timegan_training_sketch():
    """
    TimeGAN training in 3 phases:
      Phase 1: Autoencoder warmup (Embedder + Recovery)
      Phase 2: Supervised loss (Generator, initialized from Embedder)
      Phase 3: Joint GAN training (all 4 modules)
    """
    # Phase 1: Autoencoder
    #   Minimize ||X - Recovery(Embedder(X))||²

    # Phase 2: Supervised loss
    #   Embedder gives H_real; Generator learns H_real step-by-step
    #   Minimize ||H_real_{t} - Generator(H_real_{t-1})||²
    #   This initializes Generator to match real temporal dynamics.

    # Phase 3: GAN
    #   Z ~ Uniform(0,1) → H_fake = Generator(Z)
    #   Discriminator loss: D(H_real) → 1, D(H_fake) → 0
    #   Generator loss: D(H_fake) → 1
    #   + supervised loss + autoencoder loss (scaled)
    pass
```

---

## 4. Statistical Generators

### 4.1 Gaussian Copula for Multivariate TS

```python
from scipy.stats import norm, rankdata

def gaussian_copula_synthetic(
    df: pd.DataFrame,
    n_samples: int,
    preserve_autocorr: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic multivariate TS using Gaussian copula.
    Preserves marginal distributions and cross-correlations.

    Steps:
      1. Map each column to uniform via rank transform
      2. Apply normal quantile transform → pseudo-normal data
      3. Fit multivariate Gaussian to capture correlations
      4. Sample from the Gaussian → back-transform to original marginals
    """
    import pandas as pd

    n, D = df.shape
    U    = np.zeros_like(df.values, dtype=float)

    # Step 1-2: Map to uniform → normal
    for i, col in enumerate(df.columns):
        ranks = rankdata(df[col].values)
        U[:, i] = norm.ppf((ranks - 0.5) / n)

    # Step 3: Fit multivariate Gaussian
    mu  = U.mean(axis=0)
    cov = np.cov(U.T)

    # Step 4: Sample and back-transform
    Z  = np.random.multivariate_normal(mu, cov, size=n_samples)
    synth = pd.DataFrame(index=range(n_samples))

    for i, col in enumerate(df.columns):
        # Normal CDF → uniform → empirical quantile
        u_vals = norm.cdf(Z[:, i])
        u_vals = np.clip(u_vals, 1e-6, 1 - 1e-6)
        # Empirical quantile function (sorted original values)
        sorted_orig = np.sort(df[col].values)
        q_idx = (u_vals * (n - 1)).astype(int)
        synth[col] = sorted_orig[q_idx]

    return synth


def arima_synthetic(
    series: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """
    Fit ARIMA to real series; generate synthetic samples by simulation.
    Preserves trend, seasonality, and autocorrelation structure.
    """
    from statsmodels.tsa.arima.model import ARIMA
    model   = ARIMA(series, order=(2, 1, 2))
    results = model.fit()
    synth   = [results.simulate(len(series)).values for _ in range(n_samples)]
    return np.array(synth)
```

---

## 5. TSTR Evaluation Protocol

```python
def tstr_evaluation(
    X_real_train: np.ndarray,
    y_real_train: np.ndarray,
    X_real_test:  np.ndarray,
    y_real_test:  np.ndarray,
    X_synth:      np.ndarray,
    y_synth:      np.ndarray,
    model_factory = None,
) -> dict:
    """
    Train-on-Synthetic / Test-on-Real (TSTR) evaluation.

    Compares:
      TRTR: Train Real → Test Real  (upper bound)
      TSTR: Train Synth → Test Real  (quality of synthetic data)
      TSTS: Train Synth → Test Synth (lower bound)

    Parameters
    ----------
    X_real_train, y_real_train : real training data
    X_real_test, y_real_test   : real test data (held-out)
    X_synth, y_synth           : synthetic training data
    model_factory              : callable() → sklearn classifier/regressor

    Returns
    -------
    dict with TRTR, TSTR, TSTS scores
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    if model_factory is None:
        model_factory = lambda: RandomForestClassifier(n_estimators=100, random_state=42)

    # TRTR: upper bound
    m_trtr = model_factory()
    m_trtr.fit(X_real_train, y_real_train)
    trtr   = accuracy_score(y_real_test, m_trtr.predict(X_real_test))

    # TSTR: synthetic quality measure
    m_tstr = model_factory()
    m_tstr.fit(X_synth, y_synth)
    tstr   = accuracy_score(y_real_test, m_tstr.predict(X_real_test))

    # TSTS: lower bound
    m_tsts = model_factory()
    m_tsts.fit(X_synth, y_synth)
    tsts   = accuracy_score(y_synth, m_tsts.predict(X_synth))

    ratio  = tstr / trtr if trtr > 0 else 0.0

    result = {"TRTR": round(trtr, 4), "TSTR": round(tstr, 4),
              "TSTS": round(tsts, 4), "TSTR/TRTR": round(ratio, 4)}

    print("TSTR Evaluation:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
    print(f"  Quality: {100*ratio:.1f}% of real data performance")

    return result
```

### 5.1 Discriminative Score

```python
def discriminative_score(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    n_splits: int = 5,
) -> float:
    """
    Train a binary classifier to distinguish real vs. synthetic.
    Score = 0.5 → indistinguishable (perfect synthesis).
    Score → 1.0 → easily distinguished (poor synthesis).
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    n_real  = len(X_real)
    n_synth = len(X_synth)

    # Flatten time series for feature input
    X = np.vstack([X_real.reshape(n_real, -1), X_synth.reshape(n_synth, -1)])
    y = np.concatenate([np.ones(n_real), np.zeros(n_synth)])

    clf    = GradientBoostingClassifier(n_estimators=100)
    scores = cross_val_score(clf, X, y, cv=n_splits, scoring="roc_auc")
    disc   = float(scores.mean())

    print(f"Discriminative score: {disc:.4f} "
          f"({'good synthesis' if disc < 0.6 else 'poor synthesis'})")
    return disc
```

---

## 6. Privacy-Preserving Synthesis

```python
def dp_noise_synthesis(
    series: np.ndarray,
    epsilon: float = 1.0,
    sensitivity: float = None,
) -> np.ndarray:
    """
    Differentially-private TS generation via Laplace mechanism.

    Adds Laplace noise calibrated to ε-differential privacy.
    Provides mathematical privacy guarantee but degrades signal.

    Parameters
    ----------
    series      : original 1D time series
    epsilon     : privacy budget (smaller = more private, more noise)
    sensitivity : L1 sensitivity of the statistic (default = range)
    """
    if sensitivity is None:
        sensitivity = float(series.max() - series.min())

    scale  = sensitivity / epsilon
    noise  = np.random.laplace(0, scale, series.shape)
    return series + noise


def k_anonymize_ts(
    df: pd.DataFrame,
    entity_col: str,
    k: int = 5,
) -> pd.DataFrame:
    """
    k-anonymize a TS dataset by grouping entities into clusters of size ≥ k.
    Each group's time series is replaced by the group average.
    Prevents re-identification of individual entities.
    """
    from sklearn.cluster import KMeans

    entities = df[entity_col].unique()
    # Pivot to (entity, T) matrix
    pivot    = df.pivot(index=entity_col, columns="timestamp", values="value")

    n_clusters = max(1, len(entities) // k)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(pivot.fillna(0).values)

    result = df.copy()
    for cluster_id in np.unique(labels):
        cluster_entities = pivot.index[labels == cluster_id]
        avg_series = pivot.loc[cluster_entities].mean(axis=0)
        for entity in cluster_entities:
            mask = result[entity_col] == entity
            result.loc[mask, "value"] = avg_series.values[:mask.sum()]

    return result
```

---

## 7. Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def demo_augmentation():
    """Compare augmentation strategies on a synthetic dataset."""
    np.random.seed(42)
    T = 100
    t = np.linspace(0, 4*np.pi, T)
    original = np.sin(t) + 0.2 * np.random.normal(0, 1, T)

    augmented = {
        "Original":     original,
        "Jitter":       jitter(original, sigma=0.05),
        "Scaling":      scaling(original, sigma=0.2),
        "Time Warp":    time_warp(original, sigma=0.2),
        "Window Warp":  window_warp(original, window_ratio=0.1),
        "Magnitude Warp": magnitude_warp(original, sigma=0.2),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (name, series) in zip(axes.flatten(), augmented.items()):
        ax.plot(t, series, linewidth=1.5, color="#2196F3" if name == "Original" else "#FF5722")
        if name != "Original":
            ax.plot(t, original, linewidth=1.0, color="gray", alpha=0.5, linestyle="--",
                    label="Original")
            ax.legend(fontsize=8)
        ax.set_title(name, fontsize=11); ax.grid(alpha=0.3)

    plt.suptitle("Time Series Augmentation Strategies", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("ts_augmentation_strategies.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    demo_augmentation()
    # See code/03_ts_generation.py for full practical
```

---

*← [05 — Diffusion Models](./05_diffusion_models_for_ts.md) | [Module README](./README.md)*
