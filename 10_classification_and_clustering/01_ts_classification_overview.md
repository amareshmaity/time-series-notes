# 01 — Time Series Classification Overview

> **Module**: 10 Classification & Clustering | **File**: 1 of 5
>
> Time series classification (TSC) asks: given a sequence, which class does it belong to? From ECG diagnosis to activity recognition to financial pattern detection, TSC spans every domain. This note covers problem framing, the benchmark landscape, evaluation methodology, and how to choose the right algorithm.

---

## Table of Contents

1. [Problem Framing](#1-problem-framing)
2. [Classification vs. Forecasting vs. Anomaly Detection](#2-classification-vs-forecasting-vs-anomaly-detection)
3. [UCR and UEA Archives](#3-ucr-and-uea-archives)
4. [Algorithm Landscape](#4-algorithm-landscape)
5. [Evaluation Methodology](#5-evaluation-methodology)
6. [Data Preparation and Alignment](#6-data-preparation-and-alignment)
7. [Choosing the Right Classifier](#7-choosing-the-right-classifier)
8. [Implementation Boilerplate](#8-implementation-boilerplate)

---

## 1. Problem Framing

### 1.1 Formal Definition

```
Given:
  A dataset D = {(X₁, y₁), ..., (Xₙ, yₙ)}
  Where:
    Xᵢ ∈ ℝᵀ — a time series of length T (univariate)
    yᵢ ∈ {1, ..., C} — a class label

Goal:
  Learn f: ℝᵀ → {1, ..., C} that minimises classification error on unseen series.

Key properties:
  - Each sample is an entire sequence (vs. a single row of tabular features)
  - Temporal ordering matters — shuffling destroys information
  - Variable-length sequences require special handling
  - Labels are assigned per sequence, not per time step
```

### 1.2 Variants

| Variant                        | Description                                        | Example                             |
|--------------------------------|----------------------------------------------------|-------------------------------------|
| **Univariate TSC**             | Single channel per sample                          | ECG beat → normal / arrhythmia      |
| **Multivariate TSC**           | D channels per sample                              | IMU (acc_x, acc_y, acc_z) → gesture |
| **Early classification**       | Classify before full series observed               | Fault detection as early as possible|
| **Streaming classification**   | Classify incrementally as data arrives             | Real-time activity recognition      |
| **Unequal-length series**      | Samples have different lengths T_i                 | Variable-duration audio clips        |

### 1.3 Key Challenges

```
1. High dimensionality: T can range from 50 to 10,000+
   → Feature selection or dimensionality reduction needed

2. Temporal alignment: two instances of the same pattern may be
   time-shifted or speed-varied → DTW or elastic distances

3. Small datasets: UCR archive has datasets as small as 16 training samples
   → Few-shot learning, data augmentation, transfer learning

4. Class imbalance: common in fault detection, medical diagnosis
   → SMOTE-TS, weighted loss functions

5. Interpretability: "Why did the model classify this ECG as anomalous?"
   → Shapelet-based methods, attention mechanisms
```

---

## 2. Classification vs. Forecasting vs. Anomaly Detection

```
Problem Type       | Input            | Output           | Key Algorithm
───────────────────────────────────────────────────────────────────────
Forecasting        | Series X[1..t]   | X[t+1..t+H]      | ARIMA, LSTM, TFT
Classification     | Full series X     | Label y ∈ {1..C} | ROCKET, DTW-kNN
Anomaly Detection  | Series X          | Anomaly flag at t | Isolation Forest, AE
Segmentation       | Series X          | Changepoint at t  | PELT, BOCPD

Classification maps the WHOLE sequence to one label.
Segmentation maps each point to a segment label.
The two are complementary (segment first → classify each segment).
```

---

## 3. UCR and UEA Archives

### 3.1 UCR Archive (Univariate)

The UCR Time Series Classification Archive (Dau et al., 2019) contains **128 datasets** and is the standard benchmark for evaluating TSC algorithms:

```
Key properties of UCR datasets:
  - Pre-split into train and test sets
  - Variable T (length), C (classes), N (samples)
  - Covers: ECG, sensor, image outline, motion, simulated

Example datasets:
  - ECG200:      N=100 train, T=96, C=2 (normal / myocardial infarction)
  - GunPoint:    N=50 train, T=150, C=2 (point gun or draw)
  - FordA:       N=3601 train, T=500, C=2 (engine fault classification)
  - ElectricDevices: N=8926 train, T=96, C=7

Standard evaluation:
  - Accuracy on the fixed test split
  - Critical Difference (CD) diagram to rank algorithms across datasets
```

### 3.2 UEA Archive (Multivariate)

The UEA Multivariate Time Series Archive (Bagnall et al., 2018) contains **30 multivariate datasets**:

```
Example datasets:
  - BasicMotions:  N=40 train, T=100, D=6 channels, C=4 (walking/standing/etc.)
  - EigenWorms:    N=128 train, T=17984, D=6, C=5 (C. elegans behavior)
  - InsectWingbeat: N=25000 train, T=30, D=200 channels, C=10
```

### 3.3 Loading UCR Datasets

```python
from sktime.datasets import load_UCR_UEA_dataset
import numpy as np

def load_ucr_dataset(name: str):
    """
    Load a UCR/UEA dataset via sktime.

    Returns
    -------
    X_train, y_train, X_test, y_test
    Each X is a pd.DataFrame with nested series per cell (sktime format).
    Use sktime's transformers to convert to numpy arrays.
    """
    X_train, y_train = load_UCR_UEA_dataset(name, split="train", return_X_y=True)
    X_test,  y_test  = load_UCR_UEA_dataset(name, split="test",  return_X_y=True)
    return X_train, y_train, X_test, y_test


def to_numpy_3d(X_sktime) -> np.ndarray:
    """
    Convert sktime nested DataFrame to (n_samples, n_channels, n_timepoints) numpy array.
    """
    from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
    return from_nested_to_3d_numpy(X_sktime)


# Example
# X_train, y_train, X_test, y_test = load_ucr_dataset("GunPoint")
# X_train_np = to_numpy_3d(X_train)   # (50, 1, 150)
```

---

## 4. Algorithm Landscape

### 4.1 Taxonomy

```
Time Series Classification Algorithms
├── Distance-Based
│   ├── 1-NN + Euclidean
│   ├── 1-NN + DTW (with/without warping window)
│   ├── kNN + LCSS / EDR / ERP
│   └── kNN + WDTW (weighted DTW)
│
├── Feature-Based
│   ├── tsfresh (700+ statistical features)
│   ├── catch22 (22 canonical features)
│   ├── BOSS (Bag of SFA Symbols)
│   └── Shapelet Transform → any classifier
│
├── Interval-Based
│   ├── Time Series Forest (TSF)
│   ├── RISE (Random Interval Spectral Ensemble)
│   └── DrCIF (Diverse Representation CIF)
│
├── Convolutional
│   ├── ROCKET (10,000 random kernels)
│   ├── MiniRocket (much faster, minimal random)
│   ├── MultiRocket (multivariate)
│   └── Hydra (competing kernel groups)
│
└── Deep Learning
    ├── FCN (Fully Convolutional Network)
    ├── ResNet-TS
    ├── InceptionTime (ensemble of Inception)
    └── LSTM / Transformer classifiers
```

### 4.2 Accuracy vs. Speed Trade-off

```
Highest Accuracy (2024):
  InceptionTime / HIVE-COTE 2.0 ≈ 88–90% average accuracy across UCR archive

Best Speed-Accuracy Trade-off:
  ROCKET / MiniRocket — near-state-of-the-art accuracy in seconds

Interpretable:
  Shapelet Transform, BOSS, Time Series Forest

Best for small datasets (N < 100):
  1-NN DTW — no parameters, surprisingly competitive

Best for multivariate:
  MultiRocket, InceptionTime, HIVECOTE with multivariate extension
```

---

## 5. Evaluation Methodology

### 5.1 Standard Metrics

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)

def evaluate_classifier(y_true, y_pred, y_proba=None, class_names=None):
    """
    Comprehensive classifier evaluation for time series classification.

    Parameters
    ----------
    y_true      : true class labels
    y_pred      : predicted class labels
    y_proba     : predicted probabilities (n_samples, n_classes) — for AUC
    class_names : list of class names for reporting
    """
    acc  = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, average="macro")
    f1_w = f1_score(y_true, y_pred, average="weighted")

    print(f"Accuracy:          {acc:.4f}")
    print(f"Macro F1:          {f1_m:.4f}")
    print(f"Weighted F1:       {f1_w:.4f}")

    if y_proba is not None and len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
        print(f"AUC-ROC:           {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return {"accuracy": acc, "macro_f1": f1_m, "weighted_f1": f1_w}
```

### 5.2 Critical Difference Diagrams

The standard way to compare multiple classifiers across many datasets is the **Critical Difference (CD) diagram** (Demšar, 2006):

```
1. Rank each algorithm on each dataset (best = 1, worst = k)
2. Average ranks across all datasets
3. Apply Nemenyi post-hoc test:
   CD = q_α · √(k(k+1) / (6·N))
   where k = number of algorithms, N = number of datasets
4. Plot average ranks; connect algorithms within CD of each other
   (connected = not significantly different)

In Python: use scikit-posthocs or Orange library
```

### 5.3 Stratified Cross-Validation for Small Datasets

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

def ts_cross_val(classifier, X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """
    Stratified k-fold CV for time series classification.

    X: (n_samples, T) or (n_samples, channels, T)
    y: (n_samples,) class labels

    Note: does NOT shuffle — preserves temporal ordering within each fold's test set.
    For independent samples (e.g., separate ECG beats), shuffling is acceptable.
    """
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=False)
    scores = cross_val_score(classifier, X, y, cv=skf, scoring="accuracy")
    print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores
```

---

## 6. Data Preparation and Alignment

### 6.1 Normalization

```python
import numpy as np

def z_normalize_series(X: np.ndarray) -> np.ndarray:
    """
    Z-normalize each time series independently (standard for TSC).

    X: (n_samples, T) or (n_samples, channels, T)

    Z-normalization is REQUIRED for:
      - DTW distance computation (otherwise longer/larger series dominate)
      - ROCKET features (kernel correlation depends on scale)
      - Most deep learning classifiers

    Returns normalized X with zero mean and unit variance per series.
    """
    if X.ndim == 2:
        mu    = X.mean(axis=1, keepdims=True)
        sigma = X.std(axis=1, keepdims=True) + 1e-8
        return (X - mu) / sigma
    elif X.ndim == 3:
        mu    = X.mean(axis=2, keepdims=True)
        sigma = X.std(axis=2, keepdims=True) + 1e-8
        return (X - mu) / sigma
    raise ValueError(f"Expected 2D or 3D array, got {X.ndim}D")


def pad_to_equal_length(series_list: list, pad_value: float = 0.0) -> np.ndarray:
    """
    Pad variable-length series to the longest length.

    Parameters
    ----------
    series_list : list of 1D arrays with different lengths
    pad_value   : padding value (0.0 or np.nan)

    Returns
    -------
    X: (n_samples, max_T) padded array
    """
    T_max = max(len(s) for s in series_list)
    X     = np.full((len(series_list), T_max), pad_value, dtype=np.float32)
    for i, s in enumerate(series_list):
        X[i, :len(s)] = s
    return X
```

### 6.2 Data Augmentation for Small Datasets

```python
def augment_ts(X: np.ndarray, y: np.ndarray, n_augment: int = 3) -> tuple:
    """
    Simple time series augmentation: jitter + scaling + time warping.

    Applied to training set only to increase dataset size.
    """
    X_aug, y_aug = [X], [y]

    for _ in range(n_augment):
        # Jitter: add Gaussian noise
        jitter = X + np.random.normal(0, 0.05 * X.std(), X.shape)
        X_aug.append(jitter); y_aug.append(y)

        # Scaling: random amplitude scaling
        scale  = np.random.uniform(0.8, 1.2, (len(X), 1))
        scaled = X * scale
        X_aug.append(scaled); y_aug.append(y)

        # Window slicing: random crop and resize
        T = X.shape[1]
        crop_len = int(T * np.random.uniform(0.8, 1.0))
        start    = np.random.randint(0, T - crop_len + 1)
        cropped  = X[:, start:start + crop_len]
        from scipy.interpolate import interp1d
        interp   = interp1d(np.linspace(0, 1, crop_len), cropped, axis=1)
        resized  = interp(np.linspace(0, 1, T))
        X_aug.append(resized); y_aug.append(y)

    return np.vstack(X_aug), np.concatenate(y_aug)
```

---

## 7. Choosing the Right Classifier

```
Decision Framework:

Step 1: Dataset size?
  N < 100:  1-NN DTW or Shapelet Transform (no parameters needed)
  N ≥ 100:  proceed

Step 2: Speed requirement?
  Fast (< 1 minute):  ROCKET / MiniRocket
  Moderate:            tsfresh + RandomForest, BOSS
  Slow allowed:        HIVE-COTE 2.0, InceptionTime ensemble

Step 3: Interpretability needed?
  Yes: Shapelet Transform (shows discriminative subsequences)
       BOSS (histogram of symbolic words)
  No:  ROCKET, InceptionTime

Step 4: Multivariate?
  Yes: MultiRocket, WEASEL-2.0, InceptionTime
  No:  All of the above

Step 5: Variable length?
  Yes: DTW (handles naturally), GAP (global average pooling DL)
  No:  ROCKET (requires fixed length), standard DL

Default recommendation:
  Start with ROCKET/MiniRocket → benchmark → if insufficient, try InceptionTime
```

---

## 8. Implementation Boilerplate

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeClassifierCV

# Standard ROCKET pipeline (works with sktime)
def build_rocket_pipeline():
    """
    Standard ROCKET classification pipeline:
    ROCKET features → Ridge Classifier (fast, state-of-the-art).
    """
    try:
        from sktime.transformations.panel.rocket import Rocket
        return Pipeline([
            ("rocket", Rocket(num_kernels=10_000, random_state=42)),
            ("clf",    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))),
        ])
    except ImportError:
        print("Install sktime: pip install sktime")
        return None


# Standard DTW-kNN pipeline (works with tslearn)
def build_dtw_knn_pipeline(n_neighbors=1, window=0.1):
    """
    DTW k-NN pipeline:
    Z-normalize → kNN with DTW distance.
    """
    try:
        from tslearn.neighbors import KNeighborsTimeSeriesClassifier
        return KNeighborsTimeSeriesClassifier(
            n_neighbors=n_neighbors,
            metric="dtw",
            metric_params={"global_constraint": "sakoe_chiba",
                           "sakoe_chiba_radius": window},
        )
    except ImportError:
        print("Install tslearn: pip install tslearn")
        return None


if __name__ == "__main__":
    # Quick demo with synthetic data
    np.random.seed(42)
    N, T, C = 100, 50, 3
    X = np.random.randn(N, T)
    y = np.random.choice(C, N)

    # Z-normalize
    X = z_normalize_series(X)
    print(f"Dataset: {N} samples, T={T}, C={C} classes")
    print(f"X shape: {X.shape}, normalized mean≈{X.mean():.3f}, std≈{X.std():.3f}")
```

---

*← [Module README](./README.md) | Next: [02 — DTW](./02_distance_based_methods_DTW.md) →*
