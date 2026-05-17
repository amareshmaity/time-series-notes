# 03 — Feature-Based Classification

> **Module**: 10 Classification & Clustering | **File**: 3 of 5
>
> Feature-based methods transform each time series into a fixed-length feature vector, then apply any tabular classifier. They are highly interpretable, work well with small datasets, and integrate seamlessly into existing ML pipelines. This note covers tsfresh, catch22, BOSS, and Shapelet Transform.

---

## Table of Contents

1. [The Feature-Based Paradigm](#1-the-feature-based-paradigm)
2. [tsfresh — Automated Feature Extraction](#2-tsfresh--automated-feature-extraction)
3. [catch22 — Canonical Features](#3-catch22--canonical-features)
4. [BOSS — Bag of SFA Symbols](#4-boss--bag-of-sfa-symbols)
5. [Shapelet Transform](#5-shapelet-transform)
6. [Time Series Forest (TSF)](#6-time-series-forest-tsf)
7. [Combining Features with ROCKET](#7-combining-features-with-rocket)
8. [Production Pipeline](#8-production-pipeline)

---

## 1. The Feature-Based Paradigm

### 1.1 Workflow

```
Input: N time series of length T

Step 1: Feature Extraction
  Each series xᵢ → feature vector fᵢ ∈ ℝᵈ
  Where features = statistics, spectral, structural properties

Step 2: Standard Classification
  (f₁, y₁), ..., (fₙ, yₙ) → train any tabular classifier
  (Random Forest, SVM, XGBoost, Logistic Regression)

Step 3: Inference
  new series x → extract features → classifier → label

Advantages:
  ✅ Interpretable: features have names (mean, variance, spectral_entropy)
  ✅ Any sklearn classifier works on top
  ✅ Feature importance → understand what drives classification
  ✅ Works well for small N (no deep learning sample requirements)
  ✅ Fast inference (feature extraction is O(T), classification is O(d))

Disadvantages:
  ❌ Hand-crafted features may miss important patterns
  ❌ Feature computation can be slow for T >> 1000
  ❌ May miss fine-grained temporal structure
```

### 1.2 Types of Features

| Feature Category   | Examples                                    | Captures                     |
|--------------------|---------------------------------------------|------------------------------|
| Statistical        | mean, std, skewness, kurtosis, percentiles  | Distribution shape           |
| Autocorrelation    | ACF[1], ACF[2], ..., partial ACF            | Temporal dependence           |
| Spectral           | FFT energy, spectral entropy, power bands   | Frequency content            |
| Nonlinear          | sample entropy, Hurst exponent, DFA         | Complexity, long-range memory |
| Structural         | peaks, valleys, zero-crossings, flatness    | Shape properties             |
| Wavelet            | DWT coefficients at multiple scales         | Multi-resolution structure   |

---

## 2. tsfresh — Automated Feature Extraction

### 2.1 Overview

**tsfresh** (Christ et al., 2018) automatically computes **~800 features** from a time series, covering most known statistical and spectral descriptors:

```
Default feature set:
  - 63 feature calculators → generates ~794 features per series
  - Includes: mean, std, max, min, skewness, kurtosis
  - ACF/PACF at multiple lags
  - FFT coefficients, spectral power, entropy
  - Linear trend parameters (slope, R²)
  - Complexity: approximate entropy, sample entropy
  - Count of peaks, valleys, crossings

Efficient computation:
  - Parallelized across CPUs (n_jobs parameter)
  - Supports early stopping and feature filtering
```

### 2.2 Implementation

```python
import pandas as pd
import numpy as np

def tsfresh_extract(
    X: np.ndarray,
    feature_set: str = "minimal",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Extract tsfresh features from a 2D time series array.

    Parameters
    ----------
    X           : (n_samples, T) array of time series
    feature_set : 'minimal' (fast), 'efficient' (default), 'comprehensive' (slow)
    n_jobs      : parallel workers

    Returns
    -------
    features_df : (n_samples, n_features) DataFrame
    """
    try:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
    except ImportError:
        print("Install tsfresh: pip install tsfresh")
        return pd.DataFrame()

    # tsfresh expects a "long" DataFrame with columns: id, time, value
    records = []
    for i, series in enumerate(X):
        for t, val in enumerate(series):
            records.append({"id": i, "time": t, "value": val})
    df_long = pd.DataFrame(records)

    fc_params = {
        "minimal":       MinimalFCParameters(),
        "efficient":     EfficientFCParameters(),
        "comprehensive": None,          # uses all default features
    }.get(feature_set, EfficientFCParameters())

    features = extract_features(
        df_long,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=fc_params,
        n_jobs=n_jobs,
        disable_progressbar=False,
    )
    return features


def tsfresh_select_features(
    features: pd.DataFrame,
    y: pd.Series,
    fdr_level: float = 0.05,
) -> pd.DataFrame:
    """
    Filter tsfresh features using Benjamini-Hochberg FDR control.

    Returns only statistically relevant features.
    """
    from tsfresh import select_features
    return select_features(features, y, fdr_level=fdr_level)


# Full tsfresh classification pipeline
def tsfresh_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_set: str = "efficient",
):
    """
    End-to-end tsfresh feature-based classification pipeline.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    print(f"Extracting tsfresh features ({feature_set})...")
    feat_train = tsfresh_extract(X_train, feature_set)
    feat_test  = tsfresh_extract(X_test, feature_set)

    # Select relevant features on training set
    y_s = pd.Series(y_train, index=feat_train.index)
    feat_train_sel = tsfresh_select_features(feat_train, y_s)
    selected_cols  = feat_train_sel.columns.tolist()
    feat_test_sel  = feat_test[selected_cols]

    print(f"Selected {len(selected_cols)} / {feat_train.shape[1]} features")

    # Classifier pipeline
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("rf",      RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
    ])
    clf.fit(feat_train_sel, y_train)
    y_pred = clf.predict(feat_test_sel)

    return clf, y_pred, selected_cols
```

### 2.3 Feature Importance from tsfresh

```python
def tsfresh_feature_importance(clf_pipeline, feature_names: list, top_k: int = 20):
    """Extract and display top features from a RF trained on tsfresh features."""
    rf      = clf_pipeline.named_steps["rf"]
    imp     = rf.feature_importances_
    idx     = np.argsort(imp)[::-1][:top_k]

    print(f"\nTop {top_k} tsfresh features:")
    for rank, i in enumerate(idx, 1):
        print(f"  {rank:2d}. {feature_names[i]:60s} : {imp[i]:.5f}")
```

---

## 3. catch22 — Canonical Features

### 3.1 Overview

**catch22** (Lubba et al., 2019) is a curated set of **22 canonical features** that were selected from ~7700 features across 93 datasets using greedy redundancy reduction:

```
catch22 features (22 total):
  - Linear autocorrelation at specific lags
  - Nonlinear autocorrelation features
  - Distribution features (outlier presence, extreme value proportions)
  - Information-theoretic features (first 1/e crossing, min/max spread)
  - Structural features (number of turning points, flat regions)

Advantages over tsfresh:
  ✅ Fast: O(T) computation for all 22 features
  ✅ Compact: 22 features → less overfitting risk
  ✅ Competitive accuracy on UCR benchmark

pip install pycatch22
```

```python
import numpy as np
import pandas as pd

def extract_catch22(X: np.ndarray) -> pd.DataFrame:
    """
    Extract catch22 features for all time series in X.

    X: (n_samples, T) array

    Returns
    -------
    (n_samples, 22) DataFrame
    """
    try:
        import pycatch22
    except ImportError:
        print("Install pycatch22: pip install pycatch22")
        return pd.DataFrame()

    all_feats = []
    for series in X:
        result = pycatch22.catch22_all(list(series))
        all_feats.append(dict(zip(result["names"], result["values"])))

    return pd.DataFrame(all_feats)


def catch22_pipeline(X_train, y_train, X_test):
    """catch22 + RandomForest classification pipeline."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    feat_train = extract_catch22(X_train)
    feat_test  = extract_catch22(X_test)

    scaler     = StandardScaler()
    X_tr_s     = scaler.fit_transform(feat_train.values)
    X_te_s     = scaler.transform(feat_test.values)

    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    clf.fit(X_tr_s, y_train)

    return clf.predict(X_te_s)
```

---

## 4. BOSS — Bag of SFA Symbols

### 4.1 Algorithm

**BOSS** (Schäfer, 2015) converts each series into a **bag of words** using Symbolic Fourier Approximation (SFA):

```
Algorithm:
  1. Divide series into sliding windows of length w
  2. For each window:
       a. Apply Discrete Fourier Transform (DFT) → Fourier coefficients
       b. Apply Multiple Coefficient Binning (MCB) → discretize to symbols [a,b,c,d]
  3. Build histogram of word occurrences (bag of words)
  4. Classify with nearest-neighbor using BOSS distance

BOSS distance:
  d_BOSS(X, Y) = Σ (bₓ(w) - bᵧ(w))² where bₓ(w) > 0 or bᵧ(w) > 0

Key parameters:
  - window length w
  - word length l (DFT coefficients to keep)
  - alphabet size a (symbols per bin)

Advantage: very robust to noise (discretization smooths noise)
           Fast: O(T) per series after training
```

```python
def boss_pipeline(X_train, y_train, X_test):
    """
    BOSS classifier via sktime.
    """
    try:
        from sktime.classification.dictionary_based import BOSSEnsemble
        clf = BOSSEnsemble(max_ensemble_size=50, random_state=42, n_jobs=-1)
        # sktime expects (n_samples, n_channels, n_timepoints)
        Xtr = X_train[:, np.newaxis, :] if X_train.ndim == 2 else X_train
        Xte = X_test[:, np.newaxis, :]  if X_test.ndim == 2  else X_test
        clf.fit(Xtr, y_train)
        return clf.predict(Xte)
    except ImportError:
        print("Install sktime: pip install sktime")
        return None
```

---

## 5. Shapelet Transform

### 5.1 What Are Shapelets?

```
A shapelet is a time series subsequence that is maximally discriminative
between classes.

Example:
  Class 0 (normal ECG): always has a flat segment of length 20 at position 30–50
  Class 1 (arrhythmia): this segment always varies/spikes

Shapelet = [0.1, 0.1, 0.1, ..., 0.1] (flat segment)
  Smin(q, s) = min over all positions p of d(q[p:p+|s|], s)

  Class 0: Smin ≈ 0  (flat region found → close to shapelet)
  Class 1: Smin >> 0 (no flat region → far from shapelet)

Feature vector for series q:
  f(q) = [Smin(q, s₁), Smin(q, s₂), ..., Smin(q, sₖ)]
       = minimum distances to k selected shapelets

These features are fed to any classifier (typically Random Forest or Decision Tree).
```

### 5.2 Shapelet Discovery

```
Optimal shapelet selection:
  For each candidate subsequence s (from training data):
    Compute quality measure (e.g., information gain, F-statistic)
    → Select top-k shapelets with highest quality

Computational challenge:
  Brute-force: O(N²·T³) — infeasible for large datasets
  
Optimizations:
  - Shapelet caching
  - Early termination with LB pruning
  - Random shapelet sampling (RT-SHAP)
  - Learned shapelets (LTS) with gradient descent
```

### 5.3 Implementation

```python
def shapelet_pipeline(X_train, y_train, X_test, n_shapelets: int = 200):
    """
    Shapelet Transform classifier via sktime.
    """
    try:
        from sktime.classification.shapelet_based import ShapeletTransformClassifier
        from sklearn.ensemble import RandomForestClassifier

        clf = ShapeletTransformClassifier(
            estimator=RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
            time_contract_in_mins=5,   # search for 5 minutes
            random_state=42,
        )
        Xtr = X_train[:, np.newaxis, :] if X_train.ndim == 2 else X_train
        Xte = X_test[:, np.newaxis, :]  if X_test.ndim == 2  else X_test
        clf.fit(Xtr, y_train)
        return clf.predict(Xte)
    except ImportError:
        print("Install sktime: pip install sktime")
        return None


def manual_shapelet_features(
    X: np.ndarray,
    shapelets: list,
) -> np.ndarray:
    """
    Compute shapelet-distance feature matrix from pre-selected shapelets.

    Parameters
    ----------
    X         : (n_samples, T) array
    shapelets : list of 1D shapelet arrays

    Returns
    -------
    F : (n_samples, len(shapelets)) feature matrix
    """
    n   = len(X)
    k   = len(shapelets)
    F   = np.zeros((n, k))

    for j, s in enumerate(shapelets):
        L = len(s)
        for i in range(n):
            # Minimum subsequence distance
            dists = [np.sqrt(((X[i, p:p+L] - s)**2).mean())
                     for p in range(len(X[i]) - L + 1)]
            F[i, j] = min(dists) if dists else 0.0

    return F
```

---

## 6. Time Series Forest (TSF)

### 6.1 Algorithm

**Time Series Forest** (Deng et al., 2013) — an ensemble that extracts features from **random intervals** of the series:

```
For each tree in the ensemble:
  1. Sample r random intervals [start, end] from the series
  2. For each interval: compute [mean, std, slope]
  3. Build a decision tree on these 3r features
  4. Ensemble = vote across all trees

Advantages:
  - O(T·log(T)) training (much faster than shapelet)
  - Interpretable: each split reveals important time intervals
  - Robust: random intervals cover different parts of the series

Default parameters: 200 trees, √T intervals per tree
```

```python
def tsf_pipeline(X_train, y_train, X_test):
    """Time Series Forest classifier via sktime."""
    try:
        from sktime.classification.interval_based import TimeSeriesForestClassifier
        clf = TimeSeriesForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        Xtr = X_train[:, np.newaxis, :] if X_train.ndim == 2 else X_train
        Xte = X_test[:, np.newaxis, :]  if X_test.ndim == 2  else X_test
        clf.fit(Xtr, y_train)
        return clf.predict(Xte)
    except ImportError:
        print("Install sktime: pip install sktime")
        return None
```

---

## 7. Combining Features with ROCKET

### 7.1 ROCKET Feature Extraction

**ROCKET** (Random Convolutional Kernel Transform, Dempster et al., 2020) generates 10,000 random convolutional kernels and extracts two features per kernel (max pooling + proportion of positive values):

```
Kernel hyperparameters (all random):
  - Length: sampled from {7, 9, 11}
  - Weights: sampled from N(0,1)
  - Dilation: sampled exponentially: 2^e, e ~ U[0, log₂(T)]
  - Padding: random 0 or 1

Features per kernel (2 features):
  - max(PPV): global max of kernel activations → presence of pattern
  - PPV: Proportion of Positive Values → frequency of pattern

Total: 10,000 kernels × 2 = 20,000 features

Why it works:
  Random kernels span a huge variety of patterns and scales.
  A Ridge classifier (L2 regularized linear) selects the relevant subset.
```

```python
import numpy as np

def rocket_pipeline(X_train, y_train, X_test, n_kernels=10_000):
    """
    ROCKET + RidgeClassifierCV — the recommended fast TSC pipeline.

    Near-state-of-the-art accuracy with training time measured in seconds.
    """
    try:
        from sktime.transformations.panel.rocket import Rocket
        from sklearn.linear_model import RidgeClassifierCV
        from sklearn.pipeline import Pipeline

        rocket = Rocket(num_kernels=n_kernels, random_state=42, n_jobs=-1)
        clf    = Pipeline([
            ("rocket", rocket),
            ("ridge",  RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),
                                          normalize=True)),
        ])

        Xtr = X_train[:, np.newaxis, :] if X_train.ndim == 2 else X_train
        Xte = X_test[:, np.newaxis, :]  if X_test.ndim == 2  else X_test
        clf.fit(Xtr, y_train)
        return clf.predict(Xte), clf
    except ImportError:
        print("Install sktime: pip install sktime")
        return None, None
```

---

## 8. Production Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

class FeatureBasedTSClassifier:
    """
    Production feature-based time series classifier.
    Uses catch22 (fast) + custom statistics → Random Forest.
    Falls back gracefully if pycatch22 is not installed.
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._fitted      = False

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract feature matrix from (n_samples, T) array."""
        feats = []
        for s in X:
            row = {
                "mean":       float(s.mean()),
                "std":        float(s.std()),
                "min":        float(s.min()),
                "max":        float(s.max()),
                "range":      float(s.max() - s.min()),
                "skewness":   float(pd.Series(s).skew()),
                "kurtosis":   float(pd.Series(s).kurt()),
                "acf_1":      float(pd.Series(s).autocorr(1) or 0.0),
                "acf_5":      float(pd.Series(s).autocorr(5) or 0.0),
                "n_peaks":    int((np.diff(np.sign(np.diff(s))) < 0).sum()),
                "rms":        float(np.sqrt((s**2).mean())),
                "energy":     float((s**2).sum()),
                "zcr":        float((np.diff(np.sign(s)) != 0).mean()),
                "q25":        float(np.percentile(s, 25)),
                "q75":        float(np.percentile(s, 75)),
                "iqr":        float(np.percentile(s, 75) - np.percentile(s, 25)),
            }
            feats.append(row)

        # Try to add catch22 if available
        try:
            import pycatch22
            for i, s in enumerate(X):
                result = pycatch22.catch22_all(list(s))
                for name, val in zip(result["names"], result["values"]):
                    feats[i][f"c22_{name}"] = float(val) if np.isfinite(float(val)) else 0.0
        except ImportError:
            pass

        return pd.DataFrame(feats).values

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureBasedTSClassifier":
        F = self._extract_features(X)
        self._pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     RandomForestClassifier(n_estimators=self.n_estimators,
                                                n_jobs=-1, random_state=self.random_state)),
        ])
        self._pipeline.fit(F, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call .fit() first."
        F = self._extract_features(X)
        return self._pipeline.predict(F)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call .fit() first."
        F = self._extract_features(X)
        return self._pipeline.predict_proba(F)
```

---

*← [02 — DTW](./02_distance_based_methods_DTW.md) | [Module README](./README.md) | Next: [04 — Deep Learning Classification](./04_deep_learning_classification.md) →*
