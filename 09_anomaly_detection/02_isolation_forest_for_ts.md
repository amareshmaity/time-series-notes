# 02 — Isolation Forest for Time Series

> **Module**: 09 Anomaly Detection | **File**: 2 of 6
>
> Isolation Forest is the industry-standard unsupervised anomaly detector for tabular data. For time series, the key challenge is **feature engineering**: transforming the raw signal into meaningful features that expose anomalous structure. This note also covers One-Class SVM and Local Outlier Factor adapted for temporal data.

---

## Table of Contents

1. [Isolation Forest — Core Idea](#1-isolation-forest--core-idea)
2. [Feature Engineering for Time Series](#2-feature-engineering-for-time-series)
3. [Isolation Forest Implementation](#3-isolation-forest-implementation)
4. [One-Class SVM](#4-one-class-svm)
5. [Local Outlier Factor (LOF)](#5-local-outlier-factor-lof)
6. [Threshold Selection and Scoring](#6-threshold-selection-and-scoring)
7. [Ensemble Detector](#7-ensemble-detector)
8. [Production Pipeline](#8-production-pipeline)

---

## 1. Isolation Forest — Core Idea

### 1.1 Intuition

```
Key Insight: Anomalies are RARE and DIFFERENT from normal points.
→ They are EASIER to isolate (separate) using random splits.

Algorithm:
  1. Randomly select a feature
  2. Randomly select a split value between [min, max] of that feature
  3. Repeat recursively → builds a random isolation tree
  4. Depth at which a point is isolated = anomaly score
     (shallow depth → point isolated quickly → likely anomaly)

Anomaly score ∈ (0, 1):
  → 1.0: almost certainly an anomaly
  → 0.5: indistinguishable from random data
  → < 0.5: almost certainly normal

Key advantage: Does NOT require labeled anomaly data (fully unsupervised)
```

### 1.2 Mathematical Formulation

```
For a point x, its isolation score is:

s(x, n) = 2^{-E[h(x)] / c(n)}

Where:
  h(x)  = path length from root to leaf for point x
  E[h]  = expected path length averaged over all trees
  c(n)  = average path length of unsuccessful BST search
          = 2·H(n-1) - 2(n-1)/n
          H(n) = harmonic number ≈ ln(n) + 0.5772

s close to 1 → anomaly (short path)
s close to 0 → normal (long path, hard to isolate)
```

### 1.3 Contamination Parameter

```
contamination: expected fraction of anomalies in the dataset
  → Controls the decision threshold (not the model itself)
  → Default: 0.1 (10% assumed anomalies)
  → In practice: start with 0.01–0.05 for most monitoring scenarios

"auto": sklearn uses the theoretical score 0.5 as threshold
```

---

## 2. Feature Engineering for Time Series

### 2.1 Why Features Matter

Raw time series values are 1-dimensional. Isolation Forest works best with informative multivariate features that capture temporal context:

```
Raw approach:
  Feature = [xₜ]  ← only current value
  → Misses contextual anomalies (e.g., normal value at wrong time of day)

Feature-engineered approach:
  Feature = [xₜ, lag₁, lag₂, rolling_mean, rolling_std, hour_of_day, ...]
  → Contextual anomalies become visible in feature space
```

### 2.2 Feature Categories

| Feature Type        | Examples                                    | Detects                    |
|---------------------|---------------------------------------------|----------------------------|
| Lag features        | xₜ₋₁, xₜ₋₂, xₜ₋ₛ                        | Point anomalies             |
| Rolling statistics  | μ_W, σ_W, skew_W, kurtosis_W               | Level/variance shifts       |
| Difference features | Δxₜ = xₜ - xₜ₋₁, Δᵢₙₜₑᵣ = xₜ - xₜ₋ₛ    | Sudden changes              |
| Calendar features   | hour, day_of_week, month, is_holiday        | Contextual anomalies        |
| Spectral features   | FFT amplitude at key frequencies            | Cyclical pattern breaks     |

### 2.3 Feature Engineering Function

```python
import numpy as np
import pandas as pd

def build_anomaly_features(
    series: pd.Series,
    lags: list = None,
    rolling_windows: list = None,
    include_calendar: bool = False,
) -> pd.DataFrame:
    """
    Build a feature matrix for time series anomaly detection.

    Parameters
    ----------
    series          : pd.Series with DatetimeIndex
    lags            : lag periods to include (e.g., [1, 2, 7, 14])
    rolling_windows : rolling statistic window sizes (e.g., [7, 30])
    include_calendar: whether to add calendar features (requires DatetimeIndex)

    Returns
    -------
    DataFrame with one row per time step, columns = features
    """
    if lags is None:
        lags = [1, 2, 3, 7]
    if rolling_windows is None:
        rolling_windows = [7, 30]

    feat = pd.DataFrame(index=series.index)
    feat["value"] = series.values

    # Lag features
    for lag in lags:
        feat[f"lag_{lag}"] = series.shift(lag).values

    # Difference features
    feat["diff_1"] = series.diff(1).values
    if max(lags) >= 7:
        feat["diff_7"] = series.diff(7).values

    # Rolling statistics
    for w in rolling_windows:
        rol = series.rolling(w, min_periods=w // 2)
        feat[f"rolling_mean_{w}"] = rol.mean().values
        feat[f"rolling_std_{w}"]  = rol.std().values
        feat[f"rolling_z_{w}"]    = ((series - rol.mean()) / (rol.std() + 1e-12)).values
        feat[f"rolling_max_{w}"]  = rol.max().values
        feat[f"rolling_min_{w}"]  = rol.min().values

    # Calendar features
    if include_calendar and hasattr(series.index, 'hour'):
        idx = series.index
        feat["hour"]        = idx.hour if hasattr(idx, 'hour') else 0
        feat["day_of_week"] = idx.dayofweek if hasattr(idx, 'dayofweek') else 0
        feat["month"]       = idx.month if hasattr(idx, 'month') else 0
        feat["is_weekend"]  = (feat["day_of_week"] >= 5).astype(int)

    return feat.dropna()
```

---

## 3. Isolation Forest Implementation

### 3.1 Core Implementation

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def isolation_forest_detector(
    series: pd.Series,
    lags: list = None,
    rolling_windows: list = None,
    n_estimators: int = 200,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Isolation Forest anomaly detector for time series.

    Parameters
    ----------
    series          : pd.Series with optional DatetimeIndex
    lags            : lag periods for feature engineering
    rolling_windows : rolling window sizes
    n_estimators    : number of isolation trees
    contamination   : expected fraction of anomalies
    random_state    : reproducibility seed

    Returns
    -------
    DataFrame with anomaly scores and flags aligned to original index
    """
    # Build features
    features = build_anomaly_features(series, lags, rolling_windows)
    aligned_series = series.loc[features.index]

    # Scale features (Isolation Forest is scale-invariant but scaling helps LOF/SVM)
    scaler = StandardScaler()
    X      = scaler.fit_transform(features.values)

    # Fit and predict
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X)

    scores = iso.decision_function(X)   # higher = more normal
    labels = iso.predict(X)             # -1 = anomaly, +1 = normal

    return pd.DataFrame({
        "value":         aligned_series.values,
        "anomaly_score": -scores,          # flip sign → higher = more anomalous
        "anomaly":       labels == -1,
    }, index=features.index)


def isolation_forest_batch(
    train_series: pd.Series,
    test_series: pd.Series,
    **kwargs,
) -> pd.DataFrame:
    """
    Fit on training data, score on test data.
    Proper production pattern: train on known-normal data.
    """
    lags    = kwargs.pop("lags", [1, 2, 3, 7])
    windows = kwargs.pop("rolling_windows", [7, 30])

    feat_train = build_anomaly_features(train_series, lags, windows)
    feat_test  = build_anomaly_features(test_series,  lags, windows)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(feat_train.values)
    X_test  = scaler.transform(feat_test.values)

    iso = IsolationForest(**kwargs)
    iso.fit(X_train)

    scores = iso.decision_function(X_test)
    labels = iso.predict(X_test)

    return pd.DataFrame({
        "value":         test_series.loc[feat_test.index].values,
        "anomaly_score": -scores,
        "anomaly":       labels == -1,
    }, index=feat_test.index)
```

### 3.2 Hyperparameter Guide

| Parameter       | Effect                                              | Recommended Range  |
|-----------------|-----------------------------------------------------|--------------------|
| `n_estimators`  | More trees → stable scores (diminishing returns)   | 100–500            |
| `max_samples`   | Subsampling size per tree                           | 256 (default)      |
| `contamination` | Controls decision threshold only (not the model)   | 0.01–0.10          |
| `max_features`  | Feature subsampling per split                       | 1.0 (all features) |

---

## 4. One-Class SVM

### 4.1 Concept

One-Class SVM (Schölkopf et al., 1999) learns a tight boundary around the normal data in kernel space:

```
Objective: find a hyperplane that separates the origin from the data
           with maximum margin ν (nu)

Kernel trick: data mapped to high-dimensional RKHS where a tight sphere
              is fit around normal data

nu (ν) parameter:
  - Upper bound on the fraction of training errors (false positives)
  - Lower bound on the fraction of support vectors
  - Typically: ν ≈ expected contamination fraction
```

```python
from sklearn.svm import OneClassSVM

def ocsvm_detector(
    train_series: pd.Series,
    test_series: pd.Series,
    nu: float = 0.05,
    kernel: str = "rbf",
    gamma: str = "scale",
    lags: list = None,
    rolling_windows: list = None,
) -> pd.DataFrame:
    """
    One-Class SVM detector for time series anomalies.

    Parameters
    ----------
    nu     : upper bound on false positive rate (≈ contamination)
    kernel : 'rbf' (nonlinear), 'linear', or 'poly'
    gamma  : RBF kernel bandwidth ('scale' = 1/(n_features * X.var()))
    """
    lags    = lags or [1, 2, 3, 7]
    windows = rolling_windows or [7, 30]

    feat_tr = build_anomaly_features(train_series, lags, windows)
    feat_te = build_anomaly_features(test_series, lags, windows)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(feat_tr.values)
    X_test  = scaler.transform(feat_te.values)

    ocsvm   = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    ocsvm.fit(X_train)

    scores  = ocsvm.decision_function(X_test)   # higher = more normal
    labels  = ocsvm.predict(X_test)             # -1 = anomaly, +1 = normal

    return pd.DataFrame({
        "value":         test_series.loc[feat_te.index].values,
        "anomaly_score": -scores,
        "anomaly":       labels == -1,
    }, index=feat_te.index)
```

---

## 5. Local Outlier Factor (LOF)

### 5.1 Concept

LOF compares the **local density** of a point to the density of its k-nearest neighbors. An anomaly is a point that is in a lower-density region than its neighbors:

```
LOF_k(x) = (1/|Nₖ(x)|) · Σ_{o ∈ Nₖ(x)} [lrd_k(o) / lrd_k(x)]

Where:
  Nₖ(x)   = k-nearest neighbors of x
  lrd_k(x) = local reachability density of x
             = 1 / (average reachability distance to Nₖ)

LOF ≈ 1 → density similar to neighbors (normal)
LOF > 1 → lower density than neighbors (potential anomaly)
LOF >> 1 → much lower density → strong anomaly
```

```python
from sklearn.neighbors import LocalOutlierFactor

def lof_detector(
    train_series: pd.Series,
    test_series: pd.Series,
    n_neighbors: int = 20,
    contamination: float = 0.05,
    lags: list = None,
    rolling_windows: list = None,
) -> pd.DataFrame:
    """
    LOF detector for time series. Uses novelty=True for train/test split.

    Parameters
    ----------
    n_neighbors   : neighborhood size k (larger → more global)
    contamination : expected anomaly fraction (sets threshold)
    """
    lags    = lags or [1, 2, 3, 7]
    windows = rolling_windows or [7, 30]

    feat_tr = build_anomaly_features(train_series, lags, windows)
    feat_te = build_anomaly_features(test_series, lags, windows)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(feat_tr.values)
    X_test  = scaler.transform(feat_te.values)

    lof     = LocalOutlierFactor(n_neighbors=n_neighbors,
                                  contamination=contamination,
                                  novelty=True)
    lof.fit(X_train)

    scores  = lof.decision_function(X_test)   # higher = more normal
    labels  = lof.predict(X_test)

    return pd.DataFrame({
        "value":         test_series.loc[feat_te.index].values,
        "anomaly_score": -scores,
        "anomaly":       labels == -1,
    }, index=feat_te.index)
```

---

## 6. Threshold Selection and Scoring

### 6.1 Score Calibration

```python
import numpy as np

def calibrate_isolation_forest(
    iso_scores: np.ndarray,
    target_rate: float = 0.02,
) -> dict:
    """
    Determine optimal threshold from anomaly score distribution.

    Parameters
    ----------
    iso_scores  : anomaly scores (higher = more anomalous)
    target_rate : desired fraction to flag as anomaly

    Returns
    -------
    dict with threshold and detection statistics
    """
    threshold = float(np.quantile(iso_scores, 1 - target_rate))

    return {
        "threshold":    threshold,
        "target_rate":  target_rate,
        "n_flagged":    int((iso_scores > threshold).sum()),
        "frac_flagged": float((iso_scores > threshold).mean()),
        "score_p95":    float(np.quantile(iso_scores, 0.95)),
        "score_p99":    float(np.quantile(iso_scores, 0.99)),
        "score_max":    float(iso_scores.max()),
    }
```

---

## 7. Ensemble Detector

```python
import numpy as np
import pandas as pd

class EnsembleAnomalyDetector:
    """
    Ensemble of Isolation Forest + LOF + OCSVM.
    Combines scores via rank aggregation (more robust than averaging raw scores).
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state  = random_state
        self.detectors_    = {}
        self.scalers_      = {}

    def fit(self, X_train: np.ndarray) -> "EnsembleAnomalyDetector":
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        from sklearn.neighbors import LocalOutlierFactor

        models = {
            "iso":  IsolationForest(n_estimators=200, contamination=self.contamination,
                                     random_state=self.random_state, n_jobs=-1),
            "ocsvm": OneClassSVM(nu=self.contamination, kernel="rbf", gamma="scale"),
            "lof":   LocalOutlierFactor(n_neighbors=20, contamination=self.contamination,
                                         novelty=True),
        }
        for name, model in models.items():
            model.fit(X_train)
            self.detectors_[name] = model

        return self

    def score(self, X_test: np.ndarray) -> np.ndarray:
        """Return ensemble anomaly score — higher = more anomalous."""
        raw_scores = {}
        for name, model in self.detectors_.items():
            s = -model.decision_function(X_test)   # flip sign
            raw_scores[name] = s

        # Rank-based normalization: convert each to [0, 1] rank-score
        from scipy.stats import rankdata
        rank_scores = np.column_stack([
            rankdata(s) / len(s) for s in raw_scores.values()
        ])
        return rank_scores.mean(axis=1)   # ensemble = mean rank

    def predict(self, X_test: np.ndarray, threshold: float = None) -> np.ndarray:
        """Return binary predictions (True = anomaly)."""
        scores = self.score(X_test)
        if threshold is None:
            threshold = 1 - self.contamination
        return scores > threshold
```

---

## 8. Production Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class TSAnomalyPipeline:
    """
    End-to-end time series anomaly detection pipeline:
    feature engineering → scaling → Isolation Forest → calibrated threshold.
    """

    def __init__(
        self,
        lags: list = None,
        rolling_windows: list = None,
        n_estimators: int = 200,
        target_fpr: float = 0.02,
        random_state: int = 42,
    ):
        self.lags            = lags or [1, 2, 3, 7, 14]
        self.rolling_windows = rolling_windows or [7, 30]
        self.n_estimators    = n_estimators
        self.target_fpr      = target_fpr
        self.random_state    = random_state
        self._fitted         = False

    def fit(self, series: pd.Series) -> "TSAnomalyPipeline":
        features     = build_anomaly_features(series, self.lags, self.rolling_windows)
        self._scaler = StandardScaler()
        X            = self._scaler.fit_transform(features.values)

        self._iso    = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.target_fpr,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._iso.fit(X)

        # Calibrate threshold on training data
        train_scores    = -self._iso.decision_function(X)
        self._threshold = float(np.quantile(train_scores, 1 - self.target_fpr))
        self._fitted    = True
        return self

    def predict(self, series: pd.Series) -> pd.DataFrame:
        assert self._fitted, "Call .fit() first."
        features = build_anomaly_features(series, self.lags, self.rolling_windows)
        X        = self._scaler.transform(features.values)
        scores   = -self._iso.decision_function(X)

        return pd.DataFrame({
            "value":         series.loc[features.index].values,
            "anomaly_score": scores,
            "anomaly":       scores > self._threshold,
        }, index=features.index)
```

---

*← [01 — Statistical Detection](./01_statistical_anomaly_detection.md) | [Module README](./README.md) | Next: [03 — Autoencoder Detection](./03_autoencoder_anomaly_detection.md) →*
