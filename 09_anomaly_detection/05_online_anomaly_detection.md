# 05 — Online Anomaly Detection

> **Module**: 09 Anomaly Detection | **File**: 5 of 6
>
> Offline detectors assume all data is available at once. In production, data arrives as a **stream** — one observation at a time, with tight latency requirements. Online anomaly detection algorithms update in O(1) time per observation, maintain bounded memory, and adapt to distribution shifts without retraining.

---

## Table of Contents

1. [Online vs. Offline Detection](#1-online-vs-offline-detection)
2. [ADWIN — Adaptive Windowing](#2-adwin--adaptive-windowing)
3. [HBOS — Histogram-Based Outlier Score](#3-hbos--histogram-based-outlier-score)
4. [Half-Space Trees](#4-half-space-trees)
5. [Online Z-Score with Welford Update](#5-online-z-score-with-welford-update)
6. [RRCF — Robust Random Cut Forest](#6-rrcf--robust-random-cut-forest)
7. [river Library Integration](#7-river-library-integration)
8. [Production Streaming Pipeline](#8-production-streaming-pipeline)

---

## 1. Online vs. Offline Detection

### 1.1 Core Differences

```
Offline (batch) detector:
  - All data available → fit once, score all
  - Can use global statistics, complex models
  - Latency: acceptable (minutes/hours)
  - Memory: unlimited (stores full history)
  - Distribution shift: manual retraining required

Online (streaming) detector:
  - One observation arrives at a time
  - Must score within milliseconds
  - Memory: bounded (cannot store full history)
  - Distribution shift: MUST adapt automatically
  - State: small summary structure updated incrementally

Key requirement: Online algorithms must process each observation
                  in O(1) time and O(1) or O(log n) memory.
```

### 1.2 Design Principles

```
1. Sufficient statistics: maintain only statistics needed for scoring
   (mean, variance, histogram bins — not raw data)

2. Forgetting factor: weight recent data more than old data
   (exponential decay or sliding window)

3. Concept drift adaptation: detect when the distribution changes
   and reset/update the model accordingly

4. Low false alarm rate: streaming alerts must be actionable
   (too many alerts → ignored by operators → model abandoned)
```

---

## 2. ADWIN — Adaptive Windowing

### 2.1 Algorithm

ADWIN (Bifet & Gavalda, 2007) detects **concept drift** by maintaining a variable-length sliding window. It splits the window at every possible point and tests whether the left and right halves have the same mean:

```
Window W maintained adaptively:
  At each step:
    1. Add new observation to window W
    2. For all valid splits of W into W₀ (old) and W₁ (new):
         If |μ(W₀) - μ(W₁)| > ε_cut:
           → Distribution changed → DROP old part of W
           → W := W₁  (keep only the new distribution portion)

ε_cut = √[(1/(2m₀) + 1/(2m₁)) · ln(4n²/δ)]

Where:
  m₀, m₁ = sizes of old and new sub-windows
  n       = current window size
  δ       = confidence parameter (e.g., 0.002)

Key property: false positive rate ≤ δ
```

### 2.2 Implementation

```python
import numpy as np
from collections import deque

class ADWIN:
    """
    Adaptive Windowing (ADWIN) drift and anomaly detector.
    Maintains a variable-length window that shrinks on distribution change.

    Reference: Bifet & Gavalda (2007)
    """

    def __init__(self, delta: float = 0.002):
        """
        Parameters
        ----------
        delta : confidence parameter — lower = fewer false alarms, slower detection
        """
        self.delta    = delta
        self._window  = deque()
        self._n       = 0
        self._total   = 0.0
        self._total2  = 0.0

    @property
    def mean(self) -> float:
        return self._total / self._n if self._n > 0 else 0.0

    @property
    def variance(self) -> float:
        if self._n < 2:
            return 0.0
        return (self._total2 - self._total**2 / self._n) / (self._n - 1)

    def update(self, x: float) -> dict:
        """
        Add new observation. Returns drift detection result.

        Returns
        -------
        dict with {drift_detected, current_mean, window_size}
        """
        self._window.append(x)
        self._n      += 1
        self._total  += x
        self._total2 += x**2

        drift = self._detect_drift()

        return {
            "drift_detected": drift,
            "current_mean":   self.mean,
            "window_size":    self._n,
            "variance":       self.variance,
        }

    def _detect_drift(self) -> bool:
        """Test all splits for significant mean difference."""
        if self._n < 4:
            return False

        window_list = list(self._window)

        # Check splits (simplified: compare first half vs. second half)
        split_points = range(max(1, self._n // 10), self._n - 1, max(1, self._n // 20))

        for m0 in split_points:
            m1 = self._n - m0
            if m0 < 1 or m1 < 1:
                continue

            w0 = window_list[:m0]
            w1 = window_list[m0:]
            mu0, mu1 = np.mean(w0), np.mean(w1)

            # epsilon_cut
            epsilon = np.sqrt(
                (1 / (2 * m0) + 1 / (2 * m1))
                * np.log(4 * self._n**2 / self.delta)
            )

            if abs(mu0 - mu1) > epsilon:
                # Drop old part of window
                for _ in range(m0):
                    old = self._window.popleft()
                    self._n      -= 1
                    self._total  -= old
                    self._total2 -= old**2
                return True

        return False
```

---

## 3. HBOS — Histogram-Based Outlier Score

### 3.1 Algorithm

HBOS (Goldstein & Dengel, 2012) is one of the fastest anomaly detectors. It assumes **feature independence** and uses histograms to estimate the density:

```
For each feature j:
  1. Build histogram H_j with K bins from training data
  2. Density estimate at value v: p_j(v) = frequency(bin containing v) / total

Anomaly score:
  HBOS(x) = Σⱼ log(1 / p_j(xⱼ))   (sum of negative log-likelihoods)

High HBOS → low density in all features → likely anomaly

Advantages:
  ✅ O(1) per observation (just histogram lookup)
  ✅ Linear time training
  ✅ Handles high-dimensional data well
  ❌ Ignores feature correlations (assumes independence)
```

### 3.2 Implementation

```python
import numpy as np

class OnlineHBOS:
    """
    Online Histogram-Based Outlier Score.
    Maintains histograms with online updates via exponential forgetting.
    """

    def __init__(self, n_bins: int = 20, alpha: float = 0.01):
        """
        Parameters
        ----------
        n_bins : number of histogram bins per feature
        alpha  : forgetting factor — lower = slower adaptation to distribution shifts
        """
        self.n_bins   = n_bins
        self.alpha    = alpha
        self._fitted  = False

    def fit(self, X_train: np.ndarray) -> "OnlineHBOS":
        """Initialize histograms from training data."""
        X_train         = np.asarray(X_train, dtype=float)
        self.n_features = X_train.shape[1] if X_train.ndim > 1 else 1
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)

        self._edges  = []
        self._counts = []

        for j in range(self.n_features):
            counts, edges = np.histogram(X_train[:, j], bins=self.n_bins)
            self._edges.append(edges)
            self._counts.append(counts.astype(float) + 1e-6)   # Laplace smoothing

        self._fitted = True
        return self

    def _density(self, x: float, j: int) -> float:
        """Estimate density of value x for feature j using histogram."""
        edges   = self._edges[j]
        counts  = self._counts[j]
        total   = counts.sum()

        # Find bin
        bin_idx = np.searchsorted(edges[1:], x)
        bin_idx = np.clip(bin_idx, 0, len(counts) - 1)

        # Density = count / (total * bin_width)
        width   = edges[bin_idx + 1] - edges[bin_idx] + 1e-12
        return float(counts[bin_idx] / (total * width))

    def score(self, x: np.ndarray) -> float:
        """Compute HBOS score for a single observation."""
        assert self._fitted, "Call .fit() first."
        x = np.asarray(x, dtype=float).flatten()
        if len(x) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x)}")

        score = sum(
            np.log(1 / (self._density(x[j], j) + 1e-12))
            for j in range(self.n_features)
        )
        return float(score)

    def update(self, x: np.ndarray) -> None:
        """Update histograms online with new observation (soft update)."""
        x = np.asarray(x, dtype=float).flatten()
        for j in range(self.n_features):
            edges  = self._edges[j]
            bin_i  = np.searchsorted(edges[1:], x[j])
            bin_i  = np.clip(bin_i, 0, self.n_bins - 1)
            # Exponential forgetting: decay old counts, add new
            self._counts[j] *= (1 - self.alpha)
            self._counts[j][bin_i] += 1.0
```

---

## 4. Half-Space Trees

### 4.1 Concept

Half-Space Trees (Tan et al., 2011) are specifically designed for streaming anomaly detection. They split the feature space with random axis-aligned hyperplanes and measure mass (density) in each partition:

```
Algorithm:
  1. Build a set of random trees, each with random axis-aligned splits
  2. For each new observation x:
       - Navigate each tree to find the partition containing x
       - The anomaly score = size of partition (small partition → rare region → anomaly)
  3. Mass estimation updates in O(1) per observation

Key advantage: designed for streaming; handles concept drift
```

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class HSTNode:
    """Node in a Half-Space Tree."""
    feature:  int   = 0
    threshold: float = 0.0
    left:     Optional["HSTNode"] = None
    right:    Optional["HSTNode"] = None
    l_mass:   float = 0.0   # mass counter (reference window)
    r_mass:   float = 0.0
    l_cmass:  float = 0.0   # latest window mass counter
    r_cmass:  float = 0.0

class HalfSpaceTree:
    """
    Half-Space Tree for streaming anomaly detection.
    Simplified implementation for single-variate series.
    """

    def __init__(self, max_depth: int = 15, window_size: int = 250, n_trees: int = 25):
        self.max_depth   = max_depth
        self.window_size = window_size
        self.n_trees     = n_trees
        self._t          = 0   # observation counter
        self._trees      = []

    def _build_tree(self, depth: int, min_v: float, max_v: float) -> HSTNode:
        node = HSTNode(feature=0, threshold=(min_v + max_v) / 2)
        if depth < self.max_depth:
            mid = (min_v + max_v) / 2
            node.left  = self._build_tree(depth + 1, min_v, mid)
            node.right = self._build_tree(depth + 1, mid, max_v)
        return node

    def fit(self, X_init: np.ndarray, min_v: float = None, max_v: float = None):
        """Initialize trees with reference window."""
        X = np.asarray(X_init, dtype=float).flatten()
        if min_v is None: min_v = X.min()
        if max_v is None: max_v = X.max()
        self._trees = [self._build_tree(0, min_v, max_v) for _ in range(self.n_trees)]
        for x in X:
            self._update_mass(x, reference=True)
        return self

    def _update_mass(self, x: float, reference: bool = False):
        for tree in self._trees:
            node = tree
            while node is not None:
                if x <= node.threshold:
                    if reference: node.l_mass  += 1
                    else:         node.l_cmass += 1
                    node = node.left
                else:
                    if reference: node.r_mass  += 1
                    else:         node.r_cmass += 1
                    node = node.right

    def score(self, x: float) -> float:
        """Anomaly score: small r_mass or l_mass → anomaly (low density)."""
        total_score = 0.0
        for tree in self._trees:
            node = tree
            depth = 0
            while node is not None:
                depth += 1
                if x <= node.threshold:
                    total_score += node.l_mass * (2 ** depth)
                    node = node.left
                else:
                    total_score += node.r_mass * (2 ** depth)
                    node = node.right
        # Normalize and invert (low mass = high anomaly score)
        return float(1.0 / (total_score / self.n_trees + 1e-12))
```

---

## 5. Online Z-Score with Welford Update

### 5.1 Welford's Algorithm

Compute running mean and variance **without storing all observations**:

```
At each new observation xₙ:
  δ  = xₙ - μₙ₋₁
  μₙ = μₙ₋₁ + δ/n
  δ₂ = xₙ - μₙ
  Mₙ = Mₙ₋₁ + δ·δ₂
  σₙ² = Mₙ / (n-1)

Memory: O(1) — stores only (μ, M, n)
```

### 5.2 Implementation

```python
class OnlineZScoreDetector:
    """
    Online anomaly detector using Welford's running mean/variance.
    Supports exponential forgetting for drift adaptation.
    """

    def __init__(self, threshold: float = 3.0, forgetting: float = None):
        """
        Parameters
        ----------
        threshold  : |z| > threshold → anomaly
        forgetting : exponential forgetting factor λ ∈ (0, 1)
                     None = equal weighting of all history
        """
        self.threshold  = threshold
        self.forgetting = forgetting
        self._n    = 0
        self._mu   = 0.0
        self._M    = 0.0    # Welford's M (sum of squared deviations)
        self._w    = 0.0    # total weight (for forgetting)

    @property
    def mean(self) -> float:
        return self._mu

    @property
    def std(self) -> float:
        if self._n < 2:
            return 1.0
        denom = self._w - self._w / self._n if self.forgetting else self._n - 1
        return float(np.sqrt(max(self._M / denom, 1e-12)))

    def update(self, x: float) -> dict:
        """Process one observation; return anomaly result."""
        self._n += 1

        if self.forgetting is not None:
            lam = self.forgetting
            self._w  = lam * self._w + 1.0
            delta    = x - self._mu
            self._mu += delta / self._w
            self._M  = lam * self._M + delta * (x - self._mu)
        else:
            # Welford's online update
            delta    = x - self._mu
            self._mu += delta / self._n
            delta2   = x - self._mu
            self._M  += delta * delta2
            self._w   = float(self._n)

        z       = (x - self._mu) / (self.std + 1e-12)
        anomaly = abs(z) > self.threshold

        return {
            "value":   x,
            "z_score": float(z),
            "mu":      self._mu,
            "std":     self.std,
            "anomaly": bool(anomaly),
        }
```

---

## 6. RRCF — Robust Random Cut Forest

### 6.1 Concept

RRCF (Guha et al., 2016) — the algorithm behind **Amazon Lookout for Metrics** and AWS Kinesis Analytics anomaly detection. It is designed for streaming data:

```
Algorithm:
  1. Maintain a forest of Robust Random Cut Trees
  2. Each tree partitions the data space randomly
  3. Anomaly score = CODISP(x, tree):
       Collusive Displacement = how much the model complexity changes
       when point x is added/removed

  High CODISP → point is anomalous (its removal simplifies the tree significantly)

Key features:
  ✅ Handles concept drift (trees updated as stream progresses)
  ✅ Works on multivariate data
  ✅ O(log n) per observation
  ✅ Used in production at Amazon scale
```

```python
# pip install rrcf

def rrcf_streaming_detector(
    series: np.ndarray,
    shingle_size: int = 4,
    n_trees: int = 40,
    tree_size: int = 256,
    threshold_pct: float = 99.0,
) -> np.ndarray:
    """
    RRCF anomaly detection for streaming time series.

    Parameters
    ----------
    shingle_size   : window size for shingling (creates multivariate features)
    n_trees        : number of trees in the forest
    tree_size      : maximum tree size (old points evicted when full)
    threshold_pct  : percentile of scores to use as anomaly threshold

    Returns
    -------
    avg_codisp : anomaly score per time step (NaN for warm-up period)
    """
    try:
        import rrcf
    except ImportError:
        print("Install rrcf: pip install rrcf")
        return np.full(len(series), np.nan)

    forest     = [rrcf.RCTree() for _ in range(n_trees)]
    avg_codisp = np.zeros(len(series))
    shingle    = np.zeros(shingle_size)

    for t, x in enumerate(series):
        # Update shingle (sliding window)
        shingle = np.roll(shingle, -1)
        shingle[-1] = x

        if t < shingle_size - 1:
            avg_codisp[t] = np.nan
            continue

        # Update each tree and compute average CODISP
        codisp = 0.0
        for tree in forest:
            # Remove oldest point if tree is full
            if len(tree.leaves) > tree_size:
                oldest = min(tree.leaves, key=lambda x: x)
                tree.forget_point(oldest)

            # Insert new shingle
            tree.insert_point(shingle.copy(), index=t)
            codisp += tree.codisp(t)

        avg_codisp[t] = codisp / n_trees

    return avg_codisp
```

---

## 7. river Library Integration

```python
# pip install river

def river_streaming_demo(series: np.ndarray) -> pd.DataFrame:
    """
    Online anomaly detection using the river library.
    Demonstrates HalfSpaceTrees from river — the most reliable online detector.
    """
    try:
        from river import anomaly, preprocessing
    except ImportError:
        print("Install river: pip install river")
        return pd.DataFrame()

    model = anomaly.HalfSpaceTrees(
        n_trees=25,
        height=15,
        window_size=250,
        seed=42,
    )

    results = []
    for t, x in enumerate(series):
        score   = model.score_one({"x": x})
        model.learn_one({"x": x})
        results.append({"t": t, "value": x, "score": score})

    return pd.DataFrame(results)
```

---

## 8. Production Streaming Pipeline

```python
import numpy as np
import pandas as pd
from collections import deque

class StreamingAnomalyPipeline:
    """
    Production-ready streaming anomaly detection pipeline.

    Combines:
      - Online Z-score (for fast, interpretable spike detection)
      - ADWIN (for drift detection → triggers model reset)
      - Score smoothing and alert management
    """

    def __init__(
        self,
        z_threshold: float = 3.5,
        forgetting:  float = 0.98,
        adwin_delta: float = 0.002,
        smooth_n:    int   = 5,
        alert_cooldown: int = 10,
    ):
        self.z_threshold    = z_threshold
        self.smooth_n       = smooth_n
        self.alert_cooldown = alert_cooldown
        self._zscore = OnlineZScoreDetector(z_threshold, forgetting)
        self._adwin  = ADWIN(adwin_delta)
        self._score_buffer = deque(maxlen=smooth_n)
        self._last_alert   = -alert_cooldown
        self._t            = 0
        self.events_       = []

    def process(self, x: float) -> dict:
        """Process one streaming observation."""
        self._t += 1

        # Z-score detector
        z_res = self._zscore.update(x)

        # ADWIN drift detector
        adwin_res = self._adwin.update(x)

        # Smoothed score
        self._score_buffer.append(abs(z_res["z_score"]))
        smoothed = np.mean(self._score_buffer)

        # Alert logic (with cooldown to avoid flooding)
        alert = (
            smoothed > self.z_threshold and
            self._t - self._last_alert >= self.alert_cooldown
        )
        if alert:
            self._last_alert = self._t

        # Drift event
        if adwin_res["drift_detected"]:
            self.events_.append({
                "t": self._t, "event": "drift",
                "new_mean": adwin_res["current_mean"],
            })

        result = {
            "t":            self._t,
            "value":        x,
            "z_score":      z_res["z_score"],
            "smoothed_z":   smoothed,
            "anomaly":      alert,
            "drift":        adwin_res["drift_detected"],
            "mu":           z_res["mu"],
            "std":          z_res["std"],
        }
        return result

    def run_batch(self, series: np.ndarray) -> pd.DataFrame:
        """Process a batch of observations through the streaming pipeline."""
        return pd.DataFrame([self.process(x) for x in series])
```

---

*← [04 — LSTM-Based Detection](./04_lstm_based_anomaly_detection.md) | [Module README](./README.md) | Next: [06 — Root Cause Analysis](./06_root_cause_analysis.md) →*
