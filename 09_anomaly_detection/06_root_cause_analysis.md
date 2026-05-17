# 06 — Root Cause Analysis

> **Module**: 09 Anomaly Detection | **File**: 6 of 6
>
> Detecting an anomaly tells you *when* something went wrong. Root cause analysis (RCA) tells you *why*. For multivariate time series, RCA traces anomalies back to the causal sensor or metric — using correlation networks, Granger causality, contribution analysis, and causal graph methods.

---

## Table of Contents

1. [Why RCA Matters](#1-why-rca-matters)
2. [Correlation-Based Attribution](#2-correlation-based-attribution)
3. [Granger Causality for RCA](#3-granger-causality-for-rca)
4. [Contribution Score Analysis](#4-contribution-score-analysis)
5. [Causal Graph Methods — PCMCI](#5-causal-graph-methods--pcmci)
6. [Anomaly Propagation Trees](#6-anomaly-propagation-trees)
7. [Practical RCA Workflow](#7-practical-rca-workflow)

---

## 1. Why RCA Matters

### 1.1 The Detection-to-Action Gap

```
Without RCA:
  Alert: "Anomaly detected at 14:32"
  Operator: "Which sensor caused this? Which subsystem? What do I fix?"
  → No answer → alert ignored or takes hours to diagnose

With RCA:
  Alert: "Anomaly detected at 14:32"
  + "Root cause: Sensor CPU_TEMP on Server-07 exceeded normal range"
  + "Correlated anomalies: 3 downstream metrics affected 90s later"
  → Operator knows exactly where to look
```

### 1.2 RCA Approaches Overview

| Method                     | Data Needed         | Detects             | Complexity  |
|----------------------------|---------------------|---------------------|-------------|
| Correlation network        | Multivariate TS     | Correlated anomalies| Low         |
| Contribution analysis      | Anomaly score + features | Per-feature impact | Low    |
| Granger causality          | Multivariate TS     | Predictive causality| Medium      |
| PCMCI causal graph         | Multivariate TS     | Directed causes     | High        |
| Shapley values (SHAP)      | Model + features    | Feature attribution | Medium      |
| Anomaly propagation tree   | Dependency graph    | Cascade failures    | Medium      |

---

## 2. Correlation-Based Attribution

### 2.1 Cross-Correlation at Anomaly Times

Simple but effective: which features are most correlated with the anomaly score at the detected time?

```python
import numpy as np
import pandas as pd

def anomaly_correlation_analysis(
    df: pd.DataFrame,
    anomaly_times: list,
    window_before: int = 5,
    window_after: int = 2,
) -> pd.DataFrame:
    """
    Compute per-feature anomaly magnitude at detected anomaly times.

    Parameters
    ----------
    df            : (T, D) DataFrame of all sensor/feature values
    anomaly_times : list of time indices where anomalies were detected
    window_before : time steps before anomaly to include in analysis
    window_after  : time steps after anomaly to include in analysis

    Returns
    -------
    DataFrame with per-feature anomaly z-scores at anomaly events
    """
    results = []
    normal_stats = df.agg(["mean", "std"])

    for t in anomaly_times:
        start = max(0, t - window_before)
        end   = min(len(df), t + window_after + 1)
        window_df = df.iloc[start:end]

        # Z-score of each feature at anomaly window vs. normal baseline
        z_scores = {}
        for col in df.columns:
            mu    = normal_stats.loc["mean", col]
            sigma = normal_stats.loc["std",  col] + 1e-12
            z     = float(np.abs((window_df[col].values - mu) / sigma).max())
            z_scores[col] = z

        results.append({"anomaly_time": t, **z_scores})

    result_df = pd.DataFrame(results).set_index("anomaly_time")
    return result_df


def top_contributing_features(
    anomaly_corr_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Rank features by their average anomaly-time z-score.
    Higher z-score → more likely to be root cause.
    """
    mean_z = anomaly_corr_df.mean(axis=0).sort_values(ascending=False)
    return pd.DataFrame({
        "feature":      mean_z.index[:top_k].tolist(),
        "mean_z_score": mean_z.values[:top_k].round(3),
        "rank":         range(1, top_k + 1),
    })
```

### 2.2 Lagged Cross-Correlation

The root cause often precedes the anomaly in time. Compute cross-correlations at different lags to find leading indicators:

```python
def lagged_cross_correlation(
    anomaly_score: np.ndarray,
    features: pd.DataFrame,
    max_lag: int = 10,
) -> pd.DataFrame:
    """
    Compute cross-correlation between each feature and the anomaly score
    at lags [-max_lag, ..., 0, ..., +max_lag].

    Negative lag (feature leads score): feature is a LEADING INDICATOR
    Positive lag (feature lags score):  feature is a DOWNSTREAM EFFECT

    Parameters
    ----------
    anomaly_score : 1D anomaly score array (same length as df)
    features      : DataFrame of candidate root-cause features
    max_lag       : maximum lag to test (in time steps)

    Returns
    -------
    DataFrame: rows = features, columns = lag values, values = correlation
    """
    score  = (anomaly_score - np.nanmean(anomaly_score)) / (np.nanstd(anomaly_score) + 1e-12)
    result = {}

    for col in features.columns:
        feat   = features[col].values.astype(float)
        feat   = (feat - np.nanmean(feat)) / (np.nanstd(feat) + 1e-12)
        corrs  = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # Feature LEADS score — shift feature forward
                corr = np.corrcoef(feat[:lag], score[-lag:])[0, 1]
            elif lag > 0:
                # Feature LAGS score — shift score forward
                corr = np.corrcoef(feat[lag:], score[:-lag])[0, 1]
            else:
                corr = np.corrcoef(feat, score)[0, 1]
            corrs[lag] = round(float(corr), 4)
        result[col] = corrs

    df = pd.DataFrame(result).T
    df.index.name    = "feature"
    df.columns.name  = "lag"
    return df
```

---

## 3. Granger Causality for RCA

### 3.1 What is Granger Causality?

```
Granger causality (Granger, 1969):
  X Granger-causes Y if:
    Predicting Y using its own past AND X's past
    is significantly better than predicting Y using its own past alone.

Test:
  H₀: X does NOT Granger-cause Y  (X does not help predict Y)
  H₁: X DOES Granger-cause Y     (X has predictive power for Y)

If p-value < α → reject H₀ → X Granger-causes Y

Limitation: Granger causality is PREDICTIVE, not structural causality.
            "X predicts Y" ≠ "X causes Y" (confounders possible)
```

### 3.2 Implementation

```python
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
import numpy as np

def granger_causality_matrix(
    df: pd.DataFrame,
    max_lag: int = 5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute pairwise Granger causality for all feature pairs.

    Returns a (D x D) matrix:
      matrix[i, j] = True if column_i Granger-causes column_j

    Parameters
    ----------
    df      : (T, D) DataFrame of stationary time series
    max_lag : maximum lag to test (start with 1–5 for most TS)
    alpha   : significance level for F-test

    Returns
    -------
    cause_matrix : DataFrame — True = row variable Granger-causes column variable
    """
    cols   = df.columns.tolist()
    n_cols = len(cols)
    matrix = pd.DataFrame(False, index=cols, columns=cols)

    for target in cols:
        for cause in cols:
            if cause == target:
                continue
            try:
                data   = df[[target, cause]].dropna()
                result = grangercausalitytests(data, maxlag=max_lag, verbose=False)

                # Check if ANY lag has significant F-test
                significant = any(
                    result[lag][0]["ssr_ftest"][1] < alpha
                    for lag in range(1, max_lag + 1)
                )
                matrix.loc[cause, target] = significant

            except Exception:
                pass  # insufficient data or non-stationary series

    return matrix


def find_root_causes_granger(
    df: pd.DataFrame,
    anomaly_feature: str,
    max_lag: int = 5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Find which features Granger-cause a specific anomalous feature.

    Parameters
    ----------
    df               : multivariate time series (stationary)
    anomaly_feature  : the feature that is anomalous (the target)
    max_lag          : lag order to test
    alpha            : significance level

    Returns
    -------
    DataFrame with candidate root causes, min p-value, and best lag
    """
    cause_cols = [c for c in df.columns if c != anomaly_feature]
    results    = []

    for cause in cause_cols:
        data = df[[anomaly_feature, cause]].dropna()
        try:
            gc_res = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            p_vals = {
                lag: gc_res[lag][0]["ssr_ftest"][1]
                for lag in range(1, max_lag + 1)
            }
            best_lag  = min(p_vals, key=p_vals.get)
            min_pval  = p_vals[best_lag]
            results.append({
                "cause":        cause,
                "min_p_value":  round(min_pval, 5),
                "best_lag":     best_lag,
                "significant":  min_pval < alpha,
            })
        except Exception:
            pass

    result_df = pd.DataFrame(results).sort_values("min_p_value")
    return result_df[result_df["significant"]]
```

---

## 4. Contribution Score Analysis

### 4.1 SHAP for Anomaly Explanation

When using a model-based anomaly detector (Isolation Forest, autoencoder), SHAP values explain **which features contributed most** to a high anomaly score:

```python
import shap
import numpy as np
import pandas as pd

def shap_anomaly_explanation(
    iso_model,
    X_background: np.ndarray,
    X_anomalous: np.ndarray,
    feature_names: list = None,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Use SHAP TreeExplainer to attribute anomaly scores to features.

    Parameters
    ----------
    iso_model    : fitted IsolationForest (sklearn)
    X_background : background data for SHAP (e.g., training set sample)
    X_anomalous  : the anomalous observations to explain
    feature_names: list of feature names
    top_k        : number of top features to return

    Returns
    -------
    DataFrame with mean |SHAP| per feature (feature importance for anomaly)
    """
    explainer  = shap.TreeExplainer(iso_model, data=X_background)
    shap_vals  = explainer.shap_values(X_anomalous)   # (n_anomalies, n_features)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_anomalous.shape[1])]

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    idx           = np.argsort(mean_abs_shap)[::-1]

    return pd.DataFrame({
        "feature":        [feature_names[i] for i in idx[:top_k]],
        "mean_|SHAP|":    mean_abs_shap[idx[:top_k]].round(5),
        "rank":           range(1, top_k + 1),
    })
```

### 4.2 Reconstruction Error Contribution (Autoencoder RCA)

For autoencoder detectors, the per-feature reconstruction error directly indicates which features were anomalous:

```python
def autoencoder_contribution(
    original: np.ndarray,
    reconstructed: np.ndarray,
    feature_names: list = None,
) -> pd.DataFrame:
    """
    Per-feature reconstruction error contribution for autoencoder RCA.

    Parameters
    ----------
    original      : (n_windows, n_features) — actual input windows
    reconstructed : (n_windows, n_features) — autoencoder output
    feature_names : feature/sensor names

    Returns
    -------
    DataFrame with per-feature reconstruction error and normalized contribution
    """
    errors     = np.abs(original - reconstructed)
    mean_error = errors.mean(axis=0)   # (n_features,)
    total      = mean_error.sum() + 1e-12

    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(len(mean_error))]

    df = pd.DataFrame({
        "feature":       feature_names,
        "mean_error":    mean_error.round(5),
        "contribution%": (100 * mean_error / total).round(2),
    }).sort_values("mean_error", ascending=False)

    return df.reset_index(drop=True)
```

---

## 5. Causal Graph Methods — PCMCI

### 5.1 PCMCI Overview

PCMCI (Runge et al., 2019) is a state-of-the-art causal discovery algorithm for time series. Unlike Granger causality (pairwise), PCMCI tests **conditional independence** to remove spurious correlations:

```
PCMCI steps:
  1. PC phase (skeleton): Use partial correlations to prune non-causal links
     Test: X ⊥ Y | {Z}: is X independent of Y given a conditioning set Z?
     → Remove link X→Y if independence holds

  2. MCI test (Momentary Conditional Independence):
     Test: X(t-τ) → Y(t) | {Y's own past, Z's past}
     → More robust to autocorrelation inflation

Output:
  - Directed causal graph over all variables and lags
  - p-values for each causal link
  - Causal effect magnitudes (path coefficients)

Advantage over Granger:
  → Removes spurious links due to common drivers (confounders)
  → Provides full directed graph, not just pairwise tests
```

```python
def run_pcmci(
    df: pd.DataFrame,
    max_lag: int = 5,
    alpha: float = 0.05,
    target_col: str = None,
) -> dict:
    """
    Run PCMCI causal discovery on multivariate time series.

    Requirements: pip install tigramite

    Parameters
    ----------
    df         : (T, D) DataFrame of time series (stationary)
    max_lag    : maximum causal lag to test
    alpha      : significance threshold for causal links
    target_col : if specified, return causes of this column only

    Returns
    -------
    dict with causal graph, p-values, and significant links
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
    except ImportError:
        print("Install tigramite: pip install tigramite")
        return {}

    data       = df.values.astype(float)
    dataframe  = pp.DataFrame(data, var_names=df.columns.tolist())

    pcmci      = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results    = pcmci.run_pcmci(tau_max=max_lag, alpha_level=alpha)

    p_matrix   = results["p_matrix"]    # (D, D, max_lag+1)
    val_matrix = results["val_matrix"]  # causal effect estimates

    cols = df.columns.tolist()
    significant_links = []
    for i, cause in enumerate(cols):
        for j, effect in enumerate(cols):
            for lag in range(1, max_lag + 1):
                if p_matrix[i, j, lag] < alpha:
                    significant_links.append({
                        "cause":  cause,
                        "effect": effect,
                        "lag":    lag,
                        "p":      round(p_matrix[i, j, lag], 5),
                        "coeff":  round(val_matrix[i, j, lag], 4),
                    })

    result = {
        "significant_links": pd.DataFrame(significant_links),
        "p_matrix":          p_matrix,
        "val_matrix":        val_matrix,
        "var_names":         cols,
    }

    if target_col:
        df_links = result["significant_links"]
        result["root_causes"] = df_links[df_links["effect"] == target_col] \
                                         .sort_values("p")

    return result
```

---

## 6. Anomaly Propagation Trees

### 6.1 Dependency Graph-Based RCA

In complex systems, anomalies propagate through dependency graphs (microservices, sensor networks, supply chains). Track the anomaly backward through the graph:

```python
from collections import defaultdict, deque

def build_dependency_graph(edges: list[tuple]) -> dict:
    """
    Build a directed dependency graph.

    Parameters
    ----------
    edges : list of (cause, effect) tuples
            e.g., [("database", "api"), ("api", "frontend"), ("cache", "api")]

    Returns
    -------
    graph : dict {node: [downstream nodes]}
    reverse: dict {node: [upstream nodes]} (for backward tracing)
    """
    graph   = defaultdict(list)
    reverse = defaultdict(list)
    for cause, effect in edges:
        graph[cause].append(effect)
        reverse[effect].append(cause)
    return dict(graph), dict(reverse)


def trace_root_causes(
    anomalous_node: str,
    reverse_graph: dict,
    anomaly_times: dict,
    max_depth: int = 5,
    time_tolerance: int = 10,
) -> list[dict]:
    """
    Trace root causes backward through dependency graph.

    Starting from the anomalous node, walk upstream through the graph
    to find nodes that were anomalous BEFORE the target node.

    Parameters
    ----------
    anomalous_node  : the node where the anomaly was detected
    reverse_graph   : {node: [upstream nodes]}
    anomaly_times   : {node: first_anomaly_timestamp}
    max_depth       : maximum hops upstream to trace
    time_tolerance  : max seconds between upstream and downstream anomaly
                      (to allow for propagation delay)

    Returns
    -------
    list of candidate root causes with propagation path
    """
    target_time = anomaly_times.get(anomalous_node, float("inf"))
    candidates  = []
    queue       = deque([(anomalous_node, 0, [anomalous_node])])

    while queue:
        node, depth, path = queue.popleft()
        if depth >= max_depth:
            continue

        for upstream in reverse_graph.get(node, []):
            upstream_time = anomaly_times.get(upstream, float("inf"))
            if upstream_time < target_time + time_tolerance:
                new_path = [upstream] + path
                candidates.append({
                    "root_candidate": upstream,
                    "anomaly_time":   upstream_time,
                    "propagation_path": " → ".join(new_path),
                    "depth": depth + 1,
                    "lead_time": target_time - upstream_time,
                })
                queue.append((upstream, depth + 1, new_path))

    return sorted(candidates, key=lambda x: x["anomaly_time"])
```

---

## 7. Practical RCA Workflow

### 7.1 Recommended Production Workflow

```python
import numpy as np
import pandas as pd

class RootCauseAnalyzer:
    """
    Practical RCA workflow combining correlation, lagged correlation,
    and contribution analysis for multivariate anomaly investigation.
    """

    def __init__(
        self,
        max_lag: int = 10,
        alpha:   float = 0.05,
        top_k:   int  = 5,
    ):
        self.max_lag = max_lag
        self.alpha   = alpha
        self.top_k   = top_k

    def analyze(
        self,
        df: pd.DataFrame,
        anomaly_score: np.ndarray,
        anomaly_times: list,
    ) -> dict:
        """
        Full RCA pipeline.

        Returns
        -------
        dict with:
          - feature_z_scores: anomaly z-scores at anomaly events
          - top_features:     ranked candidate root causes
          - lagged_corr:      lagged cross-correlations
          - granger_causes:   Granger-causal features (if stationary)
        """
        results = {}

        # 1. Feature anomaly z-scores at detected times
        results["feature_z_scores"] = anomaly_correlation_analysis(
            df, anomaly_times
        )
        results["top_features"] = top_contributing_features(
            results["feature_z_scores"], self.top_k
        )

        # 2. Lagged cross-correlation (leading indicators)
        results["lagged_corr"] = lagged_cross_correlation(
            anomaly_score, df, self.max_lag
        )

        # 3. Peak lag per feature (which feature leads the anomaly score)
        peak_lags = {}
        for col in df.columns:
            row  = results["lagged_corr"].loc[col]
            peak = int(row.abs().idxmax())
            peak_lags[col] = {
                "best_lag":    peak,
                "correlation": row[peak],
                "leading":     peak < 0,
            }
        results["peak_lags"] = pd.DataFrame(peak_lags).T.sort_values("correlation",
                                                                       ascending=False)

        return results

    def report(self, results: dict, anomaly_times: list):
        """Print formatted RCA report."""
        print("=" * 60)
        print("ROOT CAUSE ANALYSIS REPORT")
        print("=" * 60)
        print(f"Anomaly detected at times: {anomaly_times[:5]}{'...' if len(anomaly_times) > 5 else ''}")

        print("\n📊 Top Contributing Features (by max z-score):")
        print(results["top_features"].to_string(index=False))

        print("\n⏱ Leading Indicators (negative lag = feature leads anomaly):")
        leads = results["peak_lags"]
        leads = leads[leads["leading"] == True].head(self.top_k)
        if len(leads):
            print(leads[["best_lag", "correlation"]].to_string())
        else:
            print("  No leading indicators found (anomaly may be exogenous)")
```

---

*← [05 — Online Detection](./05_online_anomaly_detection.md) | [Module README](./README.md)*
