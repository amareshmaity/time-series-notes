# 04 — Drift Detection and Monitoring

> **Module**: 11 Production & MLOps | **File**: 4 of 6
>
> Models don't fail silently — they drift. Data distributions shift, input patterns evolve, and the relationship between features and target changes over time. This note covers PSI-based data drift detection, KS/Wasserstein tests, concept drift monitoring via rolling error analysis, and alerting strategies.

---

## Table of Contents

1. [Data Drift vs. Concept Drift](#1-data-drift-vs-concept-drift)
2. [PSI — Population Stability Index](#2-psi--population-stability-index)
3. [KS Test and Wasserstein Distance](#3-ks-test-and-wasserstein-distance)
4. [Concept Drift — Rolling Performance Monitoring](#4-concept-drift--rolling-performance-monitoring)
5. [CUSUM for Performance Monitoring](#5-cusum-for-performance-monitoring)
6. [Evidently AI Integration](#6-evidently-ai-integration)
7. [Alerting and Response Playbook](#7-alerting-and-response-playbook)
8. [Production Monitoring Dashboard](#8-production-monitoring-dashboard)

---

## 1. Data Drift vs. Concept Drift

### 1.1 Taxonomy

```
DATA DRIFT (covariate shift):
  Definition: The distribution P(X) of input features changes.
  Example:    A new product launches → sales patterns shift in unexpected ways
              COVID → daily traffic patterns change structurally
  Detection:  PSI, KS test, Wasserstein distance on input features
  Impact:     Model may perform poorly even if the model is "correct"
  Response:   Retrain on recent data, add new features

CONCEPT DRIFT (posterior drift):
  Definition: The relationship P(Y|X) between features and target changes.
  Example:    Consumer behavior shifts → historical price-demand elasticity invalid
              Sensor calibration drifts → same reading means different temperature
  Detection:  Rolling prediction error vs. baseline performance
  Response:   Must retrain; feature engineering alone won't fix this

LABEL DRIFT (prior probability shift):
  Definition: The distribution P(Y) of the target shifts.
  Example:    Class imbalance changes (more anomalies during an event)
  Detection:  Monitor prediction distribution and actual label distribution

TEMPORAL DRIFT:
  Seasonal shifts that are expected and cyclic (not drift per se):
  → Handled by including calendar features and seasonal models
  → Monitor for unexpected magnitude of seasonal effects
```

### 1.2 Monitoring Matrix

```
                    P(X) stable?    P(Y|X) stable?
                    ───────────────────────────────
  No drift:              YES              YES
  Data drift only:        NO              YES
  Concept drift only:    YES               NO
  Both:                   NO               NO

Detection approach:
  Data drift:    PSI / KS test on features
  Concept drift: Rolling MAE > threshold · baseline_MAE
  Both:          Combined monitoring (data + performance)
```

---

## 2. PSI — Population Stability Index

### 2.1 Formula

```
PSI = Σᵢ (pᵢ - qᵢ) · ln(pᵢ / qᵢ)

Where:
  pᵢ = fraction of training data in bucket i
  qᵢ = fraction of serving data in bucket i
  Buckets: typically 10 equal-frequency bins from training distribution

PSI interpretation:
  < 0.1:  No significant shift — model still valid
  0.1–0.25: Moderate shift — monitor closely, consider retraining
  > 0.25: Major shift — retrain immediately

PSI advantages:
  ✅ Single number per feature (easy to alert on)
  ✅ Directional: pᵢ > qᵢ vs. qᵢ > pᵢ
  ✅ Well-established in financial services (credit scoring)
  ✅ Works on both continuous and categorical features
```

### 2.2 Implementation

```python
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute Population Stability Index (PSI).

    Parameters
    ----------
    reference : 1D array from training/reference distribution
    current   : 1D array from serving/current distribution
    n_bins    : number of buckets (10 is standard)
    epsilon   : smoothing to avoid log(0)

    Returns
    -------
    psi : float — 0 = no drift, >0.25 = significant drift
    """
    reference = np.asarray(reference, float)
    current   = np.asarray(current,   float)

    # Define bins using training distribution quantiles (equal-frequency)
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    bin_edges[0]  -= 1e-10   # include minimum
    bin_edges[-1] += 1e-10   # include maximum

    # Proportions in each bin
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current,   bins=bin_edges)

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    # Smooth to avoid log(0)
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    psi = float(np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct)))
    return psi


def psi_interpretation(psi: float) -> str:
    if psi < 0.1:
        return "✅ No significant drift"
    elif psi < 0.25:
        return "⚠️ Moderate drift — monitor closely"
    else:
        return "🔴 Major drift — retrain required"


def compute_feature_psi(
    train_df: pd.DataFrame,
    serving_df: pd.DataFrame,
    feature_cols: list,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute PSI for all features.

    Returns
    -------
    DataFrame with feature, psi, interpretation, severity
    """
    results = []
    for col in feature_cols:
        if col not in train_df.columns or col not in serving_df.columns:
            continue
        ref  = train_df[col].dropna().values
        cur  = serving_df[col].dropna().values
        if len(ref) < 10 or len(cur) < 10:
            continue

        psi    = compute_psi(ref, cur, n_bins)
        interp = psi_interpretation(psi)
        sev    = "red" if psi >= 0.25 else ("yellow" if psi >= 0.1 else "green")

        results.append({
            "feature":        col,
            "psi":            round(psi, 5),
            "interpretation": interp,
            "severity":       sev,
        })

    df = pd.DataFrame(results).sort_values("psi", ascending=False)
    return df.reset_index(drop=True)
```

---

## 3. KS Test and Wasserstein Distance

### 3.1 Kolmogorov-Smirnov Test

```
KS two-sample test:
  H₀: reference and current come from the same distribution
  H₁: distributions are different

  Statistic D = max|F₁(x) - F₂(x)|  (max difference of CDFs)
  p-value < 0.05 → reject H₀ → distribution has shifted

  Advantages:
    ✅ Non-parametric (no Gaussian assumption)
    ✅ Well-calibrated p-value
    ✅ Standard and interpretable
  Disadvantages:
    ❌ Sensitive to sample size (large N → always significant)
    ❌ No magnitude (D statistic hard to interpret on its own)
```

### 3.2 Wasserstein Distance (Earth Mover's Distance)

```
W₁(P, Q) = area between CDFs of P and Q

Advantages:
  ✅ Magnitude is meaningful (in original feature units)
  ✅ More robust than KS for heavy-tailed distributions
  ✅ Works even for non-overlapping distributions
  ✅ No p-value threshold needed (monitor trend over time)
```

```python
def comprehensive_drift_report(
    train_df: pd.DataFrame,
    serving_df: pd.DataFrame,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Full drift report: PSI + KS test + Wasserstein distance per feature.

    Parameters
    ----------
    train_df   : reference (training) distribution
    serving_df : current serving distribution
    feature_cols: continuous features to test

    Returns
    -------
    DataFrame with drift metrics per feature
    """
    from scipy.stats import ks_2samp, wasserstein_distance

    results = []
    for col in feature_cols:
        if col not in train_df.columns or col not in serving_df.columns:
            continue
        ref = train_df[col].dropna().values
        cur = serving_df[col].dropna().values
        if len(ref) < 10 or len(cur) < 10:
            continue

        psi = compute_psi(ref, cur)
        ks_stat, ks_pval = ks_2samp(ref, cur)
        wass = wasserstein_distance(ref, cur)

        # Normalize Wasserstein by training std (scale-invariant comparison)
        ref_std = ref.std() + 1e-12
        wass_norm = wass / ref_std

        results.append({
            "feature":        col,
            "psi":            round(psi, 5),
            "ks_stat":        round(ks_stat, 4),
            "ks_pval":        round(ks_pval, 5),
            "wasserstein":    round(wass_norm, 4),
            "drift_detected": (psi >= 0.1) or (ks_pval < 0.05),
            "ref_mean":       round(ref.mean(), 3),
            "cur_mean":       round(cur.mean(), 3),
            "ref_std":        round(ref.std(), 3),
            "cur_std":        round(cur.std(), 3),
        })

    df = pd.DataFrame(results).sort_values("psi", ascending=False)
    n_drifted = df["drift_detected"].sum()
    print(f"Drift detected in {n_drifted}/{len(df)} features")
    return df.reset_index(drop=True)
```

---

## 4. Concept Drift — Rolling Performance Monitoring

### 4.1 Rolling Error Tracker

```python
import numpy as np
import pandas as pd
from collections import deque

class ConceptDriftMonitor:
    """
    Monitor model performance over time to detect concept drift.

    Tracks rolling MAE compared to baseline MAE.
    Alerts when rolling_mae / baseline_mae > threshold.
    """

    def __init__(
        self,
        baseline_mae: float,
        window: int = 30,             # rolling window in time steps
        drift_threshold: float = 1.5, # alert if rolling > 1.5x baseline
        recovery_threshold: float = 1.2,
    ):
        self.baseline_mae       = baseline_mae
        self.window             = window
        self.drift_threshold    = drift_threshold
        self.recovery_threshold = recovery_threshold
        self._errors            = deque(maxlen=window)
        self._timestamps        = deque(maxlen=window)
        self._in_drift          = False
        self.drift_events_      = []
        self.history_           = []

    def update(self, timestamp, actual: float, predicted: float) -> dict:
        """Process one new prediction-actual pair."""
        error = abs(actual - predicted)
        self._errors.append(error)
        self._timestamps.append(timestamp)

        rolling_mae = float(np.mean(self._errors))
        ratio       = rolling_mae / (self.baseline_mae + 1e-12)

        # Drift detection
        new_drift = ratio > self.drift_threshold
        if new_drift and not self._in_drift:
            self._in_drift = True
            self.drift_events_.append({
                "type":        "drift_start",
                "timestamp":   timestamp,
                "rolling_mae": rolling_mae,
                "ratio":       ratio,
            })
        elif not new_drift and self._in_drift and ratio < self.recovery_threshold:
            self._in_drift = False
            self.drift_events_.append({
                "type":        "drift_end",
                "timestamp":   timestamp,
                "rolling_mae": rolling_mae,
            })

        result = {
            "timestamp":   timestamp,
            "actual":      actual,
            "predicted":   predicted,
            "error":       error,
            "rolling_mae": rolling_mae,
            "ratio":       ratio,
            "in_drift":    self._in_drift,
        }
        self.history_.append(result)
        return result

    def get_report(self) -> dict:
        """Summary of monitoring period."""
        if not self.history_:
            return {}
        hist = pd.DataFrame(self.history_)
        return {
            "n_observations":  len(hist),
            "baseline_mae":    self.baseline_mae,
            "current_mae":     round(hist["rolling_mae"].iloc[-1], 4),
            "peak_mae":        round(hist["rolling_mae"].max(), 4),
            "n_drift_events":  len([e for e in self.drift_events_ if e["type"] == "drift_start"]),
            "pct_in_drift":    round(100 * hist["in_drift"].mean(), 2),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history_)
```

---

## 5. CUSUM for Performance Monitoring

### 5.1 CUSUM Applied to Errors

```python
class CUSUMPerformanceMonitor:
    """
    CUSUM control chart applied to model prediction errors.
    More sensitive to gradual drift than threshold-based monitors.
    """

    def __init__(
        self,
        baseline_mae: float,
        k: float = 0.5,      # allowance parameter (σ multiples)
        h: float = 5.0,      # decision interval (σ multiples)
        baseline_std: float = None,
    ):
        self.baseline_mae = baseline_mae
        self.baseline_std = baseline_std or baseline_mae * 0.5  # estimate if unknown
        self.k = k
        self.h = h
        self._S_pos = 0.0
        self._S_neg = 0.0
        self.history_ = []

    def update(self, timestamp, error: float) -> dict:
        """Process one prediction error."""
        # Standardize error relative to baseline
        z = (error - self.baseline_mae) / (self.baseline_std + 1e-12)

        self._S_pos = max(0, self._S_pos + z - self.k)
        self._S_neg = max(0, self._S_neg - z - self.k)

        alert = (self._S_pos > self.h) or (self._S_neg > self.h)

        result = {
            "timestamp": timestamp,
            "error":     error,
            "z":         round(z, 4),
            "S_pos":     round(self._S_pos, 4),
            "S_neg":     round(self._S_neg, 4),
            "alert":     alert,
        }
        self.history_.append(result)
        return result

    def reset(self):
        """Reset CUSUM accumulators after model retrain."""
        self._S_pos = 0.0
        self._S_neg = 0.0
        print("CUSUM reset after model retrain")
```

---

## 6. Evidently AI Integration

```python
def run_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_col: str = "actual",
    prediction_col: str = "predicted",
    feature_cols: list = None,
    output_path: str = "monitoring_report.html",
) -> None:
    """
    Generate comprehensive drift + model performance report using Evidently.

    pip install evidently

    Parameters
    ----------
    reference_df  : training/reference period data
    current_df    : current serving period data
    target_col    : actual label column
    prediction_col: model prediction column
    feature_cols  : input feature columns to check for drift
    output_path   : HTML report output path
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import (
            DataDriftPreset, RegressionPreset, TargetDriftPreset
        )
        from evidently.metrics import (
            DatasetDriftMetric, DataDriftTable,
            RegressionQualityMetric,
        )

        report = Report(metrics=[
            DataDriftPreset(),          # PSI + KS for all features
            TargetDriftPreset(),        # target distribution shift
            RegressionPreset(),         # MAE, RMSE, MAPE, residual plots
        ])

        report.run(reference_data=reference_df, current_data=current_df)
        report.save_html(output_path)
        print(f"Evidently report saved: {output_path}")

        # Extract summary
        result    = report.as_dict()
        drift_res = result["metrics"][0]["result"]
        n_drifted = drift_res.get("number_of_drifted_columns", 0)
        n_total   = drift_res.get("number_of_columns", 1)
        print(f"Data drift: {n_drifted}/{n_total} features drifted")

    except ImportError:
        print("Evidently not installed: pip install evidently")
        print("Falling back to manual drift computation...")
        if feature_cols:
            psi_df = compute_feature_psi(reference_df, current_df, feature_cols)
            print(psi_df.to_string(index=False))
```

---

## 7. Alerting and Response Playbook

### 7.1 Alert Thresholds

```python
from dataclasses import dataclass, field
from typing import Callable, List

@dataclass
class MonitoringAlert:
    """Configuration for a monitoring alert rule."""
    name:        str
    metric:      str
    threshold:   float
    comparator:  str = "gt"     # "gt", "lt", "gte", "lte"
    severity:    str = "warning" # "warning", "critical"
    cooldown_h:  int = 4         # minimum hours between alerts
    action:      str = "notify"  # "notify", "retrain", "rollback"

    def check(self, value: float) -> bool:
        ops = {"gt": value > self.threshold, "lt": value < self.threshold,
               "gte": value >= self.threshold, "lte": value <= self.threshold}
        return ops.get(self.comparator, False)


class AlertManager:
    """
    Evaluate monitoring metrics against alert rules and dispatch actions.
    """

    STANDARD_ALERTS = [
        MonitoringAlert("psi_critical",     "max_feature_psi",  0.25,  "gt", "critical", 2, "retrain"),
        MonitoringAlert("psi_warning",      "max_feature_psi",  0.10,  "gt", "warning",  24, "notify"),
        MonitoringAlert("mae_drift",        "rolling_mae_ratio", 1.5,  "gt", "critical", 4, "retrain"),
        MonitoringAlert("mae_warning",      "rolling_mae_ratio", 1.2,  "gt", "warning",  12, "notify"),
        MonitoringAlert("missing_data",     "missing_pct",       0.20, "gt", "warning",  1, "notify"),
        MonitoringAlert("cusum_alert",      "cusum_alarm",       1.0,  "gte","critical", 1, "retrain"),
    ]

    def __init__(self, alerts: list = None, notifier=None):
        self.alerts   = alerts or self.STANDARD_ALERTS
        self.notifier = notifier or (lambda msg: print(f"ALERT: {msg}"))

    def evaluate(self, metrics: dict) -> list:
        """
        Evaluate all alert rules against current metrics.

        Parameters
        ----------
        metrics : dict of metric_name → value

        Returns
        -------
        triggered : list of triggered MonitoringAlert objects
        """
        triggered = []
        for alert in self.alerts:
            if alert.metric not in metrics:
                continue
            if alert.check(metrics[alert.metric]):
                triggered.append(alert)
                msg = (f"[{alert.severity.upper()}] {alert.name} "
                       f"— {alert.metric}={metrics[alert.metric]:.4f} "
                       f"(threshold: {alert.threshold}) "
                       f"→ Action: {alert.action}")
                self.notifier(msg)

        return triggered
```

### 7.2 Response Playbook

```
ALERT: PSI > 0.25 (data drift — critical)
  1. Identify which features have highest PSI (drift report)
  2. Check if data pipeline is functioning correctly
  3. If pipeline OK → schedule immediate retraining
  4. If pipeline broken → page on-call engineer
  5. Consider whether feature definitions need updating

ALERT: Rolling MAE > 1.5x baseline (concept drift — critical)
  1. Inspect model predictions vs. actuals (look for systematic bias)
  2. Check if new events occurred (holidays, policy changes, etc.)
  3. Trigger retraining with recent data window
  4. If retraining takes > 4h → consider fallback to simpler model
  5. After retraining: reset CUSUM accumulators

ALERT: PSI 0.1-0.25 (data drift — warning)
  1. Log for trending
  2. Review feature distributions weekly
  3. Schedule preemptive retraining if trending upward

ALERT: Missing data > 20%
  1. Check upstream data pipeline
  2. Do NOT retrain on corrupt data
  3. Use last-known-good features for serving
  4. Alert data engineering team
```

---

## 8. Production Monitoring Dashboard

```python
def generate_monitoring_summary(
    drift_df: pd.DataFrame,
    perf_monitor: ConceptDriftMonitor,
    cusum_monitor: CUSUMPerformanceMonitor,
    alert_manager: AlertManager,
) -> dict:
    """
    Compile full monitoring summary for dashboard display.

    Returns
    -------
    summary dict ready for display or alert evaluation
    """
    perf_report = perf_monitor.get_report()

    # Aggregate metrics
    metrics = {
        "max_feature_psi":    float(drift_df["psi"].max()) if len(drift_df) > 0 else 0.0,
        "n_drifted_features": int(drift_df["drift_detected"].sum()) if len(drift_df) > 0 else 0,
        "rolling_mae":        perf_report.get("current_mae", 0),
        "rolling_mae_ratio":  perf_report.get("current_mae", 0) / (perf_monitor.baseline_mae + 1e-12),
        "pct_in_drift":       perf_report.get("pct_in_drift", 0),
        "cusum_alarm":        int(any(r["alert"] for r in cusum_monitor.history_[-10:])),
    }

    # Evaluate alerts
    triggered = alert_manager.evaluate(metrics)

    summary = {
        "metrics":        metrics,
        "n_alerts":       len(triggered),
        "alert_severity": "critical" if any(a.severity == "critical" for a in triggered) else
                          "warning"  if triggered else "ok",
        "actions_needed": list(set(a.action for a in triggered)),
        "top_drifted":    drift_df.head(5).to_dict("records") if len(drift_df) > 0 else [],
    }

    print(f"\n{'='*50}")
    print(f"MONITORING SUMMARY")
    print(f"{'='*50}")
    print(f"Status:          {summary['alert_severity'].upper()}")
    print(f"Alerts fired:    {summary['n_alerts']}")
    print(f"Max feature PSI: {metrics['max_feature_psi']:.4f}")
    print(f"Rolling MAE:     {metrics['rolling_mae']:.4f} ({metrics['rolling_mae_ratio']:.2f}x baseline)")
    if summary["actions_needed"]:
        print(f"Actions needed:  {', '.join(summary['actions_needed'])}")

    return summary
```

---

*← [03 — Model Registry](./03_model_registry_and_versioning.md) | [Module README](./README.md) | Next: [05 — Retraining](./05_retraining_strategies.md) →*
