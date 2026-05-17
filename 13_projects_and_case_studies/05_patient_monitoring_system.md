# 05 — Patient Monitoring System

> **Module**: 13 Projects & Case Studies | **Project**: 5 of 5 (Capstone)
> **Domain**: Healthcare / ICU | **Problem**: Real-time vitals monitoring, classification + anomaly detection
>
> ICU patient monitoring is the highest-stakes application of time series analysis. Vital signs must be analyzed in real-time, missing data is common, and false positives burden nurses while false negatives can be fatal. This capstone project integrates classification (deterioration detection), anomaly detection, and a real-time serving architecture.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Dataset & EDA](#2-dataset--eda)
3. [Missing Data Handling](#3-missing-data-handling)
4. [Deterioration Classification](#4-deterioration-classification)
5. [Online Anomaly Detection](#5-online-anomaly-detection)
6. [Real-Time Serving Architecture](#6-real-time-serving-architecture)
7. [Calibration & Clinical Safety](#7-calibration--clinical-safety)
8. [Key Lessons](#8-key-lessons)
9. [End-to-End Summary](#9-end-to-end-summary)

---

## 1. Problem Definition

```
Business goal:
  Early warning system for ICU deterioration.
  Alert nursing staff when a patient shows signs of clinical deterioration
  (sepsis, respiratory failure, cardiac event) > 4 hours before onset.

Dataset options:
  - PhysioNet CinC Challenge 2019: Early sepsis prediction, 40,000+ patients
  - MIMIC-III / MIMIC-IV: 50,000+ ICU admissions, comprehensive vitals
  - HIRID: High-resolution ICU data, 1-minute sampling

Variables (standard vitals):
  HR (heart rate), SBP/DBP (blood pressure), SpO2 (oxygen saturation),
  RR (respiratory rate), Temp (temperature), GCS (consciousness score)
  + Lab values: lactate, WBC, creatinine (available 4-6h apart)

KPIs:
  AUROC ≥ 0.85 for 4-hour ahead deterioration prediction
  Sensitivity ≥ 0.85 (patient safety: cannot miss critical events)
  Specificity ≥ 0.75 (operational: cannot alarm-fatigue nursing staff)
  Alert latency < 30 seconds

Key challenges:
  Missing data (60-80% of lab values missing at any given time)
  Class imbalance (5-15% deterioration events in ICU)
  Time-varying patient state (a patient can improve and deteriorate)
  Concept drift (patient population changes seasonally)
```

---

## 2. Dataset & EDA

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_icu_data(
    n_patients: int = 200,
    max_los:    int = 72,    # max length of stay (hours)
    seed:       int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic ICU vital signs dataset.
    ~20% of patients deteriorate (label=1) in second half of stay.
    Missing data rate: 40% (simulates real ICU data).
    """
    np.random.seed(seed)
    records = []

    # Vital sign parameters: (mean, std, min, max)
    VITALS = {
        "HR":   (80, 15, 40, 180),
        "SBP":  (120, 20, 60, 220),
        "SpO2": (97, 2, 70, 100),
        "RR":   (16, 4, 8, 40),
        "Temp": (37.0, 0.5, 35, 41),
        "GCS":  (14, 2, 3, 15),
    }

    for p in range(n_patients):
        los      = np.random.randint(12, max_los)
        patient_label = 1 if np.random.rand() < 0.2 else 0

        # Baseline values for this patient
        baselines = {v: np.random.normal(mu, sig*0.3)
                     for v, (mu, sig, lo, hi) in VITALS.items()}

        for t in range(los):
            row = {"patient_id": p, "hour": t, "label": patient_label}
            row["deteriorated"] = int(patient_label and t > los * 0.6)

            for v, (mu, sig, lo, hi) in VITALS.items():
                # Deterioration effect
                deteff = 0.0
                if row["deteriorated"]:
                    if v in ["HR","RR"]:       deteff = sig * 2.0
                    elif v in ["SBP","SpO2"]:  deteff = -sig * 1.5
                    elif v == "Temp":          deteff = 1.5
                    elif v == "GCS":           deteff = -2.5

                val = baselines[v] + deteff + np.random.normal(0, sig * 0.2)
                val = np.clip(val, lo, hi)

                # Inject missing data
                if np.random.rand() < 0.4:
                    val = np.nan

                row[v] = round(float(val), 1) if not np.isnan(val) else np.nan

            records.append(row)

    df = pd.DataFrame(records)
    print(f"ICU dataset: {n_patients} patients, {len(df)} hourly observations")
    print(f"Deterioration rate: {df.groupby('patient_id')['deteriorated'].max().mean()*100:.1f}%")
    print(f"Missing rate per vital: {df[list(VITALS)].isnull().mean().round(3).to_dict()}")
    return df


def icu_eda(df: pd.DataFrame) -> None:
    vital_cols = ["HR","SBP","SpO2","RR","Temp","GCS"]
    fig, axes  = plt.subplots(2, 3, figsize=(15, 8))

    for ax, col in zip(axes.flatten(), vital_cols):
        normal   = df[df["deteriorated"]==0][col].dropna()
        abnormal = df[df["deteriorated"]==1][col].dropna()
        ax.hist(normal,   bins=30, alpha=0.6, color="#4CAF50", label="Normal", density=True)
        ax.hist(abnormal, bins=30, alpha=0.6, color="#F44336", label="Deteriorated", density=True)
        ax.set_title(col); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle("ICU Vital Sign Distributions: Normal vs. Deteriorated", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.show()
```

---

## 3. Missing Data Handling

```python
def forward_fill_vitals(
    df: pd.DataFrame,
    vital_cols: list,
    max_fill: int = 4,
) -> pd.DataFrame:
    """
    Forward-fill missing vitals within each patient stay.
    max_fill limits propagation (stale data becomes unreliable after N hours).
    """
    df = df.sort_values(["patient_id","hour"]).copy()
    df[vital_cols] = (df.groupby("patient_id")[vital_cols]
                        .transform(lambda x: x.fillna(method="ffill", limit=max_fill)))
    return df


def indicator_imputation(
    df: pd.DataFrame,
    vital_cols: list,
    global_medians: dict = None,
) -> pd.DataFrame:
    """
    Indicator method:
      1. Add binary missingness indicator per vital column.
      2. Impute remaining NaN with global median.
      
    This lets the model learn 'missing' as a pattern itself
    (missing SpO2 is often clinically meaningful).
    """
    df = df.copy()
    if global_medians is None:
        global_medians = df[vital_cols].median().to_dict()

    for col in vital_cols:
        df[f"{col}_missing"] = df[col].isnull().astype(float)
        df[col] = df[col].fillna(global_medians[col])

    return df, global_medians


def time_since_last_observed(
    df: pd.DataFrame,
    vital_cols: list,
) -> pd.DataFrame:
    """
    Add 'time since last observed' feature per vital per patient.
    High values → stale measurement → higher uncertainty.
    """
    df = df.sort_values(["patient_id","hour"]).copy()
    for col in vital_cols:
        def tslm(group):
            is_obs = group[col].notna()
            times  = pd.Series(group["hour"].values, index=group.index)
            result = pd.Series(np.nan, index=group.index)
            last_t = np.nan
            for idx in group.index:
                if is_obs[idx]:
                    last_t = times[idx]
                result[idx] = times[idx] - last_t if not np.isnan(last_t) else 0
            return result
        df[f"{col}_hours_since"] = df.groupby("patient_id").apply(tslm).reset_index(level=0, drop=True)
    return df
```

---

## 4. Deterioration Classification

```python
def build_patient_features(
    df: pd.DataFrame,
    vital_cols: list,
    windows: list = None,
) -> pd.DataFrame:
    """
    Per-patient, per-hour features for deterioration classification.
    All features are backward-looking relative to each hour t.
    """
    windows = windows or [3, 6, 12]
    df = df.sort_values(["patient_id","hour"]).copy()
    new_cols = {}

    # Rolling stats per vital
    for col in vital_cols:
        grp = df.groupby("patient_id")[col]
        for w in windows:
            new_cols[f"{col}_mean_{w}h"] = grp.transform(lambda x: x.rolling(w, min_periods=1).mean())
            new_cols[f"{col}_std_{w}h"]  = grp.transform(lambda x: x.rolling(w, min_periods=1).std())
            new_cols[f"{col}_min_{w}h"]  = grp.transform(lambda x: x.rolling(w, min_periods=1).min())
            new_cols[f"{col}_max_{w}h"]  = grp.transform(lambda x: x.rolling(w, min_periods=1).max())

    # Trend (slope over last 6 hours)
    for col in vital_cols:
        def slope_fn(x):
            vals = x.values
            T    = len(vals)
            t_   = np.arange(T)
            if T < 2: return pd.Series(np.zeros(T), index=x.index)
            return pd.Series(
                [float(np.polyfit(t_[max(0,i-5):i+1], vals[max(0,i-5):i+1], 1)[0])
                 for i in range(T)], index=x.index
            )
        new_cols[f"{col}_trend"] = df.groupby("patient_id")[col].transform(slope_fn)

    # Time in ICU (simple feature)
    new_cols["hours_in_icu"] = df.groupby("patient_id")["hour"].transform(lambda x: x - x.min())

    for k, v in new_cols.items():
        df[k] = v.values

    return df


def train_deterioration_classifier(
    df: pd.DataFrame,
    feature_cols: list,
    target: str = "deteriorated",
    test_patients: int = 40,
) -> dict:
    """
    Train LGBM deterioration classifier with patient-level split.
    Uses patient-level split (not random row split) to prevent leakage.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        print("LightGBM not installed"); return {}

    from sklearn.metrics import roc_auc_score, classification_report
    from sklearn.preprocessing import StandardScaler

    patients = df["patient_id"].unique()
    np.random.shuffle(patients)
    test_pids  = set(patients[-test_patients:])
    train_mask = ~df["patient_id"].isin(test_pids)
    test_mask  =  df["patient_id"].isin(test_pids)

    X_tr = df[train_mask][feature_cols].fillna(0).values
    y_tr = df[train_mask][target].values
    X_te = df[test_mask][feature_cols].fillna(0).values
    y_te = df[test_mask][target].values

    # Class weight for imbalance
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        num_leaves=31, scale_pos_weight=pos_weight,
        random_state=42, verbosity=-1,
    )
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]

    auc  = roc_auc_score(y_te, prob)
    print(f"\nDeterioration Classifier:")
    print(f"  AUC-ROC: {auc:.4f}")
    print(classification_report(y_te, (prob > 0.5).astype(int),
                                 target_names=["Normal","Deteriorated"], digits=4))
    return {"model": model, "auc": auc, "prob": prob, "y_te": y_te}
```

---

## 5. Online Anomaly Detection

```python
class OnlineVitalMonitor:
    """
    Streaming anomaly detector for ICU vitals.
    Uses exponentially-weighted statistics to adapt to patient-specific baseline.
    """

    def __init__(
        self,
        alpha: float = 0.05,     # EWMA learning rate
        z_thresh: float = 3.5,   # sigma threshold for alert
        min_obs:  int = 6,       # minimum observations before alerting
    ):
        self.alpha     = alpha
        self.z_thresh  = z_thresh
        self.min_obs   = min_obs
        self._means    = {}
        self._vars     = {}
        self._counts   = {}

    def update(self, patient_id: str, vitals: dict) -> dict:
        """
        Update running statistics for a patient and return per-vital z-scores.

        Parameters
        ----------
        patient_id : patient identifier
        vitals     : dict {vital_name: value} for current timestamp

        Returns
        -------
        dict with z_score per vital and combined alert flag
        """
        if patient_id not in self._means:
            self._means[patient_id]  = {k: v for k, v in vitals.items() if v is not None}
            self._vars[patient_id]   = {k: 1.0 for k in vitals}
            self._counts[patient_id] = 0

        self._counts[patient_id] += 1
        scores = {}
        alerts = {}

        for vital, value in vitals.items():
            if value is None: continue
            mu  = self._means[patient_id].get(vital, value)
            var = self._vars[patient_id].get(vital, 1.0)

            z = abs(value - mu) / (np.sqrt(var) + 1e-8)
            scores[vital] = round(z, 3)

            # Alert if z-score exceeds threshold after warm-up
            alerts[vital] = (z > self.z_thresh and
                              self._counts[patient_id] >= self.min_obs)

            # Update EWMA mean and variance
            self._means[patient_id][vital] = (1-self.alpha)*mu + self.alpha*value
            self._vars[patient_id][vital]  = (1-self.alpha)*var + self.alpha*(value-mu)**2

        combined_alert = any(alerts.values())
        worst_vital    = max(scores, key=scores.get) if scores else None

        return {
            "z_scores":      scores,
            "alerts":        alerts,
            "combined_alert": combined_alert,
            "worst_vital":   worst_vital,
        }
```

---

## 6. Real-Time Serving Architecture

```
REAL-TIME PATIENT MONITORING PIPELINE:

  Data Sources:
    Bedside monitors → HL7/FHIR stream → Kafka topic (1-min cadence)
    Lab results       → Hospital EMR → event-driven Kafka

  Stream Processing (Kafka Streams / Apache Flink):
    1. Parse HL7/FHIR → structured vital record
    2. Forward-fill missing values (up to 4h)
    3. Compute rolling features (3h, 6h, 12h windows)
    4. Score with: OnlineVitalMonitor (z-score, 10ms)
                   LGBM deterioration model (50ms)
    5. Combine scores → alert decision

  Alert Routing:
    CRITICAL (combined score > 0.85 + z > 4) → Page nurse + physician
    WARNING  (combined score > 0.65)          → Notify nurse station
    NORMAL                                    → Log to dashboard

  Storage:
    Patient features → Redis (TTL=24h per patient)
    Alert history    → PostgreSQL
    Raw vitals       → InfluxDB (time-series DB)

  Monitoring (model health):
    Rolling AUC on retrospective labeled events (48h lag)
    Feature drift: monitor vital distributions vs. 30-day baseline
    Alert rate: if >15% of patients alerting → threshold recalibration
```

```python
from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class PatientAlert:
    patient_id:   str
    timestamp:    str
    level:        str          # "CRITICAL", "WARNING", "NORMAL"
    trigger:      str          # "z_score", "classifier", "ensemble"
    worst_vital:  Optional[str]
    score:        float
    action:       str

class PatientMonitoringService:
    """Production-grade patient monitoring service."""

    def __init__(
        self,
        classifier,       # fitted LGBM model
        feature_cols: list,
        z_thresh_warn:   float = 3.0,
        z_thresh_crit:   float = 4.5,
        clf_thresh_warn: float = 0.60,
        clf_thresh_crit: float = 0.80,
        cooldown_min:    int   = 15,
    ):
        self.classifier    = classifier
        self.feature_cols  = feature_cols
        self.z_thresh_warn = z_thresh_warn
        self.z_thresh_crit = z_thresh_crit
        self.clf_thresh_w  = clf_thresh_warn
        self.clf_thresh_c  = clf_thresh_crit
        self.cooldown_min  = cooldown_min
        self.monitor       = OnlineVitalMonitor()
        self._last_alert   = {}   # patient_id → last alert timestamp

    def process_vital_record(
        self,
        patient_id: str,
        timestamp:  str,
        vitals:     dict,
        features:   np.ndarray,
    ) -> PatientAlert:
        """
        Process one vital sign record for a patient.
        Returns an alert (or NORMAL) with severity and action.
        """
        t0 = time.time()

        # Layer 1: Online z-score
        z_result = self.monitor.update(patient_id, vitals)

        # Layer 2: Classifier score
        clf_score = float(
            self.classifier.predict_proba(features.reshape(1,-1))[0, 1]
        )

        # Ensemble score (weighted)
        z_max    = max(z_result["z_scores"].values()) if z_result["z_scores"] else 0
        z_norm   = min(z_max / 5.0, 1.0)   # normalize to [0,1]
        ens_score = 0.4 * z_norm + 0.6 * clf_score

        # Alert routing
        in_cooldown = self._in_cooldown(patient_id)
        if ens_score >= 0.80 and not in_cooldown:
            level = "CRITICAL"; action = "page_nurse_and_physician"
        elif ens_score >= 0.55 and not in_cooldown:
            level = "WARNING";  action = "notify_nurse_station"
        else:
            level = "NORMAL";   action = "log_only"

        if level != "NORMAL":
            self._last_alert[patient_id] = time.time()

        latency_ms = (time.time() - t0) * 1000

        return PatientAlert(
            patient_id  = patient_id,
            timestamp   = timestamp,
            level       = level,
            trigger     = "ensemble",
            worst_vital = z_result.get("worst_vital"),
            score       = round(ens_score, 4),
            action      = action,
        )

    def _in_cooldown(self, patient_id: str) -> bool:
        if patient_id not in self._last_alert:
            return False
        elapsed = (time.time() - self._last_alert[patient_id]) / 60
        return elapsed < self.cooldown_min
```

---

## 7. Calibration & Clinical Safety

```python
def calibrate_classifier(
    model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "isotonic",
):
    """
    Probability calibration for clinical safety.
    
    In healthcare, a score of 0.70 MUST mean ~70% probability of deterioration.
    Uncalibrated ML models are often overconfident or underconfident.

    Methods:
      'isotonic'  : isotonic regression (more flexible, needs more data)
      'sigmoid'   : Platt scaling (fewer samples needed)
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss

    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)

    y_prob_raw = model.predict_proba(X_cal)[:, 1]
    y_prob_cal = calibrated.predict_proba(X_cal)[:, 1]

    bs_raw = brier_score_loss(y_cal, y_prob_raw)
    bs_cal = brier_score_loss(y_cal, y_prob_cal)

    print(f"Brier score — Before: {bs_raw:.4f} | After calibration: {bs_cal:.4f}")
    print(f"Lower Brier = better calibrated")
    return calibrated


def sensitivity_at_threshold(y_true, y_score, target_sensitivity=0.90):
    """Find the threshold that achieves target sensitivity."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # Find smallest threshold achieving target TPR
    valid = [(thr, tp, fp) for fp, tp, thr in zip(fpr, tpr, thresholds)
             if tp >= target_sensitivity]
    if valid:
        best = max(valid, key=lambda x: x[1] - x[2])
        print(f"Threshold for {100*target_sensitivity:.0f}% sensitivity: "
              f"{best[0]:.4f} (TPR={best[1]:.4f}, FPR={best[2]:.4f})")
        return best[0]
    return 0.5
```

---

## 8. Key Lessons

```
LESSON 1: Patient-level train/test split is mandatory.
  Row-level split leaks: the same patient's later hours appear in both
  train and test. Use patient IDs to split.

LESSON 2: Missing data is not random in ICU.
  "No SpO2 recorded" often means the nurse was managing a crisis —
  a clinically significant missingness pattern.
  Always include missingness indicators as features.

LESSON 3: Sensitivity > specificity in life-critical settings.
  The cost asymmetry is extreme: missed deterioration >> false alarm.
  Set threshold to achieve desired sensitivity (≥ 90%) first,
  then report resulting specificity to manage alarm fatigue.

LESSON 4: Calibration is non-negotiable for clinical trust.
  Clinicians interpret scores as probabilities.
  Apply isotonic regression or Platt scaling post-training.

LESSON 5: Time-to-detect matters more than AUC in practice.
  A model with AUC=0.88 that alerts 1 hour before deterioration
  is more valuable than AUC=0.92 that alerts 30 minutes after.
  Report TTD alongside AUC.

LESSON 6: Cooldown prevents alarm fatigue.
  Repeated alerts for the same patient within 15 minutes → ignored.
  Implement alert cooldown with severity-based override for CRITICAL.
```

---

## 9. End-to-End Summary

```
MODULE 13 COMPLETE — End-to-End Project Journey:

  Project 1 — Stock: Learned to set honest expectations, beat random walk,
               build calibrated prediction intervals for financial forecasting.

  Project 2 — Energy: Mastered multi-seasonality, hierarchical coherence,
               probabilistic intervals, weather exogenous integration.

  Project 3 — Retail: Built global model for 200+ series, handled intermittent
               demand with Croston, applied WRMSSE and MLflow tracking.

  Project 4 — Sensors: Designed layered anomaly detection (statistical → ML → DL),
               implemented root cause analysis, built alert pipeline with cooldown.

  Project 5 — ICU (Capstone): Integrated classification + online anomaly detection,
               handled extreme class imbalance, calibrated probabilities, designed
               real-time serving with clinical safety constraints.

UNIVERSAL PRINCIPLES ACROSS ALL PROJECTS:
  ✓ Always start with EDA and a naïve baseline
  ✓ Use walk-forward / patient-level splits — never random
  ✓ Report uncertainty (intervals, calibration) not just point metrics
  ✓ Document data lineage and model versions (MLflow / DVC)
  ✓ Design the serving architecture before the model
  ✓ Monitor in production — models degrade silently without oversight
```

---

*← [04 — Sensor Anomaly](./04_sensor_anomaly_detection.md) | [Module README](./README.md) | Back to [Master README](../README.md) →*
