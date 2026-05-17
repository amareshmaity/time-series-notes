# 01 — Time Series Pipeline Architecture

> **Module**: 11 Production & MLOps | **File**: 1 of 6
>
> A production time series ML system is not just a trained model — it is an interconnected set of stages that must run reliably, reproducibly, and with strict temporal discipline. This note describes the canonical architecture, its failure modes, and how to design each stage for production.

---

## Table of Contents

1. [The Full ML Pipeline for Time Series](#1-the-full-ml-pipeline-for-time-series)
2. [Ingestion Layer](#2-ingestion-layer)
3. [Feature Engineering Layer](#3-feature-engineering-layer)
4. [Training Layer](#4-training-layer)
5. [Evaluation and Gating](#5-evaluation-and-gating)
6. [Serving Layer](#6-serving-layer)
7. [Monitoring Layer](#7-monitoring-layer)
8. [Orchestration and Scheduling](#8-orchestration-and-scheduling)
9. [Pipeline Anti-Patterns](#9-pipeline-anti-patterns)

---

## 1. The Full ML Pipeline for Time Series

### 1.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION TS ML PIPELINE                            │
├──────────────┬──────────────────┬───────────────┬────────────────────── │
│  RAW DATA    │  FEATURE STORE   │  MODEL STORE  │  SERVING + MONITOR   │
├──────────────┼──────────────────┼───────────────┼──────────────────────┤
│  Ingestion   │  Feature         │  Training     │  Serving             │
│  ↓           │  Engineering     │  ↓            │  ↓                   │
│  Validation  │  ↓               │  Evaluation   │  Caching             │
│  ↓           │  Feature Store   │  ↓            │  ↓                   │
│  Storage     │  (PIT correct)   │  Registry     │  Monitoring          │
│              │                  │  ↓            │  ↓                   │
│              │                  │  Promotion    │  Drift Detection     │
│              │                  │               │  ↓                   │
│              │                  │               │  Retraining Trigger  │
└──────────────┴──────────────────┴───────────────┴──────────────────────┘

Critical property: The feature engineering at TRAINING time must
exactly mirror the feature engineering at SERVING time.
Any discrepancy = training-serving skew → silent accuracy loss.
```

### 1.2 The 6 Pipeline Stages

```
1. INGEST:   Pull raw data from sources (databases, APIs, streams)
             Validate schema, data types, expected ranges

2. ENGINEER: Compute features with point-in-time correctness
             Store in offline/online feature store

3. TRAIN:    Walk-forward cross-validation, hyperparameter tuning
             Log all parameters, metrics, artifacts to MLflow

4. EVALUATE: Gate model on holdout test set
             Must beat current production model threshold to promote

5. SERVE:    FastAPI endpoint, batch jobs, or streaming inference
             Pre-compute where possible, cache aggressively

6. MONITOR:  Track data drift (input distribution)
             Track concept drift (prediction error over time)
             Alert → trigger retraining pipeline
```

---

## 2. Ingestion Layer

### 2.1 Data Sources

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

class TSIngestionPipeline:
    """
    Production time series data ingestion with validation.

    Responsibilities:
      - Pull raw data from configured source
      - Validate schema, timestamps, expected value ranges
      - Handle missing data and duplicates
      - Store to raw data store (parquet / database)
    """

    def __init__(
        self,
        source: str,           # "sql", "parquet", "api", "kafka"
        entity_col: str = "entity_id",
        timestamp_col: str = "timestamp",
        value_cols: list = None,
        expected_freq: str = "D",   # pandas offset string
        value_range: tuple = None,  # (min, max) expected values
    ):
        self.source        = source
        self.entity_col    = entity_col
        self.timestamp_col = timestamp_col
        self.value_cols    = value_cols or ["value"]
        self.expected_freq = expected_freq
        self.value_range   = value_range

    def ingest(
        self,
        start: datetime,
        end: datetime,
        **source_kwargs,
    ) -> pd.DataFrame:
        """Pull and validate raw data for the given time window."""
        raw = self._fetch(start, end, **source_kwargs)
        self._validate(raw)
        return raw

    def _fetch(self, start, end, **kwargs) -> pd.DataFrame:
        """Fetch data — override for specific sources."""
        if self.source == "parquet":
            path = kwargs.get("path", "data/raw.parquet")
            df   = pd.read_parquet(path)
            mask = (df[self.timestamp_col] >= start) & (df[self.timestamp_col] <= end)
            return df[mask].copy()
        elif self.source == "sql":
            import sqlalchemy as sa
            engine = sa.create_engine(kwargs["connection_string"])
            query  = f"""
                SELECT * FROM {kwargs['table']}
                WHERE {self.timestamp_col} BETWEEN '{start}' AND '{end}'
            """
            return pd.read_sql(query, engine)
        else:
            raise NotImplementedError(f"Source '{self.source}' not implemented")

    def _validate(self, df: pd.DataFrame) -> None:
        """Fail fast on data quality issues."""
        errors = []

        # Schema check
        required_cols = [self.timestamp_col] + self.value_cols
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")

        # Timestamp parsing
        if df[self.timestamp_col].dtype == object:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])

        # Duplicate timestamps per entity
        if self.entity_col in df.columns:
            dup_count = df.duplicated([self.entity_col, self.timestamp_col]).sum()
        else:
            dup_count = df.duplicated([self.timestamp_col]).sum()
        if dup_count > 0:
            errors.append(f"Duplicate timestamps: {dup_count} rows")

        # Value range check
        if self.value_range:
            lo, hi = self.value_range
            for col in self.value_cols:
                if col in df.columns:
                    out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
                    if out_of_range > 0:
                        errors.append(f"Column '{col}': {out_of_range} values outside [{lo}, {hi}]")

        # Completeness check
        null_counts = df[self.value_cols].isnull().sum()
        null_pct    = null_counts / len(df) * 100
        for col in self.value_cols:
            if col in df.columns and null_pct[col] > 10:
                errors.append(f"Column '{col}': {null_pct[col]:.1f}% missing (threshold: 10%)")

        if errors:
            raise ValueError("Data validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        print(f"Data validation passed: {len(df)} rows, "
              f"{df[self.timestamp_col].min()} → {df[self.timestamp_col].max()}")
```

### 2.2 Data Quality Checks

```python
def check_temporal_completeness(
    df: pd.DataFrame,
    timestamp_col: str,
    expected_freq: str,
    entity_col: Optional[str] = None,
) -> dict:
    """
    Check for gaps in the time series.

    Returns
    -------
    report : dict with gap count, gap locations, completeness percentage
    """
    reports = {}

    entities = df[entity_col].unique() if entity_col and entity_col in df.columns else [None]

    for entity in entities:
        sub = df[df[entity_col] == entity] if entity else df
        ts  = pd.to_datetime(sub[timestamp_col]).sort_values()

        expected = pd.date_range(ts.min(), ts.max(), freq=expected_freq)
        actual   = set(ts)
        gaps     = [t for t in expected if t not in actual]

        pct_complete = 100 * (1 - len(gaps) / len(expected)) if len(expected) > 0 else 0
        reports[entity or "all"] = {
            "n_expected":    len(expected),
            "n_actual":      len(sub),
            "n_gaps":        len(gaps),
            "pct_complete":  round(pct_complete, 2),
            "first_gap":     gaps[0] if gaps else None,
            "last_gap":      gaps[-1] if gaps else None,
        }

    return reports
```

---

## 3. Feature Engineering Layer

### 3.1 The Point-in-Time Correctness Problem

```
Problem: When computing features for a training sample at time t,
         you MUST only use data available at or before time t.

Common leakage violations:
  ✗ Using future_price.shift(-1) as a feature (direct label leakage)
  ✗ Scaling with StandardScaler fit on the full dataset
    (future statistics leak into past features)
  ✗ Rolling mean with future data: window centered at t uses t+1, t+2...
  ✗ Target encoding computed on the full training set

Correct approach:
  ✓ Only look BACKWARD from t: lag, rolling_mean(lookback window)
  ✓ Fit all scalers on TRAINING data only
  ✓ Use one-sided rolling windows (closed="left" or shift forward by 1)
  ✓ Point-in-time correct join with feature store snapshots
```

### 3.2 Sklearn-Compatible TS Transformer

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TSFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible TS feature transformer.
    Ensures point-in-time correct feature computation.

    Parameters
    ----------
    lags          : lag periods (all look backward → no leakage)
    rolling_windows: rolling stat window sizes
    include_calendar: add hour/day/month/is_weekend features
    """

    def __init__(
        self,
        lags: list = None,
        rolling_windows: list = None,
        include_calendar: bool = False,
    ):
        self.lags             = lags or [1, 2, 3, 7, 14]
        self.rolling_windows  = rolling_windows or [7, 30]
        self.include_calendar = include_calendar

    def fit(self, X, y=None):
        """Nothing to fit — all features are computed deterministically."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        X must be a pd.DataFrame with:
          - DatetimeIndex (or 'timestamp' column)
          - 'value' column (or specify)

        Returns feature DataFrame (drops NaN rows from window heads).
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            X = X.set_index("timestamp")

        s    = X["value"]
        feat = pd.DataFrame(index=X.index)
        feat["value"] = s.values

        # Lag features — strictly backward looking
        for lag in self.lags:
            feat[f"lag_{lag}"] = s.shift(lag).values

        # Difference features
        feat["diff_1"] = s.diff(1).values
        feat["diff_7"] = s.diff(7).values

        # Rolling statistics — min_periods ensures no future leakage
        for w in self.rolling_windows:
            # shift(1) ensures we don't include current value (would be available at t+1)
            rol = s.shift(1).rolling(w, min_periods=max(2, w // 4))
            feat[f"roll_mean_{w}"] = rol.mean().values
            feat[f"roll_std_{w}"]  = rol.std().values
            feat[f"roll_min_{w}"]  = rol.min().values
            feat[f"roll_max_{w}"]  = rol.max().values

        # Calendar features (always available at prediction time)
        if self.include_calendar and hasattr(X.index, "hour"):
            feat["hour"]        = X.index.hour
            feat["day_of_week"] = X.index.dayofweek
            feat["month"]       = X.index.month
            feat["is_weekend"]  = (X.index.dayofweek >= 5).astype(int)
            feat["week_of_year"] = X.index.isocalendar().week.astype(int)

        return feat.dropna()
```

---

## 4. Training Layer

### 4.1 Walk-Forward Training with MLflow

```python
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

class TSTrainingPipeline:
    """
    Production training pipeline with:
      - Walk-forward cross-validation (no data leakage)
      - MLflow experiment tracking
      - Model serialization and artifact logging
    """

    def __init__(
        self,
        model,
        feature_transformer: TSFeatureTransformer,
        experiment_name: str = "ts_forecasting",
        n_splits: int = 5,
        test_size: int = 30,
        gap: int = 1,    # gap between train end and test start (simulate latency)
    ):
        self.model               = model
        self.feature_transformer = feature_transformer
        self.experiment_name     = experiment_name
        self.n_splits            = n_splits
        self.test_size           = test_size
        self.gap                 = gap

    def _walk_forward_splits(self, n: int):
        """Generate (train_end, test_start, test_end) indices."""
        step     = n // (self.n_splits + 1)
        min_train = step

        splits = []
        for i in range(1, self.n_splits + 1):
            train_end  = min_train + (i - 1) * step
            test_start = train_end + self.gap
            test_end   = test_start + self.test_size
            if test_end > n:
                break
            splits.append((train_end, test_start, test_end))
        return splits

    def fit(self, X: pd.DataFrame, y: pd.Series, params: dict = None) -> dict:
        """
        Fit model with walk-forward CV, log to MLflow.

        Parameters
        ----------
        X       : feature DataFrame (from TSFeatureTransformer)
        y       : target series
        params  : model hyperparameters to log

        Returns
        -------
        metrics : dict with mean CV MAE, RMSE
        """
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run():
            # Log parameters
            if params:
                mlflow.log_params(params)
            mlflow.log_param("n_splits",  self.n_splits)
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("gap",       self.gap)

            # Walk-forward CV
            n      = len(X)
            splits = self._walk_forward_splits(n)
            cv_mae, cv_rmse = [], []

            for tr_end, te_start, te_end in splits:
                X_tr = X.iloc[:tr_end].values
                y_tr = y.iloc[:tr_end].values
                X_te = X.iloc[te_start:te_end].values
                y_te = y.iloc[te_start:te_end].values

                self.model.fit(X_tr, y_tr)
                y_hat = self.model.predict(X_te)

                mae  = float(np.abs(y_te - y_hat).mean())
                rmse = float(np.sqrt(((y_te - y_hat)**2).mean()))
                cv_mae.append(mae); cv_rmse.append(rmse)

            mean_mae  = float(np.mean(cv_mae))
            mean_rmse = float(np.mean(cv_rmse))

            mlflow.log_metric("cv_mae",  mean_mae)
            mlflow.log_metric("cv_rmse", mean_rmse)

            # Final fit on all data
            self.model.fit(X.values, y.values)
            mlflow.sklearn.log_model(self.model, "model")
            mlflow.log_param("model_class", type(self.model).__name__)

            print(f"Training complete — CV MAE: {mean_mae:.4f}, CV RMSE: {mean_rmse:.4f}")

        return {"cv_mae": mean_mae, "cv_rmse": mean_rmse}
```

---

## 5. Evaluation and Gating

### 5.1 Model Promotion Gate

```python
class ModelPromotionGate:
    """
    Evaluate a candidate model against the current production model.
    Only promotes the candidate if it beats the threshold.
    """

    def __init__(
        self,
        metric: str = "mae",
        improvement_threshold: float = 0.02,   # must improve by at least 2%
        min_test_samples: int = 100,
    ):
        self.metric               = metric
        self.improvement_threshold = improvement_threshold
        self.min_test_samples     = min_test_samples

    def evaluate(
        self,
        candidate_model,
        baseline_model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """
        Compare candidate vs. baseline on holdout test set.

        Returns decision: 'promote' or 'reject'
        """
        assert len(X_test) >= self.min_test_samples, \
            f"Test set too small: {len(X_test)} < {self.min_test_samples}"

        y_cand = candidate_model.predict(X_test)
        y_base = baseline_model.predict(X_test) if baseline_model is not None else np.full_like(y_test, y_test.mean())

        def _score(y_true, y_pred):
            mae  = float(np.abs(y_true - y_pred).mean())
            rmse = float(np.sqrt(((y_true - y_pred)**2).mean()))
            return {"mae": mae, "rmse": rmse}

        cand_metrics = _score(y_test, y_cand)
        base_metrics = _score(y_test, y_base)

        m     = self.metric
        rel_improvement = (base_metrics[m] - cand_metrics[m]) / (base_metrics[m] + 1e-12)

        decision = "promote" if rel_improvement > self.improvement_threshold else "reject"

        result = {
            "decision":        decision,
            "candidate":       cand_metrics,
            "baseline":        base_metrics,
            "rel_improvement": round(rel_improvement, 4),
            "threshold":       self.improvement_threshold,
        }
        print(f"Gate decision: {decision.upper()}")
        print(f"  Candidate {m.upper()}: {cand_metrics[m]:.4f}")
        print(f"  Baseline  {m.upper()}: {base_metrics[m]:.4f}")
        print(f"  Relative improvement: {100*rel_improvement:.2f}%")
        return result
```

---

## 6. Serving Layer

```python
# See 06_serving_ts_models.md and code/03_serving_api.py for full implementation.

# Key design decisions:
# 1. Serve from cached forecasts when possible (pre-compute daily batch)
# 2. Real-time endpoint for on-demand requests (with 99th-pct latency SLA)
# 3. Feature computation must match training feature transformer exactly
# 4. Model version pinned per endpoint deployment

class TSServingConfig:
    """Configuration for model serving strategy."""
    mode: str = "batch"        # "batch" | "realtime" | "hybrid"
    cache_ttl_seconds: int = 3600
    max_horizon: int = 30
    latency_slo_ms: int = 200   # 200ms P99 latency target
    model_version: str = "production"
```

---

## 7. Monitoring Layer

```python
# See 04_drift_detection_and_monitoring.md and code/02_drift_detection.py

class MonitoringConfig:
    """
    What to monitor in production:

    DATA DRIFT:
      - Input feature distributions (PSI, KS test, Wasserstein distance)
      - Missing data rate per feature
      - Value range violations

    CONCEPT DRIFT (model staleness):
      - Rolling MAE over last N days vs. baseline
      - Coverage rate for prediction intervals
      - Alert if rolling_mae > baseline_mae * drift_threshold

    INFRASTRUCTURE:
      - Prediction latency (P50, P95, P99)
      - Request throughput
      - Error rate
    """
    drift_window_days: int = 7
    drift_threshold: float = 1.5    # alert if mae > 1.5x baseline
    psi_threshold: float = 0.25     # PSI > 0.25 = significant drift
    alert_channel: str = "slack"    # "slack" | "pagerduty" | "email"
```

---

## 8. Orchestration and Scheduling

### 8.1 Pipeline Orchestrators

```
Apache Airflow:        Industry standard, complex DAGs, rich UI
Prefect:               Python-native, easier setup, cloud-friendly
Metaflow:              Netflix-origin, optimized for ML workflows
GitHub Actions:        Simple cron-based retraining for small systems
Kubernetes CronJobs:   Container-native scheduling

Recommended for TS MLOps:
  Small team:          Prefect + MLflow + FastAPI
  Enterprise:          Airflow + MLflow + Kubernetes serving
```

### 8.2 DAG Structure

```python
# Prefect-style DAG for daily retraining
from datetime import datetime

def daily_ts_retrain_dag():
    """
    Daily retraining pipeline DAG structure:

    1. ingest_raw_data()          → pull yesterday's data
    2. validate_data_quality()    → fail fast on bad data
    3. compute_features()         → feature engineering (PIT correct)
    4. retrain_model()            → walk-forward CV + MLflow logging
    5. evaluate_against_champion() → promotion gate
    6. if promoted:
         deploy_model()           → update serving endpoint
         update_feature_cache()   → warm prediction cache
    7. always:
         run_drift_checks()       → PSI + rolling MAE
         send_monitoring_report()  → Slack/email
    """
    pass
```

---

## 9. Pipeline Anti-Patterns

```
❌ ANTI-PATTERN 1: Training-Serving Skew
  Training: uses normalized features with fit on full train set
  Serving:  recomputes features with different normalization
  → Silent accuracy loss; model never flags this

  Fix: serialize the full pipeline (transformer + model) together

❌ ANTI-PATTERN 2: No Gap Between Train and Test
  Training ends at 2024-12-31, test starts at 2025-01-01
  In reality, predictions are made with 24h+ data latency
  → Optimistic test metrics; real production worse

  Fix: add gap = max(prediction_latency, feature_computation_time)

❌ ANTI-PATTERN 3: Refit Scaler on Full Dataset
  StandardScaler.fit(X_full) → mean/std computed using future data
  → Normalizes training features using future distribution
  → All features leak future statistical properties

  Fix: StandardScaler.fit(X_train_only)

❌ ANTI-PATTERN 4: Not Versioning Models
  "The model in production" — which one? trained when? on what data?
  → Impossible to reproduce or roll back

  Fix: MLflow model registry with run_id, data hash, git commit

❌ ANTI-PATTERN 5: Monitoring Only Accuracy
  Accuracy looks fine, but input features silently drift
  → Model performance will degrade but no early warning

  Fix: Monitor both data drift (PSI) AND performance (rolling MAE)

❌ ANTI-PATTERN 6: Stateless Serving for Stateful Models
  LSTM/ARIMA carry state (h_t, residuals)
  Serving treats each request as independent → breaks predictions

  Fix: Maintain per-entity state in Redis/in-memory store
```

---

*← [Module README](./README.md) | Next: [02 — Feature Stores](./02_feature_stores_for_ts.md) →*
