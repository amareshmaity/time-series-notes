# 03 — Model Registry and Versioning

> **Module**: 11 Production & MLOps | **File**: 3 of 6
>
> A model registry provides the single source of truth for every model version: what it was trained on, how it performed, when it was deployed, and who approved it. MLflow is the industry standard. This note covers experiment tracking, the model registry lifecycle, and reproducibility patterns for time series models.

---

## Table of Contents

1. [Why Model Versioning?](#1-why-model-versioning)
2. [MLflow Experiment Tracking](#2-mlflow-experiment-tracking)
3. [MLflow Model Registry Lifecycle](#3-mlflow-model-registry-lifecycle)
4. [Logging Time Series Artifacts](#4-logging-time-series-artifacts)
5. [Model Reproducibility Checklist](#5-model-reproducibility-checklist)
6. [Comparing Experiments and Promoting Models](#6-comparing-experiments-and-promoting-models)
7. [Production Registry Patterns](#7-production-registry-patterns)

---

## 1. Why Model Versioning?

### 1.1 The Reproducibility Problem

```
Without versioning:

  Week 1: Train model → MAE = 12.3 → deploy
  Week 3: Retrain → MAE = 15.1 → "why is it worse?"
           - Same code? (git state not recorded)
           - Same data? (training window not recorded)
           - Same hyperparameters? (not recorded)
           - Same preprocessing? (sklearn version different?)
           → Impossible to reproduce or debug

With MLflow:

  Every run records:
    - Git commit hash
    - Training data date range + row counts
    - All hyperparameters
    - All metrics (train, val, test)
    - Serialized model + preprocessing pipeline
    - Python environment (requirements.txt / conda.yaml)
    → Any run is fully reproducible from a single run_id
```

### 1.2 Model Registry Stages

```
Stages (MLflow model lifecycle):
  None       → model registered but not reviewed
  Staging    → candidate model under evaluation
  Production → currently serving traffic
  Archived   → superseded, kept for rollback

Promotion flow:
  Train → Log → Register(None) → Test → Staging → QA → Production
                                                       ↓
                                                    Monitor
                                                       ↓
                                              Drift → Retrain
                                                       ↓
                                            New model → Production
                                         Old model → Archived
```

---

## 2. MLflow Experiment Tracking

### 2.1 Core Tracking API

```python
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess, json, os

class TSExperimentTracker:
    """
    MLflow-based experiment tracker for time series models.

    Logs:
      - All hyperparameters
      - Walk-forward CV metrics (per fold + summary)
      - Training data metadata (date range, entity count, row count)
      - Model artifact + feature transformer
      - Git commit hash for reproducibility
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "mlruns",
    ):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    @staticmethod
    def _git_hash() -> str:
        """Get current git commit hash for reproducibility."""
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()[:8]
        except Exception:
            return "unknown"

    def run(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: dict,
        data_metadata: dict,
        run_name: str = None,
        tags: dict = None,
    ) -> str:
        """
        Execute and log one training run.

        Parameters
        ----------
        model         : sklearn-compatible model
        X_train/y_train: training arrays
        X_val/y_val   : validation arrays
        params        : hyperparameters to log
        data_metadata : {train_start, train_end, n_entities, n_rows}
        run_name      : optional human-readable name
        tags          : additional tags

        Returns
        -------
        run_id : str — unique MLflow run identifier
        """
        run_tags = {
            "git_commit":  self._git_hash(),
            "model_class": type(model).__name__,
            **(tags or {}),
        }

        with mlflow.start_run(run_name=run_name, tags=run_tags) as run:
            # Log hyperparameters
            mlflow.log_params(params)

            # Log data metadata
            mlflow.log_params({
                "data.train_start":  str(data_metadata.get("train_start", "")),
                "data.train_end":    str(data_metadata.get("train_end", "")),
                "data.n_entities":   data_metadata.get("n_entities", 0),
                "data.n_rows":       data_metadata.get("n_rows", 0),
            })

            # Train
            model.fit(X_train, y_train)

            # Evaluate
            y_val_hat = model.predict(X_val)
            mae  = float(np.abs(y_val - y_val_hat).mean())
            rmse = float(np.sqrt(((y_val - y_val_hat)**2).mean()))
            mape = float(np.abs((y_val - y_val_hat) / (np.abs(y_val) + 1e-8)).mean())

            mlflow.log_metrics({"val_mae": mae, "val_rmse": rmse, "val_mape": mape})

            # Log model
            mlflow.sklearn.log_model(model, "model")

            run_id = run.info.run_id
            print(f"Run {run_id[:8]} — MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        return run_id

    def log_walk_forward_cv(
        self,
        fold_metrics: list,
        run_id: str = None,
    ) -> None:
        """
        Log per-fold and summary metrics from walk-forward CV.

        fold_metrics: [{"fold": 1, "mae": 12.3, "rmse": 15.1}, ...]
        """
        ctx = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run()
        with ctx:
            for fold in fold_metrics:
                mlflow.log_metrics(
                    {f"fold_{fold['fold']}_mae": fold["mae"],
                     f"fold_{fold['fold']}_rmse": fold["rmse"]},
                    step=fold["fold"],
                )
            maes  = [f["mae"] for f in fold_metrics]
            rmses = [f["rmse"] for f in fold_metrics]
            mlflow.log_metrics({
                "cv_mae_mean":  float(np.mean(maes)),
                "cv_mae_std":   float(np.std(maes)),
                "cv_rmse_mean": float(np.mean(rmses)),
                "cv_rmse_std":  float(np.std(rmses)),
            })
```

### 2.2 Logging Custom Artifacts

```python
def log_ts_artifacts(
    run_id: str,
    forecast_df: pd.DataFrame = None,
    feature_importance: dict = None,
    config: dict = None,
    plots_dir: str = None,
) -> None:
    """
    Log time series-specific artifacts to an existing MLflow run.

    Artifacts include:
      - Forecast CSV (predicted vs. actual)
      - Feature importance JSON
      - Model config (hyperparameter summary)
      - Evaluation plots (PNG files)
    """
    import tempfile, os

    with mlflow.start_run(run_id=run_id):
        with tempfile.TemporaryDirectory() as tmpdir:

            # Forecast CSV
            if forecast_df is not None:
                path = os.path.join(tmpdir, "forecast.csv")
                forecast_df.to_csv(path, index=False)
                mlflow.log_artifact(path, "forecasts")

            # Feature importance
            if feature_importance:
                path = os.path.join(tmpdir, "feature_importance.json")
                with open(path, "w") as f:
                    json.dump(feature_importance, f, indent=2)
                mlflow.log_artifact(path, "analysis")

            # Config
            if config:
                path = os.path.join(tmpdir, "config.json")
                with open(path, "w") as f:
                    json.dump(config, f, indent=2)
                mlflow.log_artifact(path)

            # All PNG plots in a directory
            if plots_dir and os.path.isdir(plots_dir):
                for png in Path(plots_dir).glob("*.png"):
                    mlflow.log_artifact(str(png), "plots")
```

---

## 3. MLflow Model Registry Lifecycle

### 3.1 Registration and Stage Management

```python
from mlflow import MlflowClient

class ModelRegistryManager:
    """
    Manage model lifecycle in MLflow Model Registry.

    Handles:
      - Registering model versions
      - Transitioning stages (None → Staging → Production → Archived)
      - Loading models by stage for serving
      - Comparing versions
    """

    def __init__(self, tracking_uri: str = "mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register(self, run_id: str, model_name: str, tags: dict = None) -> str:
        """
        Register a model from a completed run.

        Returns version number (string).
        """
        uri     = f"runs:/{run_id}/model"
        version = mlflow.register_model(uri, model_name)

        if tags:
            for k, v in tags.items():
                self.client.set_model_version_tag(model_name, version.version, k, str(v))

        print(f"Registered {model_name} v{version.version} from run {run_id[:8]}")
        return version.version

    def promote_to_staging(self, model_name: str, version: str, description: str = "") -> None:
        """Move model version to Staging for evaluation."""
        self.client.transition_model_version_stage(
            name=model_name, version=version, stage="Staging",
            archive_existing_versions=False,
        )
        if description:
            self.client.update_model_version(model_name, version, description=description)
        print(f"Promoted {model_name} v{version} to Staging")

    def promote_to_production(self, model_name: str, version: str, description: str = "") -> None:
        """
        Promote model to Production.
        Archives existing Production version automatically.
        """
        self.client.transition_model_version_stage(
            name=model_name, version=version, stage="Production",
            archive_existing_versions=True,   # auto-archive current production
        )
        if description:
            self.client.update_model_version(model_name, version, description=description)
        print(f"Promoted {model_name} v{version} to Production (previous archived)")

    def archive(self, model_name: str, version: str) -> None:
        """Archive a model version (remove from active stages)."""
        self.client.transition_model_version_stage(
            name=model_name, version=version, stage="Archived"
        )

    def load_production_model(self, model_name: str):
        """Load the current Production model for inference."""
        uri = f"models:/{model_name}/Production"
        return mlflow.sklearn.load_model(uri)

    def load_staging_model(self, model_name: str):
        """Load the Staging model for evaluation."""
        uri = f"models:/{model_name}/Staging"
        return mlflow.sklearn.load_model(uri)

    def get_version_info(self, model_name: str, version: str) -> dict:
        """Get full metadata for a specific model version."""
        mv  = self.client.get_model_version(model_name, version)
        run = self.client.get_run(mv.run_id)
        return {
            "version":    mv.version,
            "stage":      mv.current_stage,
            "run_id":     mv.run_id,
            "created":    mv.creation_timestamp,
            "params":     run.data.params,
            "metrics":    run.data.metrics,
            "tags":       {**run.data.tags, **dict(mv.tags)},
            "description":mv.description,
        }

    def list_production_history(self, model_name: str) -> list:
        """List all historical production versions of a model."""
        versions = self.client.search_model_versions(f"name='{model_name}'")
        return [
            {"version": v.version, "stage": v.current_stage, "run_id": v.run_id}
            for v in sorted(versions, key=lambda v: int(v.version))
        ]
```

---

## 4. Logging Time Series Artifacts

### 4.1 What to Log for TS Models

```
MUST LOG:
  ✅ All hyperparameters (lags, rolling windows, model params)
  ✅ Training data: start date, end date, entity count, row count
  ✅ Validation metrics: MAE, RMSE, MAPE per fold + mean/std
  ✅ Model artifact (serialized pipeline: transformer + model)
  ✅ Git commit hash
  ✅ Python environment (mlflow auto-logs conda.yaml)

SHOULD LOG:
  ✅ Forecast vs. actual plot (visual sanity check)
  ✅ Feature importance (for tree models)
  ✅ Residual plot (check for autocorrelation in errors)
  ✅ Data drift score at training time (baseline for monitoring)

NICE TO HAVE:
  ✅ Walk-forward CV plot (per-fold MAE)
  ✅ Prediction interval coverage rate
  ✅ Entity-level performance breakdown (best/worst entities)
```

### 4.2 Serializing the Full Pipeline

```python
import cloudpickle
import pickle
from pathlib import Path

class TSModelArtifact:
    """
    Serializes the complete TS model package:
    feature transformer + scaler + model + metadata.
    """

    def __init__(
        self,
        feature_transformer,
        model,
        metadata: dict,
    ):
        self.feature_transformer = feature_transformer
        self.model               = model
        self.metadata            = metadata

    def predict(self, raw_df: "pd.DataFrame") -> "pd.Series":
        """End-to-end prediction from raw input."""
        features = self.feature_transformer.transform(raw_df)
        return self.model.predict(features.values)

    def save(self, path: str) -> None:
        """Serialize full artifact to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            cloudpickle.dump(self, f)
        print(f"Artifact saved: {path}")

    @classmethod
    def load(cls, path: str) -> "TSModelArtifact":
        """Load artifact from disk."""
        with open(path, "rb") as f:
            return cloudpickle.load(f)

    def log_to_mlflow(self, artifact_name: str = "ts_artifact") -> None:
        """Log full artifact to active MLflow run."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, f"{artifact_name}.pkl")
            self.save(path)
            mlflow.log_artifact(path, artifact_name)
```

---

## 5. Model Reproducibility Checklist

```
REPRODUCIBILITY CHECKLIST
════════════════════════════════════════════════════════

Data:
  □ Training data date range logged (start, end)
  □ Data hash logged (SHA256 of training dataframe)
  □ Entity list/count logged
  □ Preprocessing steps fully serialized

Code:
  □ Git commit hash logged in MLflow tags
  □ requirements.txt / pyproject.toml committed
  □ No hardcoded paths (use config files or env variables)

Model:
  □ All hyperparameters logged (every single parameter)
  □ Random seeds logged and set explicitly
  □ Feature transformer serialized alongside model
  □ Python environment logged (conda.yaml or requirements)

Evaluation:
  □ Walk-forward fold boundaries logged (train end / test start / test end)
  □ Test set never touched until final evaluation
  □ All CV fold metrics logged (not just mean)

Deployment:
  □ Model version pinned in serving config
  □ Rollback procedure documented
  □ Previous production model archived (not deleted)
```

---

## 6. Comparing Experiments and Promoting Models

```python
def compare_experiments(
    experiment_name: str,
    metric: str = "val_mae",
    n_top: int = 5,
    tracking_uri: str = "mlruns",
) -> "pd.DataFrame":
    """
    Compare all runs in an experiment, ranked by metric.

    Returns
    -------
    DataFrame of top runs with parameters and metrics
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp    = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"Experiment '{experiment_name}' not found")
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=n_top,
    )

    rows = []
    for r in runs:
        row = {
            "run_id":    r.info.run_id[:8],
            "status":    r.info.status,
            metric:      r.data.metrics.get(metric, np.nan),
            **{f"param.{k}": v for k, v in r.data.params.items()},
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nTop {n_top} runs by {metric}:")
    print(df.to_string(index=False))
    return df


def auto_promote_best(
    experiment_name: str,
    model_name: str,
    metric: str = "val_mae",
    improvement_pct: float = 5.0,
    tracking_uri: str = "mlruns",
) -> None:
    """
    Automatically promote the best run from Staging to Production
    if it improves over current production model by improvement_pct.
    """
    registry = ModelRegistryManager(tracking_uri)

    # Get current production metrics
    try:
        prod_info = registry.get_version_info(model_name, "Production")
        prod_metric = prod_info["metrics"].get(metric, float("inf"))
    except Exception:
        prod_metric = float("inf")
        print("No current production model — promoting best run directly")

    # Get best staging model
    client    = MlflowClient()
    best_run  = client.search_runs(
        experiment_ids=[client.get_experiment_by_name(experiment_name).experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=1,
    )
    if not best_run:
        print("No runs found"); return

    best_metric = best_run[0].data.metrics.get(metric, float("inf"))
    improvement  = 100 * (prod_metric - best_metric) / (prod_metric + 1e-12)

    print(f"Production {metric}: {prod_metric:.4f}")
    print(f"Candidate  {metric}: {best_metric:.4f}")
    print(f"Improvement: {improvement:.2f}% (threshold: {improvement_pct}%)")

    if improvement >= improvement_pct:
        version = registry.register(best_run[0].info.run_id, model_name)
        registry.promote_to_production(
            model_name, version,
            description=f"Auto-promoted: {improvement:.2f}% better than previous"
        )
    else:
        print("Insufficient improvement — skipping promotion")
```

---

## 7. Production Registry Patterns

### 7.1 Shadow Deployment

```
Shadow Deployment:
  - New model runs in parallel with production
  - Predictions are computed but NOT served to users
  - Predictions + actuals logged for comparison
  - After N days, compare shadow vs. production performance
  - If shadow is better → promote to production

Benefits:
  ✅ Zero risk to users during evaluation
  ✅ Real production traffic distribution
  ✅ Can detect edge cases not seen in CV
  ✅ Allows long evaluation windows (weeks)
```

### 7.2 Canary Deployment

```
Canary Deployment:
  - Route X% of traffic to new model, (100-X)% to production
  - Monitor error rates, latency, prediction distributions
  - Gradually increase canary traffic if metrics look good
  - Rollback immediately if issues detected

Typical schedule:
  Day 1: 5% canary
  Day 3: 20% canary (if no issues)
  Day 7: 50% canary (if no issues)
  Day 14: 100% canary → full production rollout

Implementation:
  Load balancer (NGINX / Kubernetes Ingress) routes traffic
  Feature flags / experiment frameworks (LaunchDarkly) for gradual rollout
```

---

*← [02 — Feature Stores](./02_feature_stores_for_ts.md) | [Module README](./README.md) | Next: [04 — Drift Detection](./04_drift_detection_and_monitoring.md) →*
