# 05 — Retraining Strategies

> **Module**: 11 Production & MLOps | **File**: 5 of 6
>
> All production models degrade over time. The question is not *whether* to retrain but *when*, *how often*, and *on what data*. This note covers scheduled retraining, trigger-based pipelines, online learning, warm starting, and the trade-offs between full retrain vs. incremental update.

---

## Table of Contents

1. [Why Models Degrade](#1-why-models-degrade)
2. [Scheduled Retraining](#2-scheduled-retraining)
3. [Trigger-Based Retraining](#3-trigger-based-retraining)
4. [Warm Starting and Incremental Learning](#4-warm-starting-and-incremental-learning)
5. [Online Learning](#5-online-learning)
6. [Training Data Window Strategies](#6-training-data-window-strategies)
7. [Retraining Pipeline](#7-retraining-pipeline)

---

## 1. Why Models Degrade

### 1.1 Degradation Mechanisms

```
1. CONCEPT DRIFT
   The relationship P(Y|X) changes over time.
   Example: A pricing model becomes inaccurate as competitor prices shift.
   Cure: Full retrain on recent data with updated features.

2. DATA DISTRIBUTION SHIFT (covariate shift)
   P(X) changes — features look different from training.
   Example: New product line added → sales patterns unlike historical.
   Cure: Retrain with data that covers new distribution.

3. LABEL DRIFT
   P(Y) changes — target distribution shifts.
   Example: Overall sales volume grew 50% → model underestimates.
   Cure: Retrain with recent target-scale data.

4. STALE SEASONAL PATTERNS
   Model learned last year's seasonality, which has shifted.
   Example: Holiday shopping now starts 2 weeks earlier.
   Cure: Expand training window or add adaptive seasonality.

5. PIPELINE STALENESS
   Model was trained with feature_v1, now feature_v2 is available.
   Cure: Retrain with new features (not just same model).
```

---

## 2. Scheduled Retraining

### 2.1 When to Use

```
Scheduled retraining is appropriate when:
  ✅ Drift rate is predictable (weekly/monthly degradation)
  ✅ Data arrives on regular schedule
  ✅ Training cost is low (< 15 minutes)
  ✅ False alarm risk is low (no need for dynamic triggers)
  ✅ Regulatory requirement for periodic model refresh

Common schedules:
  Daily:    High-frequency forecasting (intraday, next-day)
  Weekly:   Demand forecasting, retail sales
  Monthly:  Long-horizon planning models
  Quarterly: Strategic models, slow-drift environments
```

### 2.2 Implementation

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

class ScheduledRetrainer:
    """
    Runs retraining on a fixed schedule.
    Manages training window, evaluation gating, and model promotion.
    """

    def __init__(
        self,
        schedule: str = "weekly",        # "daily", "weekly", "monthly"
        train_window_days: int = 365,    # how many days of history to train on
        val_window_days: int = 30,       # validation holdout size
        gap_days: int = 1,               # gap between train end and val (simulate latency)
        improvement_threshold: float = 0.02,  # min improvement to promote
    ):
        self.schedule              = schedule
        self.train_window_days     = train_window_days
        self.val_window_days       = val_window_days
        self.gap_days              = gap_days
        self.improvement_threshold = improvement_threshold
        self._last_retrain         = None

    def should_retrain(self, current_date: datetime) -> bool:
        """Check if scheduled retrain is due."""
        if self._last_retrain is None:
            return True

        elapsed = (current_date - self._last_retrain).days
        thresholds = {"daily": 1, "weekly": 7, "monthly": 30}
        return elapsed >= thresholds.get(self.schedule, 7)

    def compute_train_window(self, current_date: datetime) -> tuple:
        """
        Compute training data window for current retrain.

        Returns
        -------
        (train_start, train_end, val_start, val_end)
        """
        val_end   = current_date
        val_start = val_end   - timedelta(days=self.val_window_days)
        train_end = val_start - timedelta(days=self.gap_days)
        train_start = train_end - timedelta(days=self.train_window_days)

        return train_start, train_end, val_start, val_end

    def run(
        self,
        current_date: datetime,
        data_fetcher,      # callable(start, end) → pd.DataFrame
        model_factory,     # callable() → new model instance
        current_model,     # currently deployed model
        feature_transformer,
    ) -> dict:
        """
        Execute one scheduled retraining cycle.

        Parameters
        ----------
        data_fetcher        : function(start, end) → raw DataFrame
        model_factory       : function() → new model instance
        current_model       : current production model (for gating)
        feature_transformer : fitted or unfitted feature transformer

        Returns
        -------
        result dict: {action, metrics, model_version}
        """
        if not self.should_retrain(current_date):
            return {"action": "skip", "reason": "not due"}

        tr_start, tr_end, val_start, val_end = self.compute_train_window(current_date)
        print(f"Retraining: train={tr_start.date()}→{tr_end.date()}, "
              f"val={val_start.date()}→{val_end.date()}")

        # Fetch data
        train_raw = data_fetcher(tr_start, tr_end)
        val_raw   = data_fetcher(val_start, val_end)

        # Feature engineering (fit transformer on TRAIN only)
        feat_train = feature_transformer.fit_transform(train_raw)
        feat_val   = feature_transformer.transform(val_raw)

        X_train = feat_train.drop("target", axis=1).values
        y_train = feat_train["target"].values
        X_val   = feat_val.drop("target", axis=1).values
        y_val   = feat_val["target"].values

        # Train candidate
        candidate = model_factory()
        candidate.fit(X_train, y_train)
        y_hat_cand = candidate.predict(X_val)

        # Evaluate vs. current model
        mae_cand = float(np.abs(y_val - y_hat_cand).mean())

        if current_model is not None:
            y_hat_curr = current_model.predict(X_val)
            mae_curr   = float(np.abs(y_val - y_hat_curr).mean())
            rel_impr   = (mae_curr - mae_cand) / (mae_curr + 1e-12)
            promote    = rel_impr > self.improvement_threshold
        else:
            mae_curr, rel_impr, promote = None, 1.0, True

        action = "promote" if promote else "reject"
        self._last_retrain = current_date

        print(f"  Candidate MAE: {mae_cand:.4f}")
        if mae_curr: print(f"  Current   MAE: {mae_curr:.4f}")
        print(f"  Decision: {action.upper()}")

        return {
            "action":       action,
            "mae_candidate":mae_cand,
            "mae_current":  mae_curr,
            "rel_improvement": rel_impr,
            "train_window": (tr_start, tr_end),
            "model":        candidate if promote else current_model,
        }
```

---

## 3. Trigger-Based Retraining

### 3.1 Trigger Types

```
PERFORMANCE TRIGGER:
  Alert: rolling_mae > baseline_mae * 1.5
  → Immediate retraining job dispatched
  Best for: high-stakes systems where degradation is costly

DATA DRIFT TRIGGER:
  Alert: max_feature_psi > 0.25
  → Schedule retraining within next N hours
  Best for: environments with sudden distribution shifts

VOLUME TRIGGER:
  Alert: n_new_samples > threshold (e.g., 10,000 new observations)
  → Retrain to incorporate new patterns
  Best for: rapidly growing datasets, online marketplaces

CALENDAR TRIGGER:
  Alert: upcoming known event (holiday, product launch, season change)
  → Proactive retraining before the event
  Best for: retail, e-commerce, energy demand

ENSEMBLE: combine multiple triggers
  → Alert if ANY trigger fires (sensitive but more false alarms)
  → Alert if MULTIPLE triggers fire simultaneously (conservative)
```

### 3.2 Trigger Pipeline

```python
from enum import Enum
from datetime import datetime

class TriggerType(Enum):
    PERFORMANCE  = "performance"
    DATA_DRIFT   = "data_drift"
    VOLUME       = "volume"
    SCHEDULED    = "scheduled"

class RetrainingTrigger:
    """Evaluates multiple trigger conditions and dispatches retraining."""

    def __init__(
        self,
        baseline_mae:       float,
        mae_ratio_threshold: float = 1.5,
        psi_threshold:       float = 0.25,
        volume_threshold:    int   = 10_000,
        min_retrain_gap_h:   int   = 6,     # minimum hours between retrains
    ):
        self.baseline_mae        = baseline_mae
        self.mae_ratio_threshold = mae_ratio_threshold
        self.psi_threshold       = psi_threshold
        self.volume_threshold    = volume_threshold
        self.min_retrain_gap_h   = min_retrain_gap_h
        self._last_retrain: Optional[datetime] = None
        self._new_samples_since_retrain = 0

    def check(
        self,
        current_mae:      Optional[float] = None,
        max_feature_psi:  Optional[float] = None,
        n_new_samples:    int = 0,
        current_time:     datetime = None,
    ) -> dict:
        """
        Evaluate all trigger conditions.

        Returns
        -------
        {should_retrain, triggers, priority}
        """
        current_time = current_time or datetime.utcnow()
        triggers     = []

        # Cooldown check
        if self._last_retrain:
            elapsed_h = (current_time - self._last_retrain).total_seconds() / 3600
            if elapsed_h < self.min_retrain_gap_h:
                return {"should_retrain": False, "reason": "cooldown", "triggers": []}

        # Performance trigger
        if current_mae and current_mae / (self.baseline_mae + 1e-12) > self.mae_ratio_threshold:
            triggers.append(TriggerType.PERFORMANCE)

        # Data drift trigger
        if max_feature_psi and max_feature_psi > self.psi_threshold:
            triggers.append(TriggerType.DATA_DRIFT)

        # Volume trigger
        self._new_samples_since_retrain += n_new_samples
        if self._new_samples_since_retrain >= self.volume_threshold:
            triggers.append(TriggerType.VOLUME)

        should_retrain = len(triggers) > 0
        priority       = "high" if TriggerType.PERFORMANCE in triggers else "normal"

        if should_retrain:
            print(f"Retraining triggered by: {[t.value for t in triggers]}")

        return {
            "should_retrain": should_retrain,
            "triggers":       [t.value for t in triggers],
            "priority":       priority,
        }

    def mark_retrained(self):
        """Call after successful retrain to reset counters."""
        self._last_retrain = datetime.utcnow()
        self._new_samples_since_retrain = 0
```

---

## 4. Warm Starting and Incremental Learning

### 4.1 Warm Start vs. Cold Start

```
COLD START (full retrain):
  Train from scratch on the full training window.
  Pros:  Most accurate; no stale weights
  Cons:  Slow; expensive for large datasets
  Use:   When distribution has shifted significantly

WARM START (incremental retrain):
  Initialize new model from previous model's parameters.
  Continue training on new data only.
  Pros:  Fast; preserves learned patterns that are still valid
  Cons:  May not escape local optima from old distribution
  Use:   Gradual drift; frequent updates (daily)

ONLINE UPDATE:
  Update model weights with each new observation (no batch retrain).
  Pros:  Continuous adaptation; zero latency
  Cons:  Catastrophic forgetting; harder to debug
  Use:   High-frequency streaming scenarios

Hybrid: warm start for scheduled retrains + online updates between retrains
```

### 4.2 Warm Start for Tree Models

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb

def warm_start_lightgbm(
    X_new: np.ndarray,
    y_new: np.ndarray,
    prev_model: lgb.Booster,
    n_new_trees: int = 50,
    lr: float = 0.05,
) -> lgb.Booster:
    """
    Warm start a LightGBM model by continuing training on new data.

    Adds n_new_trees trees to the existing model, trained on new data.
    This preserves learned patterns from old data while adapting to new.

    Parameters
    ----------
    X_new       : new training features
    y_new       : new training targets
    prev_model  : previously trained LightGBM Booster
    n_new_trees : number of new trees to add
    lr          : learning rate for new trees

    Returns
    -------
    Updated Booster with additional trees
    """
    dtrain = lgb.Dataset(X_new, label=y_new)

    params = {
        "objective":       "regression",
        "learning_rate":   lr,
        "num_leaves":      31,
        "n_jobs":          -1,
        "verbosity":       -1,
    }

    # Continue training from previous model (init_model parameter)
    updated_model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=n_new_trees,
        init_model=prev_model,    # ← warm start from existing model
        keep_training_booster=True,
    )

    print(f"Warm start: added {n_new_trees} trees "
          f"(total: {updated_model.num_trees()})")
    return updated_model


def warm_start_sklearn(
    X_new: np.ndarray,
    y_new: np.ndarray,
    prev_model: GradientBoostingRegressor,
    n_new_estimators: int = 50,
) -> GradientBoostingRegressor:
    """
    Warm start a sklearn GBM by increasing n_estimators and continuing.
    """
    prev_n = prev_model.n_estimators
    prev_model.set_params(
        n_estimators=prev_n + n_new_estimators,
        warm_start=True,
    )
    prev_model.fit(X_new, y_new)
    prev_model.set_params(warm_start=False)
    print(f"Warm start: {prev_n} → {prev_model.n_estimators} estimators")
    return prev_model
```

---

## 5. Online Learning

### 5.1 river Library for Online TS

```python
def online_sgd_forecast(
    series: np.ndarray,
    context_len: int = 10,
    lr: float = 0.01,
) -> dict:
    """
    Online SGD forecaster using river.
    Processes one observation at a time with O(1) memory.

    pip install river
    """
    try:
        from river import linear_model, optim, preprocessing

        scaler = preprocessing.StandardScaler()
        model  = linear_model.LinearRegression(
            optimizer=optim.SGD(lr=lr),
            intercept_lr=optim.SGD(lr * 0.1),
        )

        predictions = []
        actuals     = []

        for t, x in enumerate(series):
            if t < context_len:
                continue

            features = {f"lag_{i}": series[t - i] for i in range(1, context_len + 1)}
            target   = x

            # Predict first (no future data used)
            x_scaled = scaler.transform_one(features)
            y_hat    = model.predict_one(x_scaled)
            predictions.append(y_hat)
            actuals.append(target)

            # Then update
            model.learn_one(x_scaled, target)
            scaler.learn_one(features)

        mae = float(np.abs(np.array(actuals) - np.array(predictions)).mean())
        print(f"Online SGD — Processed {len(predictions)} steps, MAE: {mae:.4f}")
        return {"predictions": predictions, "actuals": actuals, "mae": mae}

    except ImportError:
        print("Install river: pip install river")
        return {}
```

---

## 6. Training Data Window Strategies

### 6.1 Window Type Comparison

```
EXPANDING WINDOW:
  Train on all data from beginning to current time.
  train_start = fixed; train_end = grows over time
  Pros:  Uses maximum data; good for stable relationships
  Cons:  Old data can hurt if distribution shifted; slow training

ROLLING WINDOW:
  Train on fixed-size recent window.
  train_start = current - W; train_end = current
  Pros:  Focuses on recent distribution; fast training
  Cons:  Throws away useful history; may miss rare events

WEIGHTED WINDOW:
  Use all data but weight recent data more heavily.
  weight(t) = λ^(T - t), λ ∈ (0.9, 0.99)
  Pros:  Balance between recency and history
  Cons:  Requires weighted training support (not all models)

HYBRID:
  Use 2 years history + recent 90 days with higher weight.
  Best of both worlds for most production systems.
```

```python
def select_training_window(
    df: pd.DataFrame,
    timestamp_col: str,
    current_date: datetime,
    strategy: str = "rolling",
    rolling_days: int = 365,
    expanding_start: datetime = None,
    decay_lambda: float = 0.99,
) -> tuple:
    """
    Select training data window and optional sample weights.

    Parameters
    ----------
    strategy : 'rolling', 'expanding', or 'weighted'
    rolling_days : window size for rolling strategy
    expanding_start : start date for expanding strategy
    decay_lambda : exponential decay factor for weighted strategy

    Returns
    -------
    (train_df, sample_weights) — weights is None for non-weighted strategies
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    if strategy == "rolling":
        start = current_date - timedelta(days=rolling_days)
        train = df[df[timestamp_col] >= start]
        return train, None

    elif strategy == "expanding":
        start = expanding_start or df[timestamp_col].min()
        train = df[df[timestamp_col] >= start]
        return train, None

    elif strategy == "weighted":
        # All data, but recent data weighted more
        train = df.copy()
        max_t = df[timestamp_col].max()
        days  = (max_t - train[timestamp_col]).dt.days
        weights = decay_lambda ** days
        return train, weights.values

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
```

---

## 7. Retraining Pipeline

```python
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime

class ProductionRetrainingPipeline:
    """
    End-to-end retraining pipeline with:
      - Trigger evaluation
      - Data window selection
      - Feature engineering
      - Walk-forward validation
      - Promotion gating
      - MLflow tracking
      - Automated rollback capability
    """

    def __init__(
        self,
        model_factory,
        feature_transformer,
        trigger: RetrainingTrigger,
        registry_manager,     # ModelRegistryManager
        model_name: str,
        experiment_name: str,
        window_strategy: str = "rolling",
        window_days: int = 365,
        val_days: int = 30,
        gap_days: int = 1,
        improvement_threshold: float = 0.03,
    ):
        self.model_factory           = model_factory
        self.feature_transformer     = feature_transformer
        self.trigger                 = trigger
        self.registry               = registry_manager
        self.model_name              = model_name
        self.experiment_name         = experiment_name
        self.window_strategy         = window_strategy
        self.window_days             = window_days
        self.val_days                = val_days
        self.gap_days                = gap_days
        self.improvement_threshold   = improvement_threshold

    def run(
        self,
        raw_df: pd.DataFrame,
        current_date: datetime = None,
        force: bool = False,
        monitoring_metrics: dict = None,
    ) -> dict:
        """
        Run the retraining pipeline.

        Parameters
        ----------
        raw_df             : full raw dataset up to current_date
        current_date       : reference date (default: today)
        force              : skip trigger check and always retrain
        monitoring_metrics : live monitoring metrics for trigger evaluation

        Returns
        -------
        result dict: {action, run_id, metrics, model_version}
        """
        current_date = current_date or datetime.utcnow()

        # 1. Check if retraining is needed
        if not force:
            trigger_result = self.trigger.check(**(monitoring_metrics or {}),
                                                 current_time=current_date)
            if not trigger_result["should_retrain"]:
                return {"action": "skip", "reason": "no trigger fired"}

        # 2. Select training window
        val_end      = current_date - pd.Timedelta(days=self.gap_days)
        val_start    = val_end - pd.Timedelta(days=self.val_days)
        train_end    = val_start - pd.Timedelta(days=self.gap_days)
        train_start  = train_end - pd.Timedelta(days=self.window_days)

        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
        train_raw = raw_df[(raw_df["timestamp"] >= train_start) &
                           (raw_df["timestamp"] <= train_end)].copy()
        val_raw   = raw_df[(raw_df["timestamp"] >= val_start) &
                           (raw_df["timestamp"] <= val_end)].copy()

        print(f"Train: {train_start.date()}→{train_end.date()} ({len(train_raw)} rows)")
        print(f"Val:   {val_start.date()}→{val_end.date()} ({len(val_raw)} rows)")

        # 3. Feature engineering
        feat_train = self.feature_transformer.fit_transform(train_raw)
        feat_val   = self.feature_transformer.transform(val_raw)

        X_tr = feat_train.drop(columns=["target"], errors="ignore").values
        y_tr = feat_train["target"].values if "target" in feat_train else train_raw["value"].values[:len(feat_train)]
        X_vl = feat_val.drop(columns=["target"], errors="ignore").values
        y_vl = feat_val["target"].values if "target" in feat_val else val_raw["value"].values[:len(feat_val)]

        # 4. Train and log with MLflow
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=f"retrain_{current_date.date()}") as run:
            model = self.model_factory()
            model.fit(X_tr, y_tr)
            y_hat_cand = model.predict(X_vl)
            mae_cand   = float(np.abs(y_vl - y_hat_cand).mean())
            rmse_cand  = float(np.sqrt(((y_vl - y_hat_cand)**2).mean()))

            mlflow.log_params({
                "train_start": str(train_start.date()),
                "train_end":   str(train_end.date()),
                "n_train":     len(X_tr),
                "strategy":    self.window_strategy,
            })
            mlflow.log_metrics({"val_mae": mae_cand, "val_rmse": rmse_cand})
            mlflow.sklearn.log_model(model, "model")
            run_id = run.info.run_id

        # 5. Promotion gating
        try:
            prod_model  = self.registry.load_production_model(self.model_name)
            y_hat_prod  = prod_model.predict(X_vl)
            mae_prod    = float(np.abs(y_vl - y_hat_prod).mean())
            rel_impr    = (mae_prod - mae_cand) / (mae_prod + 1e-12)
            promote     = rel_impr > self.improvement_threshold
        except Exception:
            mae_prod, rel_impr, promote = None, 1.0, True

        result = {"action": "evaluate", "run_id": run_id,
                  "mae_candidate": mae_cand, "mae_production": mae_prod,
                  "rel_improvement": rel_impr}

        if promote:
            version = self.registry.register(run_id, self.model_name)
            self.registry.promote_to_production(self.model_name, version)
            self.trigger.mark_retrained()
            result["action"] = "promoted"
            result["version"] = version
            print(f"Model promoted: v{version} (MAE {mae_cand:.4f}, +{100*rel_impr:.1f}%)")
        else:
            result["action"] = "rejected"
            print(f"Model rejected (improvement {100*rel_impr:.1f}% < {100*self.improvement_threshold:.1f}%)")

        return result
```

---

*← [04 — Drift Detection](./04_drift_detection_and_monitoring.md) | [Module README](./README.md) | Next: [06 — Serving](./06_serving_ts_models.md) →*
