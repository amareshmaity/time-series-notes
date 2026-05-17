# 06 — Serving Time Series Models

> **Module**: 11 Production & MLOps | **File**: 6 of 6
>
> A model that can't be served reliably in production is useless. Time series serving is uniquely challenging: predictions have temporal dependencies, inputs require feature engineering at query time, and freshness matters. This note covers FastAPI serving, batch vs. real-time strategies, caching, and latency optimization.

---

## Table of Contents

1. [Serving Modes for Time Series](#1-serving-modes-for-time-series)
2. [FastAPI Serving](#2-fastapi-serving)
3. [Batch Prediction Jobs](#3-batch-prediction-jobs)
4. [Prediction Caching](#4-prediction-caching)
5. [Latency Optimization](#5-latency-optimization)
6. [Serving Stateful Models](#6-serving-stateful-models)
7. [Prediction Interval Serving](#7-prediction-interval-serving)
8. [Health Checks and Observability](#8-health-checks-and-observability)

---

## 1. Serving Modes for Time Series

### 1.1 Batch vs. Real-Time vs. Hybrid

```
BATCH (pre-computed):
  - Run predictions once (daily/hourly) for all entities
  - Store results in a low-latency key-value store (Redis/DynamoDB)
  - API reads from cache → sub-millisecond serving latency
  - Freshness: limited by batch frequency

  Use when:
    ✅ Predictions needed on a known schedule
    ✅ Hundreds to millions of entities to score
    ✅ Low latency is critical (<10ms)
    ✅ Feature computation is expensive

REAL-TIME (on-demand):
  - API receives request → computes features → runs model → returns
  - Freshness: uses data available at query time
  - Latency: feature computation + model inference (50–500ms typical)

  Use when:
    ✅ User provides custom context at query time
    ✅ Small number of entities (< 1000)
    ✅ Freshness is more important than latency

HYBRID (recommended for most TS systems):
  - Pre-compute standard horizons (next 7 days) in batch
  - Cache results → serve in < 5ms for common queries
  - Fall back to real-time for uncommon horizons or fresh context

  cache_miss → real-time inference → optionally backfill cache
```

---

## 2. FastAPI Serving

### 2.1 Core Serving Application

```python
# serving/app.py
# pip install fastapi uvicorn pydantic

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np
import pandas as pd
import time
import logging

logger = logging.getLogger("ts_serving")

app = FastAPI(
    title="Time Series Forecasting API",
    description="Production serving for TS models",
    version="1.0.0",
)


# ── Request / Response Schemas ─────────────────────────────────────────────

class ForecastRequest(BaseModel):
    entity_id: str = Field(..., description="Entity to forecast (e.g., store_id)")
    horizon:   int = Field(7, ge=1, le=90, description="Forecast horizon in steps")
    as_of:     Optional[str] = Field(None, description="Point-in-time for PIT-correct features")
    context:   Optional[List[float]] = Field(None, description="Optional recent history override")

    @validator("entity_id")
    def entity_not_empty(cls, v):
        if not v.strip():
            raise ValueError("entity_id cannot be empty")
        return v.strip()


class ForecastPoint(BaseModel):
    step:         int
    timestamp:    str
    forecast:     float
    lower_bound:  Optional[float] = None
    upper_bound:  Optional[float] = None


class ForecastResponse(BaseModel):
    entity_id:       str
    horizon:         int
    forecasts:       List[ForecastPoint]
    model_version:   str
    latency_ms:      float
    cache_hit:       bool
    generated_at:    str


# ── Model and Feature Loading ───────────────────────────────────────────────

class ModelServer:
    """
    Loads and serves a production TS model.
    Thread-safe: model is loaded once at startup.
    """

    def __init__(self):
        self.model              = None
        self.feature_transformer = None
        self.model_version      = "unknown"
        self.online_store       = None    # feature store for fast lookup

    def load(self, model_path: str, transformer_path: str, version: str = "v1"):
        """Load model and transformer from disk / MLflow."""
        import cloudpickle
        with open(model_path, "rb") as f:
            self.model = cloudpickle.load(f)
        with open(transformer_path, "rb") as f:
            self.feature_transformer = cloudpickle.load(f)
        self.model_version = version
        logger.info(f"Model {version} loaded from {model_path}")

    def load_from_mlflow(self, model_name: str, stage: str = "Production"):
        """Load model directly from MLflow registry."""
        import mlflow
        self.model         = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
        self.model_version = f"{model_name}/{stage}"
        logger.info(f"Loaded {self.model_version} from MLflow")

    def predict(
        self,
        entity_id: str,
        horizon: int,
        as_of: str = None,
        context: list = None,
    ) -> np.ndarray:
        """
        Run model inference for one entity.

        Parameters
        ----------
        entity_id : entity to forecast
        horizon   : number of steps ahead
        as_of     : point-in-time date string (uses online store features)
        context   : optional recent history to override stored features

        Returns
        -------
        forecasts : (horizon,) array of predicted values
        """
        if self.model is None:
            raise RuntimeError("Model not loaded — call .load() first")

        # Get features from online store or context
        if context is not None:
            features = self._features_from_context(context)
        elif self.online_store is not None:
            raw_feats = self.online_store.get_online_features("ts_features", entity_id)
            features  = np.array([raw_feats.get(k, 0.0) for k in self._feature_names()])
        else:
            raise ValueError("No context provided and no online store configured")

        # Recursive multi-step forecast
        preds = []
        hist  = list(context or [features[0]])   # rolling history buffer

        for step in range(horizon):
            x       = self._build_feature_vector(hist)
            y_hat   = float(self.model.predict(x.reshape(1, -1))[0])
            preds.append(y_hat)
            hist.append(y_hat)

        return np.array(preds)

    def _features_from_context(self, context: list) -> np.ndarray:
        """Build feature vector from recent history."""
        s = pd.Series(context)
        return np.array([
            s.mean(), s.std(), s.iloc[-1], s.iloc[-2] if len(s) > 1 else s.iloc[-1],
            s.rolling(min(7, len(s))).mean().iloc[-1],
            s.rolling(min(30, len(s))).mean().iloc[-1],
        ], dtype=np.float32)

    def _build_feature_vector(self, hist: list) -> np.ndarray:
        """Build feature vector from rolling history buffer."""
        s = np.array(hist[-30:], dtype=np.float32)  # keep last 30 values
        return self._features_from_context(list(s))

    def _feature_names(self) -> list:
        return ["mean", "std", "lag_1", "lag_2", "roll_mean_7", "roll_mean_30"]


# Global server instance (singleton)
server = ModelServer()


# ── Prediction Cache ────────────────────────────────────────────────────────

class PredictionCache:
    """
    In-memory prediction cache with TTL.
    Replace with Redis for multi-instance deployments.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self._store    = {}
        self.ttl       = ttl_seconds

    def _key(self, entity_id: str, horizon: int, as_of: str) -> str:
        return f"{entity_id}:{horizon}:{as_of or 'latest'}"

    def get(self, entity_id, horizon, as_of=None):
        key   = self._key(entity_id, horizon, as_of)
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.time() - entry["ts"] > self.ttl:
            del self._store[key]
            return None
        return entry["data"]

    def set(self, entity_id, horizon, data, as_of=None):
        key = self._key(entity_id, horizon, as_of)
        self._store[key] = {"data": data, "ts": time.time()}

cache = PredictionCache(ttl_seconds=3600)


# ── API Endpoints ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Load model at startup (not per-request)."""
    # Example: load from MLflow in a real deployment
    # server.load_from_mlflow("ts_forecast_model", stage="Production")
    logger.info("API started — model loading deferred to first request for demo")


@app.get("/health")
async def health():
    """Health check endpoint for load balancer / Kubernetes probes."""
    return {
        "status":        "healthy",
        "model_version": server.model_version,
        "model_loaded":  server.model is not None,
    }


@app.get("/ready")
async def readiness():
    """Readiness probe — fails if model is not loaded."""
    if server.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest, background_tasks: BackgroundTasks):
    """
    Generate a time series forecast for one entity.

    Returns point forecasts (and optionally prediction intervals).
    Cache hit if same request was made within TTL.
    """
    t0 = time.time()

    # Check cache
    cached = cache.get(request.entity_id, request.horizon, request.as_of)
    if cached is not None:
        latency = (time.time() - t0) * 1000
        cached["latency_ms"] = round(latency, 2)
        cached["cache_hit"]  = True
        return ForecastResponse(**cached)

    # Real-time inference
    if server.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        forecasts = server.predict(
            entity_id=request.entity_id,
            horizon=request.horizon,
            as_of=request.as_of,
            context=request.context,
        )
    except Exception as e:
        logger.error(f"Prediction error for {request.entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    from datetime import datetime, timedelta
    now = datetime.utcnow()
    points = [
        ForecastPoint(
            step=i+1,
            timestamp=(now + timedelta(days=i+1)).strftime("%Y-%m-%d"),
            forecast=round(float(forecasts[i]), 4),
        )
        for i in range(request.horizon)
    ]

    latency = (time.time() - t0) * 1000
    payload = {
        "entity_id":     request.entity_id,
        "horizon":       request.horizon,
        "forecasts":     points,
        "model_version": server.model_version,
        "latency_ms":    round(latency, 2),
        "cache_hit":     False,
        "generated_at":  now.isoformat(),
    }

    # Cache result asynchronously
    background_tasks.add_task(
        cache.set, request.entity_id, request.horizon, payload, request.as_of
    )

    logger.info(f"Forecast for {request.entity_id} h={request.horizon} "
                f"in {latency:.1f}ms")
    return ForecastResponse(**payload)


@app.post("/forecast/batch")
async def batch_forecast(requests: List[ForecastRequest]):
    """
    Score multiple entities in one API call.
    More efficient than N separate /forecast calls.
    """
    t0      = time.time()
    results = []

    for req in requests:
        try:
            resp = await forecast(req, BackgroundTasks())
            results.append({"entity_id": req.entity_id, "status": "ok",
                             "forecasts": resp.forecasts})
        except Exception as e:
            results.append({"entity_id": req.entity_id, "status": "error",
                             "error": str(e)})

    total_ms = (time.time() - t0) * 1000
    return {
        "n_entities":   len(requests),
        "n_success":    sum(1 for r in results if r["status"] == "ok"),
        "total_ms":     round(total_ms, 2),
        "per_entity_ms":round(total_ms / max(len(requests), 1), 2),
        "results":      results,
    }


# ── Run locally ────────────────────────────────────────────────────────────
# uvicorn serving.app:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 3. Batch Prediction Jobs

```python
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List

def run_batch_scoring_job(
    model,
    feature_df: pd.DataFrame,
    entity_col: str,
    horizon: int,
    output_store,    # OnlineFeatureStore or parquet writer
    as_of: datetime = None,
) -> pd.DataFrame:
    """
    Batch scoring job: score all entities and write to output store.

    Designed to run daily as a scheduled job (Airflow / cron).
    Writes results to both:
      - Online store (for low-latency API serving)
      - Offline store (for monitoring, analysis, historical forecasts)

    Parameters
    ----------
    model       : fitted model with .predict() method
    feature_df  : (n_entities, n_features) DataFrame, one row per entity
    entity_col  : column with entity identifiers
    horizon     : number of steps to forecast
    output_store: online feature store to write predictions to
    as_of       : prediction point-in-time (default: now)
    """
    as_of    = as_of or datetime.utcnow()
    results  = []
    entities = feature_df[entity_col].unique()

    print(f"Batch scoring {len(entities)} entities for horizon={horizon}...")
    t0 = datetime.utcnow()

    for entity in entities:
        entity_feats = feature_df[feature_df[entity_col] == entity]
        X            = entity_feats.drop(columns=[entity_col]).values

        try:
            preds = model.predict(X)   # shape: (1, horizon) or (horizon,)
            if preds.ndim == 2:
                preds = preds[0]

            for step, pred in enumerate(preds[:horizon]):
                results.append({
                    "entity_id":  entity,
                    "as_of":      as_of,
                    "step":       step + 1,
                    "forecast":   float(pred),
                    "scored_at":  datetime.utcnow(),
                })
        except Exception as e:
            print(f"  Error scoring {entity}: {e}")

    result_df = pd.DataFrame(results)

    # Write to online store (latest forecast per entity)
    for entity in entities:
        ent_rows = result_df[result_df["entity_id"] == entity]
        forecast_payload = {
            f"forecast_step_{row['step']}": row["forecast"]
            for _, row in ent_rows.iterrows()
        }
        output_store.write_features(
            feature_view="forecasts",
            entity_id=str(entity),
            features=forecast_payload,
            timestamp=as_of.isoformat(),
        )

    elapsed = (datetime.utcnow() - t0).total_seconds()
    print(f"Batch scoring complete: {len(results)} predictions in {elapsed:.1f}s")
    return result_df
```

---

## 4. Prediction Caching

```python
import json, time, hashlib
from typing import Optional

class ForecastCache:
    """
    Two-level prediction cache:
      L1: In-memory dict (sub-millisecond, bounded size)
      L2: Redis (millisecond, shared across instances)

    Cache key: hash of (entity_id, horizon, model_version, as_of)
    TTL: configurable per forecast horizon (shorter horizons → fresher)
    """

    def __init__(
        self,
        l1_maxsize: int = 1000,
        l1_ttl:     int = 300,    # 5 minutes
        l2_ttl:     int = 3600,   # 1 hour
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        from collections import OrderedDict
        self._l1      = OrderedDict()
        self._l1_ts   = {}
        self.l1_max   = l1_maxsize
        self.l1_ttl   = l1_ttl
        self.l2_ttl   = l2_ttl

        try:
            import redis
            self._l2 = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self._l2.ping()
        except Exception:
            self._l2 = None

    def _cache_key(self, entity_id, horizon, model_version, as_of="latest") -> str:
        raw = f"{entity_id}:{horizon}:{model_version}:{as_of}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, entity_id, horizon, model_version, as_of="latest") -> Optional[list]:
        key = self._cache_key(entity_id, horizon, model_version, as_of)

        # L1 check
        if key in self._l1:
            if time.time() - self._l1_ts[key] < self.l1_ttl:
                return self._l1[key]
            del self._l1[key]; del self._l1_ts[key]

        # L2 check
        if self._l2:
            raw = self._l2.get(f"forecast:{key}")
            if raw:
                data = json.loads(raw)
                self._l1_put(key, data)
                return data

        return None

    def set(self, entity_id, horizon, model_version, data: list, as_of="latest"):
        key = self._cache_key(entity_id, horizon, model_version, as_of)
        self._l1_put(key, data)
        if self._l2:
            self._l2.setex(f"forecast:{key}", self.l2_ttl, json.dumps(data))

    def _l1_put(self, key, data):
        if len(self._l1) >= self.l1_max:
            oldest = next(iter(self._l1))
            del self._l1[oldest]; del self._l1_ts[oldest]
        self._l1[key]    = data
        self._l1_ts[key] = time.time()
```

---

## 5. Latency Optimization

```python
# Key techniques for sub-100ms latency:

# 1. Pre-load model at startup (not per-request)
model = load_model_once()  # ← do this, not inside endpoint

# 2. Batch small requests
def predict_batch(X_batch: np.ndarray) -> np.ndarray:
    """Always predict in batches (even if batch_size=1)."""
    return model.predict(X_batch)

# 3. Feature store lookup instead of computing
def get_features_fast(entity_id: str) -> np.ndarray:
    """Sub-millisecond online feature lookup."""
    return online_store.get_online_features("ts_features", entity_id)

# 4. Model quantization for deep learning
def quantize_pytorch_model(model, calibration_data):
    """INT8 quantization → 2-4x speedup, < 1% accuracy loss."""
    import torch
    model.eval()
    model_quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
    )
    return model_quantized

# 5. ONNX export for cross-platform inference
def export_to_onnx(torch_model, sample_input, onnx_path: str):
    """Export PyTorch model to ONNX for faster CPU inference."""
    import torch
    torch.onnx.export(
        torch_model, sample_input, onnx_path,
        input_names=["input"], output_names=["forecast"],
        dynamic_axes={"input": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"ONNX model saved: {onnx_path}")
```

---

## 6. Serving Stateful Models

```python
import numpy as np
from collections import defaultdict

class StatefulForecastServer:
    """
    Serves ARIMA / LSTM models that carry state between predictions.
    Maintains per-entity state (residuals, hidden states) in memory.
    """

    def __init__(self, model_template, max_entities: int = 10_000):
        self._model_template = model_template
        self._entity_states: dict = {}
        self._entity_history: dict = defaultdict(list)
        self.max_entities = max_entities

    def observe(self, entity_id: str, timestamp, value: float) -> None:
        """Record a new observation for an entity (update state)."""
        self._entity_history[entity_id].append((timestamp, value))
        # Evict least-recently-used if at capacity
        if len(self._entity_states) >= self.max_entities:
            oldest = next(iter(self._entity_states))
            del self._entity_states[oldest]

    def predict(self, entity_id: str, horizon: int = 1) -> np.ndarray:
        """
        Generate forecast using entity-specific state.
        Re-fits on entity history if no state exists yet.
        """
        history = self._entity_history.get(entity_id, [])
        if len(history) < 5:
            raise ValueError(f"Insufficient history for {entity_id}: {len(history)} < 5")

        values = np.array([v for _, v in history])

        if entity_id not in self._entity_states:
            # First call: fit model on full history
            model = self._model_template()
            model.fit(values)
            self._entity_states[entity_id] = model
        else:
            # Subsequent calls: update with latest observations
            model = self._entity_states[entity_id]
            # For models supporting online update (e.g., statsmodels SARIMA append)
            if hasattr(model, "append"):
                model = model.append(values[-1:])
                self._entity_states[entity_id] = model

        return model.forecast(horizon)
```

---

## 7. Prediction Interval Serving

```python
from pydantic import BaseModel
from typing import List, Optional

class PredictionIntervalResponse(BaseModel):
    entity_id:   str
    forecasts:   List[float]
    lower_80:    Optional[List[float]] = None
    upper_80:    Optional[List[float]] = None
    lower_95:    Optional[List[float]] = None
    upper_95:    Optional[List[float]] = None

def serve_with_intervals(
    model,
    X: np.ndarray,
    method: str = "residual_bootstrap",
    n_bootstrap: int = 200,
    alpha_levels: list = None,
) -> dict:
    """
    Serve point forecasts with prediction intervals.

    Parameters
    ----------
    model    : fitted model with .predict()
    X        : feature matrix for one entity
    method   : 'residual_bootstrap', 'quantile', 'conformal'
    n_bootstrap: number of bootstrap samples (for bootstrap method)
    alpha_levels: coverage levels e.g. [0.80, 0.95]

    Returns
    -------
    dict with point forecasts + lower/upper bounds per alpha level
    """
    alpha_levels = alpha_levels or [0.80, 0.95]
    point_pred   = model.predict(X)

    if method == "residual_bootstrap":
        # Bootstrap using stored training residuals
        if not hasattr(model, "_residuals"):
            return {"forecasts": list(point_pred)}

        residuals = model._residuals
        boot_preds = np.array([
            point_pred + np.random.choice(residuals, size=len(point_pred), replace=True)
            for _ in range(n_bootstrap)
        ])

        result = {"forecasts": list(point_pred.round(4))}
        for alpha in alpha_levels:
            lo = np.percentile(boot_preds, 100*(1-alpha)/2, axis=0)
            hi = np.percentile(boot_preds, 100*(1+alpha)/2, axis=0)
            key = str(int(alpha*100))
            result[f"lower_{key}"] = list(lo.round(4))
            result[f"upper_{key}"] = list(hi.round(4))

        return result

    elif method == "quantile":
        # Direct quantile regression (model must support)
        result = {"forecasts": list(point_pred.round(4))}
        for alpha in alpha_levels:
            key = str(int(alpha*100))
            lo_q = (1 - alpha) / 2
            hi_q = (1 + alpha) / 2
            # Model must expose quantile prediction
            if hasattr(model, "predict_quantile"):
                result[f"lower_{key}"] = list(model.predict_quantile(X, lo_q).round(4))
                result[f"upper_{key}"] = list(model.predict_quantile(X, hi_q).round(4))
        return result

    return {"forecasts": list(point_pred.round(4))}
```

---

## 8. Health Checks and Observability

```python
import time
import threading
from collections import deque

class ServingMetricsCollector:
    """
    Lightweight metrics collector for the serving layer.
    Records latency, throughput, error rate, and cache statistics.
    """

    def __init__(self, window: int = 1000):
        self._latencies   = deque(maxlen=window)
        self._errors      = deque(maxlen=window)
        self._cache_hits  = deque(maxlen=window)
        self._lock        = threading.Lock()
        self._request_count = 0

    def record_request(self, latency_ms: float, is_error: bool, cache_hit: bool):
        with self._lock:
            self._latencies.append(latency_ms)
            self._errors.append(int(is_error))
            self._cache_hits.append(int(cache_hit))
            self._request_count += 1

    def get_metrics(self) -> dict:
        with self._lock:
            lats = list(self._latencies)
            errs = list(self._errors)
            hits = list(self._cache_hits)

        if not lats:
            return {"status": "no_requests"}

        return {
            "n_requests":     self._request_count,
            "latency_p50_ms": round(float(np.percentile(lats, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(lats, 95)), 2),
            "latency_p99_ms": round(float(np.percentile(lats, 99)), 2),
            "error_rate":     round(float(np.mean(errs)), 4),
            "cache_hit_rate": round(float(np.mean(hits)), 4),
            "window_size":    len(lats),
        }

# Add metrics endpoint to FastAPI app
metrics_collector = ServingMetricsCollector()

@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    return metrics_collector.get_metrics()
```

---

*← [05 — Retraining](./05_retraining_strategies.md) | [Module README](./README.md)*
