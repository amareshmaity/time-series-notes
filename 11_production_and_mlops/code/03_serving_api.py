"""
code/03_serving_api.py
========================
Module 11 — Production & MLOps
Practical: FastAPI serving endpoint with caching and health checks.

Demonstrates:
  - FastAPI app with /forecast, /forecast/batch, /health, /metrics endpoints
  - In-memory prediction cache with TTL
  - Request latency measurement and logging
  - Graceful model loading at startup
  - Input validation with Pydantic
  - Serving metrics (P50/P95/P99 latency, cache hit rate, error rate)
  - Demo client that exercises all endpoints

Run with:
  pip install fastapi uvicorn pydantic
  python 03_serving_api.py              # self-contained demo mode
  uvicorn code.03_serving_api:app --port 8000  # actual server mode
"""

import numpy as np
import pandas as pd
import time
import json
import hashlib
import threading
import logging
from collections import deque, OrderedDict
from datetime import datetime, timedelta
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# 1. Minimal Model Stub (replaces loading from MLflow for standalone demo)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleForecastModel:
    """
    Minimal production-grade model stub for serving demos.
    Replace with your actual model in production.
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self._params = {
            "trend_weight": 0.01,
            "seasonal_amp": 10.0,
            "base_value":   100.0,
        }
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleForecastModel":
        self._mean  = float(y.mean())
        self._std   = float(y.std())
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point forecast for each row in X."""
        n = len(X)
        # Simulate a model: weighted average of input features + noise
        preds = np.array([float(X[i].mean() * 0.9 + self._mean * 0.1)
                          for i in range(n)], dtype=float)
        return preds

    def predict_horizon(self, context: list, horizon: int) -> np.ndarray:
        """Recursive multi-step forecast from context."""
        hist = list(context)
        preds = []
        for step in range(horizon):
            base  = np.mean(hist[-7:]) if len(hist) >= 7 else np.mean(hist)
            trend = 0.05 * step
            seas  = self._params["seasonal_amp"] * np.sin(2 * np.pi * (len(hist) % 365) / 365)
            noise = np.random.normal(0, 1.0)
            y_hat = base + trend + 0.1 * seas + noise
            preds.append(y_hat)
            hist.append(y_hat)
        return np.array(preds)

    @property
    def version(self):
        return "demo_v1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prediction Cache
# ─────────────────────────────────────────────────────────────────────────────

class PredictionCache:
    """Two-level in-memory cache with TTL and LRU eviction."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self._store  = OrderedDict()
        self._ts     = {}
        self.max     = max_size
        self.ttl     = ttl_seconds
        self._hits   = 0
        self._misses = 0
        self._lock   = threading.Lock()

    def _key(self, entity_id: str, horizon: int, model_version: str) -> str:
        raw = f"{entity_id}:{horizon}:{model_version}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def get(self, entity_id: str, horizon: int, model_version: str):
        key = self._key(entity_id, horizon, model_version)
        with self._lock:
            if key in self._store:
                if time.time() - self._ts[key] < self.ttl:
                    self._store.move_to_end(key)   # LRU update
                    self._hits += 1
                    return self._store[key]
                else:
                    del self._store[key]; del self._ts[key]
            self._misses += 1
        return None

    def set(self, entity_id: str, horizon: int, model_version: str, data) -> None:
        key = self._key(entity_id, horizon, model_version)
        with self._lock:
            if len(self._store) >= self.max:
                oldest = next(iter(self._store))
                del self._store[oldest]; del self._ts[oldest]
            self._store[key] = data
            self._ts[key]    = time.time()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._store)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Serving Metrics Collector
# ─────────────────────────────────────────────────────────────────────────────

class ServingMetrics:
    """Thread-safe latency, throughput, and error metrics."""

    def __init__(self, window: int = 500):
        self._latencies   = deque(maxlen=window)
        self._errors      = deque(maxlen=window)
        self._total_reqs  = 0
        self._lock        = threading.Lock()

    def record(self, latency_ms: float, is_error: bool = False) -> None:
        with self._lock:
            self._latencies.append(latency_ms)
            self._errors.append(int(is_error))
            self._total_reqs += 1

    def summary(self) -> dict:
        with self._lock:
            lats = list(self._latencies)
            errs = list(self._errors)

        if not lats:
            return {"status": "no_requests"}

        return {
            "total_requests": self._total_reqs,
            "window_size":    len(lats),
            "latency_p50_ms": round(float(np.percentile(lats, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(lats, 95)), 2),
            "latency_p99_ms": round(float(np.percentile(lats, 99)), 2),
            "latency_mean_ms":round(float(np.mean(lats)), 2),
            "error_rate":     round(float(np.mean(errs)), 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. FastAPI Application
# ─────────────────────────────────────────────────────────────────────────────

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field, validator

    app     = FastAPI(title="TS Forecast API", version="1.0.0")
    cache   = PredictionCache(max_size=1000, ttl_seconds=3600)
    metrics = ServingMetrics(window=500)
    _model  = None

    # ─── Pydantic Schemas ──────────────────────────────────────────────────

    class ForecastRequest(BaseModel):
        entity_id: str  = Field(..., min_length=1, max_length=100)
        horizon:   int  = Field(7, ge=1, le=90)
        context:   Optional[List[float]] = Field(None, min_items=1, max_items=500)

        @validator("entity_id")
        def clean_entity_id(cls, v):
            return v.strip()

    class ForecastPoint(BaseModel):
        step:      int
        timestamp: str
        forecast:  float

    class ForecastResponse(BaseModel):
        entity_id:     str
        horizon:       int
        forecasts:     List[ForecastPoint]
        model_version: str
        latency_ms:    float
        cache_hit:     bool
        generated_at:  str

    # ─── Startup ───────────────────────────────────────────────────────────

    @app.on_event("startup")
    async def startup():
        global _model
        _model = SimpleForecastModel()
        # Fit on dummy data (in production: load from MLflow)
        _model.fit(np.random.randn(100, 5), np.random.randn(100))
        logging.info(f"Model {_model.version} loaded")
        print(f"[startup] Model {_model.version} ready")

    # ─── Health / Readiness ────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {
            "status":        "healthy",
            "model_version": _model.version if _model else "not_loaded",
            "model_loaded":  _model is not None,
            "cache_size":    cache.size,
        }

    @app.get("/ready")
    async def readiness():
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {"status": "ready"}

    @app.get("/metrics")
    async def get_metrics():
        return {
            **metrics.summary(),
            "cache_hit_rate": round(cache.hit_rate, 4),
            "cache_size":     cache.size,
        }

    # ─── Forecast Endpoint ─────────────────────────────────────────────────

    @app.post("/forecast", response_model=ForecastResponse)
    async def forecast(request: ForecastRequest):
        t0 = time.time()

        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Cache lookup
        cached = cache.get(request.entity_id, request.horizon, _model.version)
        if cached is not None:
            latency = (time.time() - t0) * 1000
            metrics.record(latency)
            cached["latency_ms"] = round(latency, 2)
            cached["cache_hit"]  = True
            return ForecastResponse(**cached)

        # Inference
        try:
            context = request.context or [100.0 + np.random.normal(0, 5) for _ in range(30)]
            preds   = _model.predict_horizon(context, request.horizon)
        except Exception as e:
            latency = (time.time() - t0) * 1000
            metrics.record(latency, is_error=True)
            raise HTTPException(status_code=500, detail=str(e))

        now    = datetime.utcnow()
        points = [
            ForecastPoint(
                step=i+1,
                timestamp=(now + timedelta(days=i+1)).strftime("%Y-%m-%d"),
                forecast=round(float(preds[i]), 4),
            )
            for i in range(request.horizon)
        ]

        latency = (time.time() - t0) * 1000
        metrics.record(latency)

        payload = {
            "entity_id":     request.entity_id,
            "horizon":       request.horizon,
            "forecasts":     [p.dict() for p in points],
            "model_version": _model.version,
            "latency_ms":    round(latency, 2),
            "cache_hit":     False,
            "generated_at":  now.isoformat(),
        }
        cache.set(request.entity_id, request.horizon, _model.version, payload)
        return ForecastResponse(**payload)

    # ─── Batch Endpoint ────────────────────────────────────────────────────

    @app.post("/forecast/batch")
    async def batch_forecast(requests: List[ForecastRequest]):
        t0      = time.time()
        results = []

        for req in requests:
            try:
                resp = await forecast(req)
                results.append({
                    "entity_id": req.entity_id,
                    "status":    "ok",
                    "forecasts": [{"step": f.step, "forecast": f.forecast}
                                   for f in resp.forecasts],
                })
            except Exception as e:
                results.append({
                    "entity_id": req.entity_id,
                    "status":    "error",
                    "error":     str(e),
                })

        total_ms = (time.time() - t0) * 1000
        return {
            "n_entities":    len(requests),
            "n_success":     sum(1 for r in results if r["status"] == "ok"),
            "total_ms":      round(total_ms, 2),
            "per_entity_ms": round(total_ms / max(len(requests), 1), 2),
            "results":       results,
        }

    FASTAPI_AVAILABLE = True

except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed (pip install fastapi uvicorn)")
    print("Running standalone demo instead...")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Standalone Demo (no server needed)
# ─────────────────────────────────────────────────────────────────────────────

def run_standalone_demo():
    """
    Simulates the full serving lifecycle without starting a server.
    Demonstrates: model load → cache warm → batch scoring → metrics.
    """
    import matplotlib.pyplot as plt
    print("\n" + "="*60)
    print("STANDALONE SERVING DEMO")
    print("="*60)

    # Setup
    model   = SimpleForecastModel()
    model.fit(np.random.randn(100, 5), np.random.randn(100) + 100)
    cache_  = PredictionCache(max_size=100, ttl_seconds=60)
    metrics_= ServingMetrics(window=500)

    entities  = [f"store_{i:03d}" for i in range(50)]
    horizons  = [7, 14, 30]
    latencies = []

    print(f"\nSimulating {len(entities)} entities × {len(horizons)} horizons = "
          f"{len(entities)*len(horizons)} requests (2 rounds each)...")

    for round_n in range(2):
        for entity in entities:
            for horizon in horizons:
                t0 = time.perf_counter()

                # Cache check
                cached = cache_.get(entity, horizon, model.version)
                if cached is not None:
                    lat = (time.perf_counter() - t0) * 1000
                    metrics_.record(lat)
                    latencies.append(lat)
                    continue

                # Inference
                context = [100 + np.random.normal(0, 5) for _ in range(30)]
                preds   = model.predict_horizon(context, horizon)

                lat = (time.perf_counter() - t0) * 1000
                metrics_.record(lat)
                latencies.append(lat)

                # Cache write
                cache_.set(entity, horizon, model.version, {"preds": preds.tolist()})

    summary = metrics_.summary()
    print(f"\nPerformance Summary ({len(latencies)} requests):")
    print(f"  P50 latency:   {summary['latency_p50_ms']:.3f} ms")
    print(f"  P95 latency:   {summary['latency_p95_ms']:.3f} ms")
    print(f"  P99 latency:   {summary['latency_p99_ms']:.3f} ms")
    print(f"  Mean latency:  {summary['latency_mean_ms']:.3f} ms")
    print(f"  Error rate:    {summary['error_rate']:.4f}")
    print(f"  Cache hit rate:{cache_.hit_rate:.4f} ({100*cache_.hit_rate:.1f}%)")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: Latency histogram
    ax = axes[0]
    ax.hist(latencies, bins=30, color="#2196F3", edgecolor="white", alpha=0.85)
    for pct, col, lbl in [(50,"green","P50"), (95,"orange","P95"), (99,"red","P99")]:
        v = float(np.percentile(latencies, pct))
        ax.axvline(v, color=col, linestyle="--", linewidth=1.5, label=f"{lbl}: {v:.2f}ms")
    ax.set_title("Request Latency Distribution", fontsize=11)
    ax.set_xlabel("Latency (ms)"); ax.set_ylabel("Count")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: Example forecast
    ax = axes[1]
    context = [100 + 5*np.sin(2*np.pi*t/30) + np.random.normal(0,2) for t in range(90)]
    preds   = model.predict_horizon(context, 30)
    t_hist  = np.arange(-len(context), 0)
    t_fore  = np.arange(0, len(preds))
    ax.plot(t_hist, context, color="#2196F3", linewidth=1.5, label="History")
    ax.plot(t_fore, preds,   color="#FF5722", linewidth=2, linestyle="--", label="Forecast")
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Example Forecast (horizon=30)", fontsize=11)
    ax.set_xlabel("Time step (0 = now)"); ax.set_ylabel("Value")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 3: Cache stats
    ax = axes[2]
    categories = ["Cache Hits", "Cache Misses"]
    total      = cache_._hits + cache_._misses
    counts     = [cache_._hits, cache_._misses]
    colors     = ["#4CAF50", "#FF9800"]
    wedges, texts, autotexts = ax.pie(counts, labels=categories, colors=colors,
                                       autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Cache Performance\n(total={total} requests)", fontsize=11)

    plt.suptitle("Serving API Demo — Latency, Forecasting, Cache",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("serving_api_demo.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nPlot saved: serving_api_demo.png")

    # Batch scoring example
    print("\nBatch scoring example (5 entities, horizon=7):")
    batch_t0 = time.perf_counter()
    batch_results = []
    for entity in entities[:5]:
        context = [100 + np.random.normal(0, 5) for _ in range(30)]
        preds   = model.predict_horizon(context, 7)
        batch_results.append({"entity": entity, "forecasts": preds.tolist()})
    batch_ms = (time.perf_counter() - batch_t0) * 1000
    print(f"  Scored 5 entities in {batch_ms:.2f}ms ({batch_ms/5:.2f}ms/entity)")
    for r in batch_results[:3]:
        print(f"  {r['entity']}: {[round(f,2) for f in r['forecasts'][:3]]}...")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        print("FastAPI is available.")
        print("To start the server: uvicorn code.03_serving_api:app --host 0.0.0.0 --port 8000")
        print("Endpoints:")
        print("  GET  /health         — health check")
        print("  GET  /ready          — readiness probe")
        print("  GET  /metrics        — serving metrics")
        print("  POST /forecast       — single entity forecast")
        print("  POST /forecast/batch — multi-entity batch forecast")
        print("\nRunning standalone demo instead of starting server...")

    run_standalone_demo()
