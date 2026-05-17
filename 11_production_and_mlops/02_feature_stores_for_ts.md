# 02 — Feature Stores for Time Series

> **Module**: 11 Production & MLOps | **File**: 2 of 6
>
> Feature stores solve the hardest production ML problem: making the same features available consistently during training, batch inference, and real-time serving — all while maintaining strict point-in-time correctness. For time series, the stakes are even higher because temporal leakage is invisible in offline metrics but catastrophic in production.

---

## Table of Contents

1. [Why Feature Stores?](#1-why-feature-stores)
2. [Point-in-Time Correct Joins](#2-point-in-time-correct-joins)
3. [Offline vs. Online Feature Stores](#3-offline-vs-online-feature-stores)
4. [Feature Computation Patterns](#4-feature-computation-patterns)
5. [Feast Integration](#5-feast-integration)
6. [Custom Lightweight Feature Store](#6-custom-lightweight-feature-store)
7. [Feature Versioning and Metadata](#7-feature-versioning-and-metadata)

---

## 1. Why Feature Stores?

### 1.1 The Duplicate Effort Problem

```
Without a feature store:

  Data Scientist:
    notebook_training.py → builds lag features, rolling means, calendar feats
    → trains model → results look great

  Engineer:
    serving_service.py → also computes features → slightly different code
    → different pandas version → different rolling window defaults
    → TRAINING-SERVING SKEW

  Another team:
    different_model.py → recomputes same features from scratch
    → 40% of all feature computation is duplicated across teams

Feature store solves:
  ✅ Single feature computation logic → reused everywhere
  ✅ Point-in-time correct joins enforced by the store
  ✅ Feature lineage and metadata tracking
  ✅ Online store for < 10ms serving lookups
```

### 1.2 Feature Store Components

```
┌──────────────────────────────────────────────────────────┐
│                    FEATURE STORE                         │
├──────────────────┬───────────────────────────────────────┤
│  OFFLINE STORE   │          ONLINE STORE                 │
│  (historical)    │          (serving)                    │
│                  │                                       │
│  - Parquet/      │  - Redis / DynamoDB / BigTable        │
│    Delta Lake    │  - Key: (entity_id, feature_name)     │
│  - All history   │  - Value: latest feature value        │
│  - PIT joins     │  - Latency: < 10ms                    │
│  - Batch reads   │  - Updated by streaming or batch      │
│                  │                                       │
├──────────────────┴───────────────────────────────────────┤
│  FEATURE REGISTRY (metadata)                             │
│  - Feature definitions, owners, descriptions             │
│  - Data types, expected ranges                           │
│  - Freshness SLAs                                        │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Point-in-Time Correct Joins

### 2.1 The Problem

```
Training dataset:
  entity_id | timestamp    | label (future)
  store_001 | 2024-01-01   | sales_next_week

Feature lookup (WRONG — uses future data!):
  SELECT * FROM features WHERE entity_id='store_001' AND date='2024-01-01'
  → Uses feature values computed ON 2024-01-01, but those values
    include data from after 2024-01-01 if features are computed lazily!

Feature lookup (CORRECT — point-in-time):
  For label at 2024-01-01, use the feature row with:
    max(feature_timestamp) WHERE feature_timestamp <= 2024-01-01
  → Only features available BEFORE the label observation
```

### 2.2 Implementation

```python
import pandas as pd
import numpy as np
from typing import List

def point_in_time_join(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    entity_col: str,
    label_timestamp_col: str,
    feature_timestamp_col: str,
    feature_cols: List[str],
    max_feature_age: pd.Timedelta = None,
) -> pd.DataFrame:
    """
    Perform a point-in-time correct join between labels and features.

    For each (entity, label_time) in labels, find the most recent feature row
    with feature_timestamp <= label_timestamp.

    Parameters
    ----------
    labels                 : DataFrame with entity_col + label_timestamp_col + target
    features               : DataFrame with entity_col + feature_timestamp_col + feature_cols
    entity_col             : join key (e.g., 'store_id', 'sensor_id')
    label_timestamp_col    : timestamp column in labels (prediction time)
    feature_timestamp_col  : timestamp column in features (feature creation time)
    feature_cols           : columns to retrieve from features
    max_feature_age        : reject features older than this (e.g., pd.Timedelta('7D'))

    Returns
    -------
    joined : labels merged with features (PIT correct)
    """
    labels   = labels.copy()
    features = features.copy()

    labels[label_timestamp_col]     = pd.to_datetime(labels[label_timestamp_col])
    features[feature_timestamp_col] = pd.to_datetime(features[feature_timestamp_col])

    # Sort features for merge_asof
    features = features.sort_values([entity_col, feature_timestamp_col])

    result_frames = []

    for entity, entity_labels in labels.groupby(entity_col):
        entity_features = features[features[entity_col] == entity].copy()
        entity_labels   = entity_labels.sort_values(label_timestamp_col)

        # Backward asof merge: for each label_time, get latest feature before it
        merged = pd.merge_asof(
            entity_labels,
            entity_features[[feature_timestamp_col] + feature_cols],
            left_on=label_timestamp_col,
            right_on=feature_timestamp_col,
            direction="backward",
        )

        # Apply max feature age constraint
        if max_feature_age is not None:
            age = merged[label_timestamp_col] - merged[feature_timestamp_col]
            too_old = age > max_feature_age
            merged.loc[too_old, feature_cols] = np.nan

        result_frames.append(merged)

    return pd.concat(result_frames, ignore_index=True)


def demonstrate_pit_leakage():
    """
    Show the difference between naive join and PIT-correct join.
    """
    # Features computed daily
    features = pd.DataFrame({
        "store_id":       ["A", "A", "A", "A"],
        "feature_date":   pd.to_datetime(["2024-01-01", "2024-01-07", "2024-01-14", "2024-01-21"]),
        "rolling_mean_7": [100.0, 110.0, 105.0, 115.0],
    })

    # Labels at specific dates (we want to predict sales)
    labels = pd.DataFrame({
        "store_id":   ["A", "A", "A"],
        "pred_date":  pd.to_datetime(["2024-01-03", "2024-01-10", "2024-01-17"]),
        "sales":      [105.0, 112.0, 108.0],
    })

    # WRONG: inner merge on nearest date (might use future features)
    wrong = pd.merge_asof(
        labels.sort_values("pred_date"),
        features.sort_values("feature_date"),
        left_on="pred_date", right_on="feature_date",
        by="store_id", direction="nearest",   # ← uses nearest, not backward!
    )
    print("Wrong join (nearest — can use future features):")
    print(wrong[["pred_date", "feature_date", "rolling_mean_7"]].to_string(index=False))

    # CORRECT: backward-only merge
    correct = pd.merge_asof(
        labels.sort_values("pred_date"),
        features.sort_values("feature_date"),
        left_on="pred_date", right_on="feature_date",
        by="store_id", direction="backward",  # ← only look backward!
    )
    print("\nCorrect PIT join (backward — only past features):")
    print(correct[["pred_date", "feature_date", "rolling_mean_7"]].to_string(index=False))
```

---

## 3. Offline vs. Online Feature Stores

### 3.1 Two-Store Architecture

```
OFFLINE STORE (training + batch inference):
  Storage:   Parquet / Delta Lake / BigQuery
  Access:    SQL queries, DataFrame API, batch reads
  Use case:  Training dataset generation, batch scoring, backfill
  Freshness: Hourly to daily updates
  Latency:   Seconds to minutes (acceptable for batch)

ONLINE STORE (real-time serving):
  Storage:   Redis / DynamoDB / Bigtable / MongoDB
  Access:    Key-value lookup: get(entity_id) → {feature: value}
  Use case:  Low-latency inference (< 10ms feature lookup)
  Freshness: Near real-time (streaming or micro-batch)
  Latency:   < 10ms (P99 requirement for real-time serving)

Sync:
  Offline → Online:
    Option A: Streaming pipeline (Flink / Spark Streaming)
               Updates online store as new events arrive
    Option B: Micro-batch job (every 5–15 minutes)
               Reads latest offline features → writes to Redis
```

### 3.2 Redis-Backed Online Feature Store

```python
import json
import numpy as np
from typing import Dict, Any, List

class OnlineFeatureStore:
    """
    Redis-backed online feature store for low-latency serving.

    Key format: f"{feature_view}:{entity_id}"
    Value: JSON with feature values + timestamp

    Requires: pip install redis
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        default_ttl: int = 86400,   # 24 hours default TTL
    ):
        try:
            import redis
            self._client = redis.Redis(host=host, port=port, db=db,
                                       decode_responses=True)
            self._client.ping()
            print(f"Connected to Redis at {host}:{port}")
        except Exception as e:
            print(f"Redis not available ({e}) — using in-memory fallback")
            self._client = None
        self._memory = {}   # fallback in-memory store
        self.default_ttl = default_ttl

    def _key(self, feature_view: str, entity_id: str) -> str:
        return f"feat:{feature_view}:{entity_id}"

    def write_features(
        self,
        feature_view: str,
        entity_id: str,
        features: Dict[str, Any],
        timestamp: str = None,
        ttl: int = None,
    ) -> None:
        """Write feature row to online store."""
        import datetime
        payload = {**features, "_ts": timestamp or datetime.datetime.utcnow().isoformat()}
        key     = self._key(feature_view, entity_id)
        ttl_    = ttl or self.default_ttl

        if self._client:
            self._client.setex(key, ttl_, json.dumps(payload))
        else:
            self._memory[key] = payload

    def read_features(
        self,
        feature_view: str,
        entity_id: str,
        feature_names: List[str] = None,
    ) -> Dict[str, Any]:
        """Read latest features for entity from online store."""
        key = self._key(feature_view, entity_id)

        if self._client:
            raw = self._client.get(key)
            payload = json.loads(raw) if raw else {}
        else:
            payload = self._memory.get(key, {})

        if feature_names:
            return {k: payload.get(k) for k in feature_names}
        return {k: v for k, v in payload.items() if not k.startswith("_")}

    def batch_write(
        self,
        feature_view: str,
        df: "pd.DataFrame",
        entity_col: str,
        feature_cols: List[str],
        timestamp_col: str = None,
    ) -> None:
        """Batch write DataFrame rows to online store (pipeline for efficiency)."""
        import pandas as pd

        if self._client:
            pipe = self._client.pipeline()
            for _, row in df.iterrows():
                entity_id = str(row[entity_col])
                features  = {col: row[col] for col in feature_cols
                             if col in row.index and pd.notna(row[col])}
                ts    = str(row[timestamp_col]) if timestamp_col else None
                payload = {**features}
                if ts: payload["_ts"] = ts
                key  = self._key(feature_view, entity_id)
                pipe.setex(key, self.default_ttl, json.dumps(payload))
            pipe.execute()
        else:
            for _, row in df.iterrows():
                entity_id = str(row[entity_col])
                features  = {col: float(row[col]) for col in feature_cols if col in row.index}
                self._memory[self._key(feature_view, entity_id)] = features

        print(f"Wrote {len(df)} rows to feature view '{feature_view}'")
```

---

## 4. Feature Computation Patterns

### 4.1 Backfill vs. Streaming

```python
import pandas as pd
import numpy as np

def backfill_ts_features(
    raw_df: pd.DataFrame,
    entity_col: str,
    timestamp_col: str,
    value_col: str,
    lags: list = None,
    rolling_windows: list = None,
) -> pd.DataFrame:
    """
    Backfill historical features for the offline store.

    Computes lag/rolling features for all historical timestamps.
    Strictly backward-looking — no future data leakage.

    Parameters
    ----------
    raw_df         : raw time series DataFrame (sorted by entity + timestamp)
    entity_col     : entity identifier column
    timestamp_col  : timestamp column
    value_col      : value column to compute features from
    lags           : list of lag periods
    rolling_windows: list of rolling window sizes

    Returns
    -------
    features_df : (entity, timestamp) → feature values
    """
    lags             = lags or [1, 2, 3, 7, 14]
    rolling_windows  = rolling_windows or [7, 30]

    all_features = []

    for entity, group in raw_df.groupby(entity_col):
        g = group.sort_values(timestamp_col).copy()
        s = g.set_index(timestamp_col)[value_col]

        feat = pd.DataFrame({timestamp_col: s.index, entity_col: entity, value_col: s.values},
                             index=s.index)

        # Lag features
        for lag in lags:
            feat[f"lag_{lag}"] = s.shift(lag).values

        # Rolling features (shift(1) ensures current period not included)
        for w in rolling_windows:
            rol = s.shift(1).rolling(w, min_periods=max(2, w//4))
            feat[f"roll_mean_{w}"] = rol.mean().values
            feat[f"roll_std_{w}"]  = rol.std().values
            feat[f"roll_max_{w}"]  = rol.max().values
            feat[f"roll_min_{w}"]  = rol.min().values

        all_features.append(feat.reset_index(drop=True))

    return pd.concat(all_features, ignore_index=True)
```

---

## 5. Feast Integration

### 5.1 Feast Feature Definitions

```python
# feature_repo/features.py
# pip install feast

from feast import (
    Entity, FeatureView, Feature, FileSource, ValueType
)
from datetime import timedelta

# Define entity (what we're predicting for)
store = Entity(
    name="store_id",
    value_type=ValueType.STRING,
    description="Retail store identifier",
)

# Define source (offline store)
store_ts_source = FileSource(
    path="data/features/store_ts_features.parquet",
    event_timestamp_column="feature_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define feature view (group of related features)
store_ts_features = FeatureView(
    name="store_ts_features",
    entities=["store_id"],
    ttl=timedelta(days=7),
    features=[
        Feature(name="lag_1",        dtype=ValueType.FLOAT),
        Feature(name="lag_7",        dtype=ValueType.FLOAT),
        Feature(name="roll_mean_7",  dtype=ValueType.FLOAT),
        Feature(name="roll_std_7",   dtype=ValueType.FLOAT),
        Feature(name="roll_mean_30", dtype=ValueType.FLOAT),
    ],
    batch_source=store_ts_source,
    online=True,
)
```

### 5.2 Feast Training Dataset Generation

```python
def feast_get_training_data(
    store_path: str,
    feature_views: list,
    entity_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """
    Generate training dataset using Feast (PIT-correct feature retrieval).

    entity_df should have columns:
      - entity_id column (e.g., 'store_id')
      - 'event_timestamp' (the label timestamp — features will be fetched
        using values available strictly before this timestamp)
    """
    try:
        from feast import FeatureStore
        fs   = FeatureStore(repo_path=store_path)
        data = fs.get_historical_features(
            entity_df=entity_df,
            features=[f"{fv}:{col}" for fv, cols in feature_views for col in cols],
        ).to_df()
        return data
    except ImportError:
        print("Feast not installed: pip install feast")
        return entity_df


def feast_get_online_features(
    store_path: str,
    feature_view: str,
    feature_cols: list,
    entity_rows: list,
) -> list:
    """
    Retrieve features from Feast online store for real-time serving.

    entity_rows: [{"store_id": "A"}, {"store_id": "B"}, ...]

    Returns list of feature dicts (same order as entity_rows).
    """
    try:
        from feast import FeatureStore
        fs = FeatureStore(repo_path=store_path)
        response = fs.get_online_features(
            features=[f"{feature_view}:{col}" for col in feature_cols],
            entity_rows=entity_rows,
        )
        return response.to_dict()
    except ImportError:
        print("Feast not installed: pip install feast")
        return []
```

---

## 6. Custom Lightweight Feature Store

```python
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

class LightweightFeatureStore:
    """
    Lightweight file-based feature store for small/medium projects.

    Offline store: Parquet files partitioned by date
    Online store:  In-memory dict (or Redis for production)
    """

    def __init__(self, base_path: str = "feature_store"):
        self.base    = Path(base_path)
        self.offline = self.base / "offline"
        self.meta    = self.base / "metadata"
        self.offline.mkdir(parents=True, exist_ok=True)
        self.meta.mkdir(parents=True, exist_ok=True)
        self._online: dict = {}

    def write_offline(
        self,
        feature_view: str,
        df: pd.DataFrame,
        partition_col: str = "date",
    ) -> None:
        """Write to offline store (partitioned parquet)."""
        path = self.offline / feature_view
        path.mkdir(exist_ok=True)
        out  = path / f"features.parquet"
        if out.exists():
            existing = pd.read_parquet(out)
            df       = pd.concat([existing, df]).drop_duplicates()
        df.to_parquet(out, index=False)
        print(f"Wrote {len(df)} rows to offline/{feature_view}")

    def read_offline(self, feature_view: str, start=None, end=None) -> pd.DataFrame:
        """Read from offline store with optional time filter."""
        path = self.offline / feature_view / "features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No offline store for '{feature_view}'")
        df = pd.read_parquet(path)
        if start: df = df[df["timestamp"] >= pd.to_datetime(start)]
        if end:   df = df[df["timestamp"] <= pd.to_datetime(end)]
        return df

    def materialize_to_online(self, feature_view: str, entity_col: str) -> None:
        """Sync latest offline features to online store."""
        df = self.read_offline(feature_view)
        latest = df.groupby(entity_col).last().reset_index()
        self._online[feature_view] = latest.set_index(entity_col).to_dict(orient="index")
        print(f"Materialized {len(latest)} entities to online store '{feature_view}'")

    def get_online_features(self, feature_view: str, entity_id: str) -> dict:
        """Get latest features for entity from online store (< 1ms)."""
        view = self._online.get(feature_view, {})
        return view.get(str(entity_id), {})

    def register_feature_view(self, name: str, meta: dict) -> None:
        """Register feature view metadata."""
        meta_path = self.meta / f"{name}.json"
        with open(meta_path, "w") as f:
            json.dump({**meta, "name": name}, f, indent=2, default=str)

    def get_training_dataset(
        self,
        feature_view: str,
        labels: pd.DataFrame,
        entity_col: str,
        label_timestamp_col: str,
        feature_cols: list,
    ) -> pd.DataFrame:
        """Generate PIT-correct training dataset from offline store."""
        features = self.read_offline(feature_view)
        return point_in_time_join(
            labels=labels,
            features=features,
            entity_col=entity_col,
            label_timestamp_col=label_timestamp_col,
            feature_timestamp_col="timestamp",
            feature_cols=feature_cols,
        )
```

---

## 7. Feature Versioning and Metadata

```python
import json
from datetime import datetime
from pathlib import Path

class FeatureMetadata:
    """
    Track feature definitions, owners, and lineage.
    """

    def __init__(self, store_path: str = "feature_store"):
        self.meta_dir = Path(store_path) / "metadata"
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        name: str,
        feature_type: str,         # 'lag', 'rolling', 'calendar', 'derived'
        computation: str,          # human-readable description
        inputs: list,              # upstream raw columns required
        owner: str = "unknown",
        dtype: str = "float64",
        expected_range: tuple = None,
    ) -> None:
        meta = {
            "name":           name,
            "type":           feature_type,
            "computation":    computation,
            "inputs":         inputs,
            "owner":          owner,
            "dtype":          dtype,
            "expected_range": expected_range,
            "registered_at":  datetime.utcnow().isoformat(),
        }
        path = self.meta_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Registered feature: {name}")

    def describe(self, name: str) -> dict:
        path = self.meta_dir / f"{name}.json"
        if not path.exists():
            raise KeyError(f"Feature '{name}' not registered")
        with open(path) as f:
            return json.load(f)

    def list_features(self) -> list:
        return [f.stem for f in self.meta_dir.glob("*.json")]


# Example registration
if __name__ == "__main__":
    meta = FeatureMetadata()
    meta.register("lag_7",       "lag",     "value shifted by 7 periods",      ["value"])
    meta.register("roll_mean_30","rolling", "28-period rolling mean of value",  ["value"])
    meta.register("is_weekend",  "calendar","1 if Saturday or Sunday else 0",   ["timestamp"])
    print("Registered features:", meta.list_features())
```

---

*← [01 — Pipeline Architecture](./01_ts_pipeline_architecture.md) | [Module README](./README.md) | Next: [03 — Model Registry](./03_model_registry_and_versioning.md) →*
