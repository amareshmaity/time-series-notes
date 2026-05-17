# 03 — Retail Sales Forecasting

> **Module**: 13 Projects & Case Studies | **Project**: 3 of 5
> **Domain**: Retail / Supply Chain | **Problem**: Many-series global model, intermittent demand, hierarchical
>
> Retail forecasting at scale is fundamentally different from single-series forecasting. With 50,000+ SKUs, per-SKU models are impractical. The industry answer is the **global model**: one LightGBM trained on all SKUs with item-specific lag features — the approach that won the M5 Competition.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [M5 Dataset Overview](#2-m5-dataset-overview)
3. [Feature Engineering at Scale](#3-feature-engineering-at-scale)
4. [Global LightGBM Model](#4-global-lightgbm-model)
5. [Intermittent Demand](#5-intermittent-demand)
6. [Walk-Forward Backtesting at Scale](#6-walk-forward-backtesting-at-scale)
7. [Hierarchical Reconciliation](#7-hierarchical-reconciliation)
8. [MLflow Tracking](#8-mlflow-tracking)
9. [Key Lessons](#9-key-lessons)

---

## 1. Problem Definition

```
Business goal:
  Forecast daily unit sales for 50,000+ SKUs across 10 stores
  for next 28 days (M5 horizon).

Hierarchy:
  Item → Store → State → National (4 levels)
  Category, Department levels also exist.

KPIs:
  WRMSSE: Weighted Root Mean Squared Scaled Error (M5 metric)
  In-stock rate: < X% stockouts from over-forecasting
  Waste rate:    < Y% over-ordering from under-forecasting

Dataset: M5 Forecasting Accuracy (Kaggle)
  - 1941 days of sales (2011-2016)
  - 30,490 bottom-level series
  - 3,049 unique items × 10 stores
  - Includes: events, SNAP food stamps, sell prices

Constraints:
  ✓ 28-day ahead forecast
  ✓ Point + quantile forecasts required
  ✓ Training time < 4 hours (full dataset)
  ✓ Single global model (not per-SKU)
```

---

## 2. M5 Dataset Overview

```python
import pandas as pd
import numpy as np

def generate_retail_data(
    n_items: int = 200,
    n_stores: int = 3,
    n_days: int = 400,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic retail data mimicking M5 structure.
    
    Returns dict with:
      sales_df : (n_series × n_days) wide format
      prices   : item × store price table
      calendar : date-level features (events, weekday)
    """
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    
    # Calendar
    calendar = pd.DataFrame({
        "date":       dates,
        "wday":       dates.dayofweek,
        "month":      dates.month,
        "year":       dates.year,
        "snap":       np.random.binomial(1, 0.3, n_days),   # SNAP day
        "event_name": "",
    })
    # Add a few sporting/cultural events
    event_idx = np.random.choice(n_days, size=20, replace=False)
    calendar.loc[event_idx, "event_name"] = "Event"

    # Prices (item × store)
    items  = [f"ITEM_{i:04d}" for i in range(n_items)]
    stores = [f"STORE_{s}" for s in range(n_stores)]
    base_prices = np.random.uniform(1.5, 25.0, n_items)
    prices = pd.DataFrame({
        "item_id":   np.repeat(items, n_stores),
        "store_id":  stores * n_items,
        "price":     np.repeat(base_prices, n_stores) * np.random.uniform(0.9, 1.1, n_items*n_stores),
    })

    # Sales matrix (wide: rows=series, cols=days)
    n_series = n_items * n_stores
    sales    = np.zeros((n_series, n_days), dtype=np.float32)
    row_meta = []

    for i, item in enumerate(items):
        cat   = f"CAT_{i % 3}"
        dept  = f"DEPT_{i % 6}"
        for s, store in enumerate(stores):
            idx       = i * n_stores + s
            base      = base_prices[i] * np.random.uniform(0.5, 5.0)  # base demand
            trend     = np.linspace(0, np.random.uniform(-0.2, 0.3), n_days)
            weekly    = np.array([1.0 if d.dayofweek >= 5 else 0.7 for d in dates])
            price_eff = np.random.uniform(0.8, 1.0, n_days)

            raw = base * (1 + trend) * weekly * price_eff * np.random.exponential(1.0, n_days)
            # Intermittency: some items have sparse demand
            if np.random.rand() < 0.3:
                raw *= np.random.binomial(1, 0.6, n_days)

            sales[idx] = np.round(np.maximum(raw, 0)).astype(np.float32)
            row_meta.append({
                "id": f"{item}_{store}",
                "item_id": item, "store_id": store,
                "cat_id": cat, "dept_id": dept,
            })

    sales_df = pd.DataFrame(sales, columns=[str(d.date()) for d in dates])
    meta_df  = pd.DataFrame(row_meta)
    return {"sales": sales_df, "meta": meta_df, "calendar": calendar, "prices": prices}


def retail_eda(data: dict, sample_ids: int = 4) -> None:
    """Quick EDA: demand distribution, zeros fraction, weekly seasonality."""
    sales = data["sales"].values
    meta  = data["meta"]
    n_days = sales.shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Demand distribution (log-scale)
    ax = axes[0]
    flat = sales[sales > 0].flatten()
    ax.hist(flat, bins=50, color="#2196F3", edgecolor="white", alpha=0.8)
    ax.set_yscale("log"); ax.set_title("Non-Zero Demand Distribution"); ax.grid(alpha=0.3)

    # Intermittency: fraction of zeros per series
    ax = axes[1]
    zero_frac = (sales == 0).mean(axis=1)
    ax.hist(zero_frac, bins=30, color="#FF5722", edgecolor="white", alpha=0.8)
    ax.set_title(f"Intermittency — Zero Fraction per Series\n"
                 f"(median={np.median(zero_frac):.2f})")
    ax.set_xlabel("Fraction of zero-sales days"); ax.grid(alpha=0.3)

    # Weekly seasonality (average per weekday)
    ax = axes[2]
    import matplotlib.pyplot as plt
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    daily_total = sales.sum(axis=0)
    weekly = pd.Series(daily_total, index=dates).groupby(lambda d: d.dayofweek).mean()
    days   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    ax.bar(days, weekly.values, color="#4CAF50")
    ax.set_title("Average Total Sales by Weekday"); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout(); plt.show()
    print(f"\nDataset: {sales.shape[0]} series × {n_days} days")
    print(f"Overall zero fraction: {(sales == 0).mean():.3f}")
```

---

## 3. Feature Engineering at Scale

```python
def build_retail_features(
    sales_wide: pd.DataFrame,
    meta_df:    pd.DataFrame,
    calendar:   pd.DataFrame,
    prices:     pd.DataFrame,
    lags:       list = None,
    rolls:      list = None,
) -> pd.DataFrame:
    """
    Build long-format feature DataFrame for all series.
    Converts wide (series × days) to long (rows = series×day pairs).

    Memory note: for 30,490 × 1941 = ~59M rows, use chunked processing.
    For this demo, we work with a smaller subset.
    """
    lags  = lags  or [7, 14, 21, 28, 35]
    rolls = rolls or [7, 28]

    # Melt to long format
    sales_wide = sales_wide.copy()
    sales_wide["id"] = meta_df["id"].values
    df_long = sales_wide.melt(id_vars="id", var_name="date", value_name="sales")
    df_long["date"] = pd.to_datetime(df_long["date"])
    df_long = df_long.sort_values(["id","date"]).reset_index(drop=True)

    # Merge metadata
    df_long = df_long.merge(meta_df[["id","item_id","store_id","cat_id","dept_id"]], on="id")

    # Calendar features
    df_long = df_long.merge(calendar.rename(columns={"date":"date"}), on="date", how="left")

    # Lag features (within each series)
    for lag in lags:
        df_long[f"lag_{lag}"] = df_long.groupby("id")["sales"].shift(lag)

    # Rolling mean/std
    for w in rolls:
        df_long[f"rmean_{w}"] = (df_long.groupby("id")["sales"]
                                   .transform(lambda x: x.shift(7).rolling(w, min_periods=1).mean()))
        df_long[f"rstd_{w}"]  = (df_long.groupby("id")["sales"]
                                   .transform(lambda x: x.shift(7).rolling(w, min_periods=1).std()))

    # Encode categoricals as integers
    for col in ["id","item_id","store_id","cat_id","dept_id"]:
        df_long[col] = df_long[col].astype("category").cat.codes

    return df_long.dropna()
```

---

## 4. Global LightGBM Model

```python
def train_global_lgbm(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "sales",
    n_valid_days: int = 28,
) -> dict:
    """
    Global LightGBM model trained on ALL series simultaneously.
    
    Key design decisions:
      - All series share the same model weights
      - Series identity (item_id, store_id) are features → series-specific intercepts
      - Lag features provide series-specific temporal context
      - One model = fast training, natural cross-series learning
    """
    try:
        import lightgbm as lgb
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        print("LightGBM not installed — demo only")
        return {}

    # Walk-forward train/val split (last 28 days = validation)
    cutoff   = df["date"].max() - pd.Timedelta(days=n_valid_days)
    df_train = df[df["date"] <= cutoff]
    df_val   = df[df["date"] >  cutoff]

    X_tr = df_train[feature_cols].values
    y_tr = df_train[target_col].values
    X_va = df_val[feature_cols].values
    y_va = df_val[target_col].values

    model = lgb.LGBMRegressor(
        n_estimators     = 1000,
        learning_rate    = 0.05,
        max_depth        = 7,
        num_leaves       = 63,
        min_child_samples= 50,
        feature_fraction = 0.8,
        bagging_fraction = 0.8,
        bagging_freq     = 1,
        reg_alpha        = 0.1,
        reg_lambda       = 0.1,
        random_state     = 42,
        verbose          = -1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    preds = model.predict(X_va)
    preds = np.maximum(preds, 0)   # sales cannot be negative

    mae  = float(np.mean(np.abs(y_va - preds)))
    rmse = float(np.sqrt(np.mean((y_va - preds)**2)))
    zero_mask = y_va > 0
    mape = float(np.mean(np.abs((y_va[zero_mask] - preds[zero_mask]) / y_va[zero_mask])) * 100)

    print(f"\nGlobal LightGBM:")
    print(f"  Val MAE:  {mae:.4f}")
    print(f"  Val RMSE: {rmse:.4f}")
    print(f"  Val MAPE (non-zero): {mape:.2f}%")
    print(f"  Best iteration: {model.best_iteration_}")

    # Feature importance (top 15)
    fi = pd.Series(model.feature_importances_, index=feature_cols)
    fi = fi.sort_values(ascending=False)[:15]
    print(f"\nTop features:\n{fi.to_string()}")

    return {"model": model, "mae": mae, "rmse": rmse, "mape": mape,
            "feature_importance": fi}
```

---

## 5. Intermittent Demand

```python
def croston_tsb(
    series: np.ndarray,
    alpha_d: float = 0.1,
    alpha_p: float = 0.1,
) -> np.ndarray:
    """
    Croston's TSB (Teunter-Syntetos-Babai) method for intermittent demand.

    Standard Croston: smooths demand and inter-arrival separately.
    TSB variant:      smooths demand probability (avoids bias in Croston's).

    Returns one-step-ahead forecasts for the entire series.
    """
    n    = len(series)
    d    = np.zeros(n)   # smoothed demand (when non-zero)
    p    = np.zeros(n)   # smoothed demand probability
    preds = np.zeros(n)

    d[0] = series[0] if series[0] > 0 else 1.0
    p[0] = 1.0 if series[0] > 0 else 0.5

    for t in range(1, n):
        if series[t-1] > 0:
            d[t] = alpha_d * series[t-1] + (1-alpha_d) * d[t-1]
            p[t] = alpha_p * 1.0         + (1-alpha_p) * p[t-1]
        else:
            d[t] = d[t-1]
            p[t] = alpha_p * 0.0         + (1-alpha_p) * p[t-1]
        preds[t] = d[t] * p[t]

    return preds


def classify_demand_pattern(series: np.ndarray) -> str:
    """
    Classify demand pattern per Syntetos-Boylan-Croston matrix.

    Uses:
      ADI (Average Demand Interval) = avg time between demands
      CV² (Squared Coefficient of Variation of non-zero demand)

    Classification:
      ADI < 1.32, CV² < 0.49 → Smooth    → ETS / ARIMA
      ADI < 1.32, CV² ≥ 0.49 → Erratic   → LightGBM with sales lag features
      ADI ≥ 1.32, CV² < 0.49 → Lumpy     → Croston / TSB
      ADI ≥ 1.32, CV² ≥ 0.49 → Sporadic  → Croston TSB / Teunter
    """
    non_zero = series[series > 0]
    n        = len(series)

    if len(non_zero) == 0:
        return "Zero (no demand)"

    intervals = []
    last_t = 0
    for t in range(n):
        if series[t] > 0:
            intervals.append(t - last_t + 1)
            last_t = t

    adi  = float(np.mean(intervals)) if intervals else float(n)
    cv2  = float((non_zero.std() / (non_zero.mean() + 1e-12))**2)

    if adi < 1.32:
        pattern = "Smooth" if cv2 < 0.49 else "Erratic"
    else:
        pattern = "Lumpy" if cv2 < 0.49 else "Sporadic"

    return pattern
```

---

## 6. Walk-Forward Backtesting at Scale

```python
def wrmsse(y_true: np.ndarray, y_pred: np.ndarray,
           scale: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted Root Mean Squared Scaled Error — the M5 competition metric.

    RMSSE_i = sqrt(mean((y_i - ŷ_i)²) / scale_i)
    WRMSSE  = sum_i (weight_i × RMSSE_i)

    scale_i   = mean squared difference of naive forecast on training data
    weight_i  = revenue share of series i
    """
    n_series = y_true.shape[0]
    rmsse    = np.zeros(n_series)

    for i in range(n_series):
        mse      = np.mean((y_true[i] - y_pred[i])**2)
        sc       = max(scale[i], 1e-8)
        rmsse[i] = np.sqrt(mse / sc)

    return float(np.sum(weights * rmsse))


def rolling_backtest(
    df:           pd.DataFrame,
    feature_cols: list,
    model,
    n_windows:    int = 3,
    horizon:      int = 28,
) -> pd.DataFrame:
    """
    Rolling walk-forward backtest for global model.
    Each window shifts the train/val cutoff by one horizon.
    """
    results = []
    max_date = df["date"].max()

    for w in range(n_windows):
        cutoff = max_date - pd.Timedelta(days=horizon * (n_windows - w))
        df_tr  = df[df["date"] <= cutoff]
        df_va  = df[(df["date"] > cutoff) & (df["date"] <= cutoff + pd.Timedelta(days=horizon))]

        if len(df_va) == 0:
            continue

        preds  = np.maximum(model.predict(df_va[feature_cols].values), 0)
        actual = df_va["sales"].values
        mae    = float(np.mean(np.abs(actual - preds)))

        results.append({"window": w+1, "cutoff": str(cutoff.date()), "mae": mae, "n_samples": len(df_va)})
        print(f"  Window {w+1}: cutoff={cutoff.date()}, MAE={mae:.4f}, n={len(df_va)}")

    return pd.DataFrame(results)
```

---

## 7. Hierarchical Reconciliation

```python
def aggregate_forecasts(
    item_store_preds: pd.DataFrame,
    hierarchy_cols:   list = None,
) -> pd.DataFrame:
    """
    Bottom-up aggregation from item×store → dept → cat → store → national.
    Ensures: sum of parts = whole at each level.
    """
    hierarchy_cols = hierarchy_cols or ["cat_id","dept_id","store_id"]
    results = [item_store_preds.copy()]

    for level in hierarchy_cols:
        agg = item_store_preds.groupby([level,"date"])["forecast"].sum().reset_index()
        agg["id"] = agg[level]
        results.append(agg[["id","date","forecast"]])

    national = item_store_preds.groupby("date")["forecast"].sum().reset_index()
    national["id"] = "NATIONAL"
    results.append(national)

    return pd.concat(results, ignore_index=True)
```

---

## 8. MLflow Tracking

```python
def track_retail_experiment(
    model,
    feature_importance: pd.Series,
    metrics: dict,
    params: dict,
    run_name: str = "global_lgbm_v1",
) -> str:
    """Track retail model experiment with MLflow."""
    try:
        import mlflow
        import mlflow.lightgbm

        mlflow.set_experiment("retail_sales_forecasting")
        with mlflow.start_run(run_name=run_name) as run:
            # Log params
            mlflow.log_params(params)
            # Log metrics
            mlflow.log_metrics({k: round(float(v), 6) for k, v in metrics.items()})
            # Log feature importance as artifact
            fi_path = "feature_importance.csv"
            feature_importance.to_csv(fi_path)
            mlflow.log_artifact(fi_path)
            # Log model
            mlflow.lightgbm.log_model(model, "model")

        print(f"MLflow run ID: {run.info.run_id}")
        return run.info.run_id

    except ImportError:
        print("MLflow not installed: pip install mlflow")
        return ""
```

---

## 9. Key Lessons

```
LESSON 1: Global models outperform per-series models at scale.
  50,000 local ARIMA models = unmaintainable + poor on sparse series.
  1 global LightGBM = learns from all data + handles new items naturally.

LESSON 2: Intermittent demand requires special handling.
  Negative predictions (from LightGBM) → clip to 0.
  High-zero series → Croston/TSB or classify and treat separately.
  WRMSSE downweights low-revenue items appropriately.

LESSON 3: Lag 7 is king for weekly retail data.
  Sales last week, 2 weeks ago, 4 weeks ago → dominate feature importance.
  Include promotional flags and price as exogenous inputs.

LESSON 4: WRMSSE is not the same as MAE.
  Items matter by their revenue weight, not count.
  Optimizing global MAE can ignore high-value, hard-to-forecast SKUs.

LESSON 5: Encode all identifiers as integer categoricals.
  item_id, store_id, dept_id → LightGBM categorical features.
  This lets the model learn series-specific intercepts efficiently.

LESSON 6: Version everything from day one.
  Feature pipeline version, model version, training data cutoff
  → Track with MLflow. Retail data changes (new items, price resets)
  can silently corrupt a model trained on stale pipelines.
```

---

*← [02 — Energy](./02_energy_demand_forecasting.md) | [Module README](./README.md) | Next: [04 — Sensor Anomaly](./04_sensor_anomaly_detection.md) →*
