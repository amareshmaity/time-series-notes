# 03 — Global vs. Local Models

> **Module**: 07 Forecasting Strategies | **File**: 3 of 6
>
> The most impactful architectural decision in modern time series forecasting is whether to train one model per series (Local) or one model across all series simultaneously (Global). This choice determines scalability, cold-start capability, and cross-series generalization.

---

## Table of Contents

1. [The Core Trade-off](#1-the-core-trade-off)
2. [Local Models](#2-local-models)
3. [Global Models](#3-global-models)
4. [Pooling Strategies](#4-pooling-strategies)
5. [Handling Heterogeneous Series](#5-handling-heterogeneous-series)
6. [When to Use Each Approach](#6-when-to-use-each-approach)
7. [Implementation — Global LightGBM](#7-implementation--global-lightgbm)
8. [Implementation — Global Neural Forecast (Nixtla)](#8-implementation--global-neural-forecast-nixtla)

---

## 1. The Core Trade-off

### 1.1 The Problem with Local Models at Scale

Suppose you have **50,000 SKUs** to forecast weekly:
- **Local ARIMA** per series: 50,000 models to fit, tune, store, and retrain
- **Global LightGBM**: 1 model; trained on all series stacked into one dataset

The industry has shifted decisively toward global models since ~2018 (M4 competition, Walmart forecasting). The key insight:

> "A single model trained on many series can learn patterns no individual series has enough data to discover on its own."

### 1.2 Taxonomy

```
                    ┌─────────────────────────────────────┐
                    │       Forecasting Model Scope        │
                    └──────────────┬──────────────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               ▼                   ▼                   ▼
           Local                Grouped              Global
        (1 model /           (1 model /          (1 model /
         1 series)            cluster)           all series)
           │                    │                    │
         ARIMA               ARIMA per           LightGBM
         ETS                 segment             XGBoost
         Prophet             Prophet             LSTM
         (per SKU)           (per category)      N-BEATS
                                                 TFT
                                                 Chronos
```

---

## 2. Local Models

### 2.1 Characteristics

| Property              | Detail                                                        |
|-----------------------|---------------------------------------------------------------|
| **Training unit**     | Each series trained independently                            |
| **Parameters**        | Separate parameter set per series                             |
| **Cross-series info** | ❌ None — each model sees only its own history               |
| **Cold start**        | ❌ Cannot forecast new series without retraining              |
| **Scalability**       | ❌ O(N × fitting_cost)                                       |
| **Interpretability**  | ✅ Model directly describes one series                        |
| **When it wins**      | Few series, long history, distinct dynamics per series        |

### 2.2 Classic Local Models

```python
import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def fit_local_arima(series: pd.Series, order=(1,1,1), seasonal_order=(1,1,1,7)):
    """Fit a local SARIMA model to a single series."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False)
    return result


def fit_all_series_local(panel_df: pd.DataFrame, id_col: str, target_col: str) -> dict:
    """
    Fit one ARIMA model per unique series in a panel DataFrame.

    Parameters
    ----------
    panel_df   : DataFrame in long format (id_col, date_col, target_col)
    id_col     : column identifying each time series
    target_col : the value to forecast

    Returns
    -------
    dict of {series_id: fitted_model}
    """
    models = {}
    for uid, group in panel_df.groupby(id_col):
        series = group.set_index("ds")[target_col]
        try:
            models[uid] = fit_local_arima(series)
            print(f"  Fitted ARIMA for {uid}")
        except Exception as e:
            print(f"  ⚠ Failed for {uid}: {e}")
    return models
```

### 2.3 Local Model Scalability Problem

```python
# Illustrating the O(N × T) cost of local models
import time

def estimate_local_fitting_cost(n_series: int, series_length: int):
    """
    Rough cost estimate for fitting N local ARIMA models.
    Uses empirical timing (1 ARIMA per series ≈ 0.5–5 seconds).
    """
    ARIMA_SECONDS_PER_SERIES = 2.0  # conservative estimate
    total_seconds = n_series * ARIMA_SECONDS_PER_SERIES
    total_hours   = total_seconds / 3600

    print(f"N series          : {n_series:,}")
    print(f"Series length     : {series_length}")
    print(f"Estimated fit time: {total_hours:.1f} hours")
    print(f"Models to store   : {n_series:,}")

# 50,000 SKUs → ~27 hours just to fit
estimate_local_fitting_cost(50_000, 200)
```

---

## 3. Global Models

### 3.1 Characteristics

| Property              | Detail                                                         |
|-----------------------|----------------------------------------------------------------|
| **Training unit**     | All series stacked into one dataset                           |
| **Parameters**        | Single shared parameter set                                    |
| **Cross-series info** | ✅ Learned implicitly through shared patterns                 |
| **Cold start**        | ✅ Works on new series immediately (with static covariates)    |
| **Scalability**       | ✅ O(N × series_length) data, O(1) models                     |
| **Interpretability**  | Moderate — global feature importance, not per-series          |
| **When it wins**      | Many similar series, short histories, retail/demand forecasting |

### 3.2 The "Stacking" Paradigm

All series are converted to a flat table of (series_id, timestamp, features, target):

```
Local format (per series):
  t=1  y=10
  t=2  y=12
  t=3  y=15
  ...

Global format (stacked panel):
  series_id | t | lag_1 | lag_7 | month_sin | ... | y
  ──────────┼───┼───────┼───────┼───────────┼─────┼──
  SKU_A     | 8 | 10    | 8     | 0.5       | ... | 12
  SKU_A     | 9 | 12    | 9     | 0.6       | ... | 15
  SKU_B     | 8 | 200   | 180   | 0.5       | ... | 210
  SKU_B     | 9 | 210   | 190   | 0.6       | ... | 220
  ...       | . | ...   | ...   | ...       | ... | ...

→ One LightGBM model trained on ALL rows simultaneously
```

### 3.3 Why Global Models Work

```
1. Shared patterns: seasonality, trend, holiday effects are common across series
   → Global model has 50,000× more data to learn these patterns

2. Regularization by diversity: seeing many series prevents overfitting to noise

3. Transfer learning effect: cross-series knowledge improves forecasts for
   short or sparse series that wouldn't have enough data locally

4. Series ID as feature: the model can distinguish series while sharing structure
```

---

## 4. Pooling Strategies

Between pure local and pure global, **partial pooling** offers a middle ground:

### 4.1 Complete Pooling (Global)

One model, all series: the model ignores series identity (or treats it as a feature).

```python
# Series ID encoded as a numeric or categorical feature
panel_df["series_id_encoded"] = panel_df["unique_id"].astype("category").cat.codes
```

### 4.2 No Pooling (Local)

One model per series. Maximum flexibility, minimum data sharing.

### 4.3 Partial Pooling — Grouped Models

One model per cluster/segment — series are grouped by similar behavior:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

def cluster_series_for_grouped_models(
    panel_df: pd.DataFrame,
    id_col: str,
    target_col: str,
    n_clusters: int = 10,
    feature_length: int = 52,
) -> dict:
    """
    Cluster series by their recent behavior; train one model per cluster.

    Parameters
    ----------
    panel_df      : long-format DataFrame
    id_col        : series identifier column
    target_col    : target value column
    n_clusters    : number of groups/models
    feature_length: recent history length used for clustering

    Returns
    -------
    dict {series_id: cluster_label}
    """
    series_vectors = []
    ids = []

    for uid, group in panel_df.groupby(id_col):
        vals = group[target_col].values[-feature_length:]
        if len(vals) < feature_length:
            vals = np.pad(vals, (feature_length - len(vals), 0), mode="edge")
        series_vectors.append(vals)
        ids.append(uid)

    X = StandardScaler().fit_transform(np.array(series_vectors))
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)

    return dict(zip(ids, labels))
```

### 4.4 Partial Pooling — Hierarchical Bayesian (Concept)

```
Bayesian hierarchical model:
  μ_global ~ Normal(0, 10)          # global prior
  σ_global ~ HalfNormal(1)
  
  μ_series[i] ~ Normal(μ_global, σ_global)   # series-level, pulled toward global
  y[i,t] ~ Normal(μ_series[i] + f(X[i,t]), σ_noise)

→ Short series: pulled strongly toward global mean
→ Long series: uses mostly its own data
→ Balances local fit and global regularization automatically
```

---

## 5. Handling Heterogeneous Series

### 5.1 The Scale Problem

Series with wildly different scales cause gradient boosting trees to split on magnitude:

```python
import pandas as pd
import numpy as np

def normalize_panel(
    panel_df: pd.DataFrame,
    id_col: str,
    target_col: str,
    method: str = "robust",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize each series independently before global model training.

    Parameters
    ----------
    panel_df   : long-format panel DataFrame
    id_col     : series identifier column
    target_col : column to normalize
    method     : 'robust' (median/IQR) or 'minmax' or 'zscore'

    Returns
    -------
    normalized_df  : DataFrame with normalized target
    scalers_df     : DataFrame with per-series scaling parameters (for inverse transform)
    """
    results  = []
    scalers  = []

    for uid, group in panel_df.groupby(id_col):
        group = group.copy()
        vals  = group[target_col].values

        if method == "robust":
            median = np.median(vals)
            iqr    = np.percentile(vals, 75) - np.percentile(vals, 25)
            iqr    = iqr if iqr > 1e-8 else 1.0
            group[target_col] = (vals - median) / iqr
            scalers.append({"unique_id": uid, "center": median, "scale": iqr})

        elif method == "zscore":
            mean = vals.mean()
            std  = vals.std() or 1.0
            group[target_col] = (vals - mean) / std
            scalers.append({"unique_id": uid, "center": mean, "scale": std})

        elif method == "minmax":
            lo, hi = vals.min(), vals.max()
            rng    = hi - lo if hi - lo > 1e-8 else 1.0
            group[target_col] = (vals - lo) / rng
            scalers.append({"unique_id": uid, "center": lo, "scale": rng})

        results.append(group)

    return pd.concat(results, ignore_index=True), pd.DataFrame(scalers)


def inverse_normalize(preds: np.ndarray, scaler_row: pd.Series) -> np.ndarray:
    """Reverse the normalization to get predictions in original scale."""
    return preds * scaler_row["scale"] + scaler_row["center"]
```

### 5.2 Series-Level Static Features

Encode each series' characteristics as static features so the global model can adapt:

```python
def add_static_series_features(
    panel_df: pd.DataFrame,
    id_col: str,
    target_col: str,
) -> pd.DataFrame:
    """
    Compute and append per-series static features to every row of that series.

    These features inform the global model about the nature of each series:
    scale, volatility, trend direction, and sparsity.
    """
    panel_df = panel_df.copy()
    static_features = {}

    for uid, group in panel_df.groupby(id_col):
        vals = group[target_col].values
        static_features[uid] = {
            "series_mean":     vals.mean(),
            "series_std":      vals.std(),
            "series_cv":       vals.std() / (vals.mean() + 1e-8),  # coeff of variation
            "series_length":   len(vals),
            "series_sparsity": (vals == 0).mean(),          # fraction of zeros
            "series_trend":    np.polyfit(range(len(vals)), vals, deg=1)[0],  # slope
        }

    stats_df = pd.DataFrame(static_features).T.reset_index().rename(columns={"index": id_col})
    return panel_df.merge(stats_df, on=id_col, how="left")
```

---

## 6. When to Use Each Approach

```
Decision Tree:

How many series do you have?
├── < 10 series
│     → Local models (ARIMA, Prophet, ETS per series)
│       Each series has unique dynamics; pooling hurts
│
├── 10 – 1,000 series
│     → Grouped or Global models
│       Try Global first; fall back to grouped if series are very heterogeneous
│
└── > 1,000 series
      → Global model is standard
        Local is computationally infeasible for fitting + retraining
```

| Situation                          | Recommended Approach           |
|------------------------------------|--------------------------------|
| Few series, long histories         | Local (ARIMA, Prophet)         |
| Many short series (retail)         | Global (LightGBM, NHITS)       |
| New series arriving regularly      | Global (instant cold-start)    |
| Very heterogeneous series dynamics | Grouped (cluster first)        |
| High interpretability requirement  | Local (per-series coefficients)|
| Production at scale                | Global (one model, easy deploy)|

---

## 7. Implementation — Global LightGBM

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

class GlobalLGBMForecaster:
    """
    Global LightGBM forecaster trained on a panel of multiple time series.

    Stacks all series into one dataset with lag features + static series features.
    """

    def __init__(self, horizon: int = 12, lags: list = None):
        self.horizon = horizon
        self.lags    = lags or [1, 7, 14, 28]
        self.model_  = None
        self.scalers_: pd.DataFrame = None

    def _build_panel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build lag + calendar features for every series in the panel."""
        panels = []
        for uid, group in df.groupby("unique_id"):
            group = group.sort_values("ds").copy()
            t = "y"

            for lag in self.lags:
                group[f"lag_{lag}"] = group[t].shift(lag)

            s1 = group[t].shift(1)
            for w in [7, 14, 30]:
                group[f"roll{w}_mean"] = s1.rolling(w, min_periods=1).mean()
                group[f"roll{w}_std"]  = s1.rolling(w, min_periods=1).std()

            idx = pd.to_datetime(group["ds"])
            group["month_sin"]  = np.sin(2 * np.pi * idx.dt.month / 12)
            group["month_cos"]  = np.cos(2 * np.pi * idx.dt.month / 12)
            group["dow_sin"]    = np.sin(2 * np.pi * idx.dt.dayofweek / 7)
            group["dow_cos"]    = np.cos(2 * np.pi * idx.dt.dayofweek / 7)
            group["is_weekend"] = (idx.dt.dayofweek >= 5).astype(int)

            panels.append(group)

        return pd.concat(panels, ignore_index=True).dropna()

    def fit(self, df: pd.DataFrame) -> "GlobalLGBMForecaster":
        """
        Fit global model on panel DataFrame in Nixtla long format.

        Parameters
        ----------
        df : DataFrame with columns [unique_id, ds, y, (optional static features)]
        """
        panel = self._build_panel_features(df)

        feature_cols = [c for c in panel.columns
                        if c not in ("unique_id", "ds", "y")]
        X = panel[feature_cols]
        y = panel["y"]

        self.feature_cols_ = feature_cols

        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05,
            num_leaves=63, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            objective="regression", verbose=-1,
        )

        # Time-safe split
        split = int(len(X) * 0.85)
        model.fit(
            X.iloc[:split], y.iloc[:split],
            eval_set=[(X.iloc[split:], y.iloc[split:])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        self.model_ = model
        print(f"Global model trained on {len(X):,} rows from {df['unique_id'].nunique()} series")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate h-step recursive forecasts for all series in df.

        Parameters
        ----------
        df : DataFrame in long format with full history of each series

        Returns
        -------
        DataFrame with columns [unique_id, ds, forecast]
        """
        all_preds = []

        for uid, group in df.groupby("unique_id"):
            group = group.sort_values("ds").copy()
            series = group["y"].values

            from collections import deque
            buffer = deque(series[-max(self.lags):], maxlen=max(self.lags))
            last_date = pd.to_datetime(group["ds"].max())
            freq = pd.infer_freq(group["ds"])

            for h in range(self.horizon):
                future_date = last_date + pd.tseries.frequencies.to_offset(freq) * (h + 1)
                lag_feats = {f"lag_{lag}": list(buffer)[-lag] for lag in self.lags}
                s1 = list(buffer)
                for w in [7, 14, 30]:
                    recent = s1[-w:]
                    lag_feats[f"roll{w}_mean"] = np.mean(recent)
                    lag_feats[f"roll{w}_std"]  = np.std(recent) if len(recent) > 1 else 0.0

                month  = future_date.month
                dow    = future_date.dayofweek
                lag_feats.update({
                    "month_sin":  np.sin(2 * np.pi * month / 12),
                    "month_cos":  np.cos(2 * np.pi * month / 12),
                    "dow_sin":    np.sin(2 * np.pi * dow / 7),
                    "dow_cos":    np.cos(2 * np.pi * dow / 7),
                    "is_weekend": int(dow >= 5),
                })

                X_pred = pd.DataFrame([lag_feats])[self.feature_cols_]
                pred   = self.model_.predict(X_pred)[0]
                buffer.append(pred)

                all_preds.append({"unique_id": uid, "ds": future_date, "forecast": pred})

        return pd.DataFrame(all_preds)
```

---

## 8. Implementation — Global Neural Forecast (Nixtla)

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATS

# Global neural models natively handle panels — no stacking needed
models = [
    NHITS(
        h=12,
        input_size=48,
        max_steps=500,
        stack_types=["identity", "trend", "seasonality"],
        n_freq_downsample=[2, 1, 1],
        random_seed=42,
    ),
    NBEATS(
        h=12,
        input_size=48,
        max_steps=500,
        stack_types=["trend", "seasonality"],
        random_seed=42,
    ),
]

nf = NeuralForecast(models=models, freq="M")
nf.fit(df=df)   # df has [unique_id, ds, y] columns — all series stacked

# Forecast all series simultaneously
forecasts = nf.predict()
# forecasts: DataFrame with [unique_id, ds, NHITS, NBEATS]

print(forecasts.head(10))
print(f"Series forecasted: {forecasts['unique_id'].nunique()}")
```

---

*← [02 — Multi-Step Forecasting](./02_multi_step_forecasting.md) | [Module README](./README.md) | Next: [04 — Hierarchical Forecasting](./04_hierarchical_forecasting.md) →*
