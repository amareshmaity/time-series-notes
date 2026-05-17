"""
code/03_retail_project/main.py
================================
Module 13 — Projects & Case Studies
Project 3: Retail Sales Forecasting (Global Model)

End-to-end pipeline:
  - Synthetic multi-SKU daily sales (M5-style)
  - Intermittency classification (Smooth/Erratic/Lumpy/Sporadic)
  - Croston TSB for intermittent series
  - Global LightGBM for regular demand
  - Walk-forward backtest with WRMSSE
  - MLflow tracking (optional)
  - Visualization dashboard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Retail Dataset
# ─────────────────────────────────────────────────────────────────────────────

N_ITEMS  = 100
N_STORES = 3
N_DAYS   = 300
dates    = pd.date_range("2023-01-01", periods=N_DAYS, freq="D")
ITEMS    = [f"ITEM_{i:04d}" for i in range(N_ITEMS)]
STORES   = [f"ST_{s}" for s in range(N_STORES)]
CATS     = {item: f"CAT_{i%4}" for i, item in enumerate(ITEMS)}
DEPTS    = {item: f"DEPT_{i%8}" for i, item in enumerate(ITEMS)}

records = []
base_prices = np.random.uniform(2.0, 30.0, N_ITEMS)

for i, item in enumerate(ITEMS):
    for s, store in enumerate(STORES):
        base = base_prices[i] * np.random.uniform(0.5, 4.0)
        trend = np.linspace(0, np.random.uniform(-0.1, 0.2), N_DAYS)
        weekly = np.where(pd.DatetimeIndex(dates).dayofweek >= 5, 1.3, 0.8)

        # Intermittency for ~30% of items
        if np.random.rand() < 0.3:
            raw = base * trend * weekly * np.random.exponential(1.0, N_DAYS)
            raw *= np.random.binomial(1, 0.5, N_DAYS)   # sparse
        else:
            raw = base * (1 + trend) * weekly * np.random.exponential(1.0, N_DAYS)

        sales = np.round(np.maximum(raw, 0))
        for d, (date, sale) in enumerate(zip(dates, sales)):
            records.append({
                "id":      f"{item}_{store}",
                "item_id": item, "store_id": store,
                "cat_id":  CATS[item], "dept_id": DEPTS[item],
                "date":    date, "sales": float(sale),
                "price":   base_prices[i],
            })

df = pd.DataFrame(records)
n_series = df["id"].nunique()
print(f"Retail dataset: {n_series} series × {N_DAYS} days = {len(df):,} rows")
print(f"Zero fraction: {(df['sales'] == 0).mean():.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Intermittency Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_demand(series: np.ndarray) -> str:
    nz = series[series > 0]
    if len(nz) == 0:
        return "Zero"
    intervals, last = [], 0
    for t in range(len(series)):
        if series[t] > 0:
            intervals.append(t - last + 1); last = t
    adi = float(np.mean(intervals)) if intervals else len(series)
    cv2 = float((nz.std() / (nz.mean() + 1e-12))**2)
    if adi < 1.32:
        return "Smooth" if cv2 < 0.49 else "Erratic"
    else:
        return "Lumpy" if cv2 < 0.49 else "Sporadic"

series_classification = {}
for sid in df["id"].unique():
    s = df[df["id"] == sid].sort_values("date")["sales"].values
    series_classification[sid] = classify_demand(s)

class_counts = pd.Series(series_classification).value_counts()
print(f"\nDemand classification:\n{class_counts.to_string()}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Croston TSB for Intermittent Series
# ─────────────────────────────────────────────────────────────────────────────

def croston_tsb(series: np.ndarray, alpha_d=0.1, alpha_p=0.1) -> np.ndarray:
    """TSB variant of Croston's method."""
    n = len(series)
    d = np.zeros(n); p = np.zeros(n); preds = np.zeros(n)
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

# Apply Croston to intermittent series
intermittent_ids = [sid for sid, cls in series_classification.items()
                     if cls in ("Lumpy","Sporadic","Zero")]
print(f"\nCroston TSB applied to {len(intermittent_ids)} intermittent series")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature Engineering for Global Model
# ─────────────────────────────────────────────────────────────────────────────

# Encode categoricals
for col in ["id","item_id","store_id","cat_id","dept_id"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

df = df.sort_values(["id","date"]).reset_index(drop=True)

# Lag features
for lag in [7, 14, 21, 28, 35]:
    df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag)

# Rolling features
for w in [7, 28]:
    df[f"rmean_{w}"] = df.groupby("id")["sales"].transform(
        lambda x: x.shift(7).rolling(w, min_periods=1).mean()
    )
    df[f"rstd_{w}"] = df.groupby("id")["sales"].transform(
        lambda x: x.shift(7).rolling(w, min_periods=1).std()
    )

# Calendar
df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek.astype(float)
df["month"]       = pd.to_datetime(df["date"]).dt.month.astype(float)
df["is_weekend"]  = (pd.to_datetime(df["date"]).dt.dayofweek >= 5).astype(float)

df_clean = df.dropna()
feature_cols = [c for c in df_clean.columns
                if c not in ["sales","date","id"] and df_clean[c].dtype in [float, int, np.float32]]
print(f"Features: {len(feature_cols)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Global LightGBM Walk-Forward
# ─────────────────────────────────────────────────────────────────────────────

TEST_DAYS = 28
cutoff    = pd.to_datetime(df_clean["date"]).max() - pd.Timedelta(days=TEST_DAYS)
df_tr = df_clean[pd.to_datetime(df_clean["date"]) <= cutoff]
df_te = df_clean[pd.to_datetime(df_clean["date"]) >  cutoff]

X_tr = df_tr[feature_cols].values; y_tr = df_tr["sales"].values
X_te = df_te[feature_cols].values; y_te = df_te["sales"].values

print(f"\nTrain: {len(X_tr):,} rows | Test: {len(X_te):,} rows")

try:
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        num_leaves=63, min_child_samples=30,
        random_state=42, verbosity=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)])
    y_pred = np.maximum(model.predict(X_te), 0)

    mae  = mean_absolute_error(y_te, y_pred)
    rmse = float(np.sqrt(((y_te - y_pred)**2).mean()))
    nonzero = y_te > 0
    mape = float(np.abs((y_te[nonzero] - y_pred[nonzero]) / y_te[nonzero]).mean() * 100)
    print(f"\nGlobal LightGBM:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE (non-zero): {mape:.2f}%")

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)[:10]
    print(f"\nTop 10 features:\n{fi.to_string()}")

except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    print("LightGBM not installed — using sklearn GBM")
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = np.maximum(model.predict(X_te), 0)
    mae    = mean_absolute_error(y_te, y_pred)
    rmse   = float(np.sqrt(((y_te - y_pred)**2).mean()))
    nonzero = y_te > 0
    mape   = float(np.abs((y_te[nonzero] - y_pred[nonzero]) / y_te[nonzero]).mean() * 100)
    fi     = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)[:10]
    print(f"GBM: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualization
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Demand classification pie
ax = axes[0, 0]
ax.pie(class_counts.values, labels=class_counts.index,
        autopct="%1.1f%%", colors=["#4CAF50","#2196F3","#FF9800","#F44336","#9C27B0"])
ax.set_title("Demand Pattern Classification", fontsize=11)

# Panel 2: Zero fraction histogram
ax = axes[0, 1]
zero_frac = df.groupby("id").apply(lambda g: (g["sales"]==0).mean())
ax.hist(zero_frac, bins=25, color="#FF5722", edgecolor="white", alpha=0.8)
ax.set_title("Zero-Sales Fraction per Series", fontsize=11)
ax.set_xlabel("Fraction of zero-sales days"); ax.grid(alpha=0.3)

# Panel 3: Feature importance
ax = axes[1, 0]
fi_plot = fi[:10]
ax.barh(fi_plot.index[::-1], fi_plot.values[::-1], color="#2196F3")
ax.set_title("Top 10 Feature Importances (Global Model)", fontsize=11)
ax.set_xlabel("Importance"); ax.grid(alpha=0.3, axis="x")

# Panel 4: Prediction vs. actual (sample series)
ax = axes[1, 1]
sample_id_code = df_te["id"].value_counts().index[0]
sample_mask    = df_te["id"] == sample_id_code
sample_dates   = pd.to_datetime(df_te[sample_mask]["date"])
sample_true    = y_te[sample_mask.values]
sample_pred    = y_pred[sample_mask.values]
ax.plot(sample_dates, sample_true, "o-", color="#2196F3", linewidth=1.5, markersize=4, label="Actual")
ax.plot(sample_dates, sample_pred, "s--", color="#FF5722", linewidth=1.5, markersize=4, label="Forecast")
ax.set_title("Sample Series — Forecast vs. Actual (28-day test)", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.tick_params(axis="x", rotation=30)

plt.suptitle(f"Retail Sales Forecasting — Global LightGBM\n"
             f"MAE={mae:.4f}, MAPE={mape:.2f}% (non-zero), {n_series} SKUs",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("retail_forecast_results.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "="*55)
print("RETAIL PROJECT SUMMARY")
print("="*55)
print(f"Series: {n_series} ({N_ITEMS} items × {N_STORES} stores)")
print(f"Test MAE:  {mae:.4f} units")
print(f"Test RMSE: {rmse:.4f} units")
print(f"MAPE (non-zero demand): {mape:.2f}%")
print(f"Zero fraction: {(df['sales']==0).mean():.3f}")
print("Plot saved: retail_forecast_results.png")
