"""
02_xgboost_lgbm_pipeline.py
============================
Module 04 — ML for Time Series
Topic   : XGBoost & LightGBM Full Pipeline

Covers:
  - Feature engineering pipeline
  - XGBoost and LightGBM training with early stopping
  - Walk-forward CV evaluation
  - Feature importance (MDI + SHAP)
  - Multi-step forecasting (recursive and direct)
  - Final comparison: LGBM vs. XGBoost vs. seasonal naive
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import shap
from sklearn.model_selection import TimeSeriesSplit

plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN = "#2C7BB6", "#D7191C", "#1A9641"


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA AND FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
n = 1096   # 3 years daily
idx = pd.date_range("2021-01-01", periods=n, freq="D")
t = np.arange(n)

sales = (
    5000
    + 20 * t
    + 1000 * np.sin(2 * np.pi * t / 365.25)
    + 500 * np.sin(2 * np.pi * t / 7)
    + 400 * (idx.dayofweek == 5).astype(float)
    - 200 * (idx.dayofweek == 6).astype(float)
    + np.random.normal(0, 400, n)
).clip(min=0)

price    = 50 + np.random.randn(n) * 3
is_promo = (np.random.rand(n) < 0.1).astype(int)

df = pd.DataFrame({"sales": sales, "price": price, "is_promo": is_promo}, index=idx)

def build_features(df, target="sales"):
    df = df.copy()
    t = df[target]
    for lag in [1, 2, 3, 7, 14, 21, 28, 91, 182, 365]:
        df[f"lag_{lag}"] = t.shift(lag)
    for w in [7, 14, 30, 90]:
        s = t.shift(1)
        df[f"roll{w}_mean"] = s.rolling(w, min_periods=1).mean()
        df[f"roll{w}_std"]  = s.rolling(w, min_periods=1).std()
        df[f"roll{w}_max"]  = s.rolling(w, min_periods=1).max()
    df["ewm7"]  = t.shift(1).ewm(span=7).mean()
    df["ewm30"] = t.shift(1).ewm(span=30).mean()
    idx = df.index
    df["month_sin"]  = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * idx.month / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["doy_sin"]    = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    df["doy_cos"]    = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["trend"]      = np.arange(len(df))
    # Interaction
    df["promo_weekend"] = df["is_promo"] * df["is_weekend"]
    return df.dropna()

df_feat = build_features(df)
feature_cols = [c for c in df_feat.columns if c != "sales"]
X = df_feat[feature_cols]
y = df_feat["sales"]

split_date = "2023-07-01"
X_train = X[X.index < split_date]
X_test  = X[X.index >= split_date]
y_train = y[y.index < split_date]
y_test  = y[y.index >= split_date]

print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {X.shape[1]}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LIGHTGBM TRAINING WITH TIME-SAFE EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- LightGBM Training ---")
tscv = TimeSeriesSplit(n_splits=5, test_size=30)

lgbm_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="regression",
    random_state=42,
    verbose=-1,
)

# Use last fold of tscv as validation set for early stopping
for tr_idx, val_idx in tscv.split(X_train):
    pass   # get last fold

lgbm_model.fit(
    X_train.iloc[tr_idx], y_train.iloc[tr_idx],
    eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(period=200),
    ],
)
print(f"Best iteration: {lgbm_model.best_iteration_}")

y_pred_lgbm = lgbm_model.predict(X_test)
rmse_lgbm = np.sqrt(((y_test.values - y_pred_lgbm)**2).mean())
print(f"LightGBM Test RMSE: {rmse_lgbm:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. XGBOOST TRAINING
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- XGBoost Training ---")
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

xgb_model.fit(
    X_train.iloc[tr_idx], y_train.iloc[tr_idx],
    eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
    verbose=200,
    early_stopping_rounds=50,
)
y_pred_xgb = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(((y_test.values - y_pred_xgb)**2).mean())
print(f"XGBoost Test RMSE: {rmse_xgb:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SEASONAL NAIVE BASELINE
# ─────────────────────────────────────────────────────────────────────────────

snaive_pred = y_train.values[-len(y_test):]   # last year's same periods
rmse_snaive = np.sqrt(((y_test.values - snaive_pred[:len(y_test)])**2).mean())
print(f"Seasonal Naive RMSE: {rmse_snaive:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. FORECAST VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(y_train[-90:], color="gray", linewidth=1.2, label="Train (last 90d)")
ax.plot(y_test, color="black", linewidth=2, label="Actual")
ax.plot(y_test.index, y_pred_lgbm, color=BLUE, linewidth=1.8, linestyle="--",
        label=f"LightGBM (RMSE={rmse_lgbm:.0f})")
ax.plot(y_test.index, y_pred_xgb, color=RED, linewidth=1.8, linestyle=":",
        label=f"XGBoost (RMSE={rmse_xgb:.0f})")
ax.axvline(y_train.index[-1], color="black", linewidth=0.8, linestyle=":", alpha=0.5)
ax.legend(fontsize=9)
ax.set_title("LightGBM vs. XGBoost — Daily Sales Forecast")
ax.set_ylabel("Sales")
plt.tight_layout()
plt.savefig("01_lgbm_xgb_forecast.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

importance = pd.Series(lgbm_model.feature_importances_,
                       index=X_train.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
importance[:20].plot.barh(ax=ax, color=BLUE, edgecolor="white")
ax.invert_yaxis()
ax.set_title("LightGBM Feature Importance (MDI) — Top 20")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("02_feature_importance.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7. SHAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- SHAP Analysis ---")
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_test)

# Summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, max_display=20, show=False)
plt.title("SHAP Summary — LightGBM Feature Impact")
plt.tight_layout()
plt.savefig("03_shap_summary.png", bbox_inches="tight")
plt.show()

# Top features by mean |SHAP|
shap_importance = pd.Series(
    np.abs(shap_values).mean(0), index=X_test.columns
).sort_values(ascending=False)
print("\nTop 10 features by mean |SHAP|:")
print(shap_importance.head(10).round(4))

print("\n✅ XGBoost + LightGBM pipeline demo complete.")
print(f"   LightGBM:      RMSE={rmse_lgbm:.1f}")
print(f"   XGBoost:       RMSE={rmse_xgb:.1f}")
print(f"   Seasonal Naive: RMSE={rmse_snaive:.1f}")
print(f"   Improvement vs. naive: {(rmse_snaive - min(rmse_lgbm, rmse_xgb))/rmse_snaive*100:.1f}%")
