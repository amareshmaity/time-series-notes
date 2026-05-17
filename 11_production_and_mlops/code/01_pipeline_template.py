"""
code/01_pipeline_template.py
==============================
Module 11 — Production & MLOps
Practical: Modular sklearn-compatible TS pipeline.

Demonstrates:
  - TSFeatureTransformer (strictly backward-looking features)
  - Walk-forward cross-validation with gap (no leakage)
  - Full pipeline: transformer → scaler → model
  - Fold-level metrics + visualization
  - Pipeline serialization and reload
  - Anti-pattern comparison (with-leakage vs. without-leakage)
"""

import numpy as np
import pandas as pd
import pickle, cloudpickle
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature Transformer (Point-in-Time Correct)
# ─────────────────────────────────────────────────────────────────────────────

class TSFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible feature transformer for time series.
    All features strictly backward-looking (no leakage).
    """

    def __init__(
        self,
        lags: list = None,
        rolling_windows: list = None,
        include_calendar: bool = True,
    ):
        self.lags             = lags or [1, 2, 3, 7, 14]
        self.rolling_windows  = rolling_windows or [7, 30]
        self.include_calendar = include_calendar

    def fit(self, X, y=None):
        return self   # stateless transformer

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """X must have columns: timestamp, value"""
        df = X.copy()
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        s    = df["value"].astype(float)
        feat = pd.DataFrame(index=df.index)
        feat["value"] = s.values

        # Lag features (strictly past)
        for lag in self.lags:
            feat[f"lag_{lag}"] = s.shift(lag).values

        # Difference features
        feat["diff_1"] = s.diff(1).values
        feat["diff_7"] = s.diff(7).values if len(s) > 7 else np.nan

        # Rolling features (shift(1) → exclude current value)
        for w in self.rolling_windows:
            rol = s.shift(1).rolling(w, min_periods=max(2, w//4))
            feat[f"rmean_{w}"]  = rol.mean().values
            feat[f"rstd_{w}"]   = rol.std().values
            feat[f"rmin_{w}"]   = rol.min().values
            feat[f"rmax_{w}"]   = rol.max().values

        # Calendar features (available at prediction time)
        if self.include_calendar and hasattr(df.index, "dayofweek"):
            feat["day_of_week"] = df.index.dayofweek.astype(float)
            feat["month"]       = df.index.month.astype(float)
            feat["is_weekend"]  = (df.index.dayofweek >= 5).astype(float)
            if hasattr(df.index, "quarter"):
                feat["quarter"] = df.index.quarter.astype(float)

        return feat.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Synthetic Time Series Dataset
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
N = 1000
dates = pd.date_range("2021-01-01", periods=N, freq="D")

trend    = np.linspace(0, 50, N)
seasonal = 20 * np.sin(2 * np.pi * np.arange(N) / 365.25)
weekly   = 10 * np.sin(2 * np.pi * np.arange(N) / 7)
noise    = np.random.normal(0, 5, N)
values   = 100 + trend + seasonal + weekly + noise

raw_df = pd.DataFrame({"timestamp": dates, "value": values})
print(f"Synthetic dataset: {N} daily observations from {dates[0].date()} to {dates[-1].date()}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Walk-Forward Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_cv(pipeline, df, n_splits=5, test_size=60, gap=1):
    """
    Walk-forward cross-validation with gap between train and test.

    gap = simulates prediction latency (e.g., we don't get today's data
    until tomorrow, so we predict with yesterday's features).
    """
    n = len(df)
    step = (n - test_size * n_splits) // (n_splits + 1)
    fold_results = []

    for i in range(n_splits):
        train_end   = step + i * step + step
        test_start  = train_end + gap
        test_end    = test_start + test_size

        if test_end > n:
            break

        train = df.iloc[:train_end]
        test  = df.iloc[test_start:test_end]

        # Transform
        transformer = TSFeatureTransformer()
        feat_train  = transformer.fit_transform(train)
        feat_test   = transformer.transform(test)

        # Align (both have same columns after transform)
        common_cols = list(set(feat_train.columns) & set(feat_test.columns))

        X_tr = feat_train[common_cols].drop(columns=["value"]).values
        y_tr = feat_train["value"].values
        X_te = feat_test[common_cols].drop(columns=["value"]).values
        y_te = feat_test["value"].values

        # Clone and fit pipeline
        from sklearn.base import clone
        pipe = clone(pipeline)
        pipe.fit(X_tr, y_tr)
        y_hat = pipe.predict(X_te)

        mae  = mean_absolute_error(y_te, y_hat)
        rmse = np.sqrt(mean_squared_error(y_te, y_hat))
        mape = np.mean(np.abs((y_te - y_hat) / (np.abs(y_te) + 1e-8))) * 100

        fold_results.append({
            "fold":       i + 1,
            "train_rows": len(train),
            "test_rows":  len(test),
            "gap":        gap,
            "mae":        round(mae, 4),
            "rmse":       round(rmse, 4),
            "mape":       round(mape, 4),
            "y_true":     y_te,
            "y_pred":     y_hat,
            "dates":      feat_test.index,
        })
        print(f"  Fold {i+1}: train={len(train)}, test={len(test)} | "
              f"MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%")

    return fold_results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pipeline Definitions
# ─────────────────────────────────────────────────────────────────────────────

# Pipeline 1: Ridge regression
ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  Ridge(alpha=100)),
])

# Pipeline 2: Gradient Boosting
gbm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                          learning_rate=0.05, random_state=42)),
])

print("\nWalk-Forward CV — Ridge Regression:")
ridge_results = walk_forward_cv(ridge_pipeline, raw_df, n_splits=5, test_size=60, gap=1)

print("\nWalk-Forward CV — Gradient Boosting:")
gbm_results = walk_forward_cv(gbm_pipeline, raw_df, n_splits=5, test_size=60, gap=1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Anti-Pattern: Leakage Demonstration
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("ANTI-PATTERN: Leakage Demonstration")
print("="*55)

# Fit scaler on FULL dataset (leaks future stats)
transformer_leaky = TSFeatureTransformer()
feat_all          = transformer_leaky.fit_transform(raw_df)
X_all = feat_all.drop(columns=["value"]).values
y_all = feat_all["value"].values

scaler_leaky = StandardScaler().fit(X_all)   # ← WRONG: fitted on all data
X_scaled_leaky = scaler_leaky.transform(X_all)

# Split after scaling (leaky)
n_train  = int(0.8 * len(X_all))
X_tr_l, y_tr_l = X_scaled_leaky[:n_train], y_all[:n_train]
X_te_l, y_te_l = X_scaled_leaky[n_train:], y_all[n_train:]

model_leaky = Ridge(alpha=100)
model_leaky.fit(X_tr_l, y_tr_l)
y_hat_leaky = model_leaky.predict(X_te_l)
mae_leaky   = mean_absolute_error(y_te_l, y_hat_leaky)

# Correct: fit scaler on train only
X_tr_c = X_all[:n_train]; y_tr_c = y_all[:n_train]
X_te_c = X_all[n_train:]; y_te_c = y_all[n_train:]

scaler_correct = StandardScaler().fit(X_tr_c)  # ← CORRECT: train only
model_correct  = Ridge(alpha=100)
model_correct.fit(scaler_correct.transform(X_tr_c), y_tr_c)
y_hat_correct  = model_correct.predict(scaler_correct.transform(X_te_c))
mae_correct    = mean_absolute_error(y_te_c, y_hat_correct)

print(f"  Leaky   test MAE: {mae_leaky:.4f}  ← overly optimistic!")
print(f"  Correct test MAE: {mae_correct:.4f}  ← realistic estimate")
print(f"  Leakage bias:     {mae_correct - mae_leaky:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pipeline Serialization
# ─────────────────────────────────────────────────────────────────────────────

print("\nSerializing full pipeline...")
# Fit final model on full training data
transformer_final = TSFeatureTransformer()
feat_final        = transformer_final.fit_transform(raw_df)
X_final           = feat_final.drop(columns=["value"]).values
y_final           = feat_final["value"].values

gbm_pipeline.fit(X_final, y_final)

artifact = {
    "transformer": transformer_final,
    "model":       gbm_pipeline,
    "feature_cols": [c for c in feat_final.columns if c != "value"],
    "trained_on":  raw_df["timestamp"].max().isoformat(),
}
with open("ts_model_artifact.pkl", "wb") as f:
    cloudpickle.dump(artifact, f)
print("  Saved: ts_model_artifact.pkl")

# Reload and verify
with open("ts_model_artifact.pkl", "rb") as f:
    loaded = cloudpickle.load(f)

test_input = raw_df.iloc[-60:].copy()
feat_test  = loaded["transformer"].transform(test_input)
X_test_  = feat_test.drop(columns=["value"]).values
y_test_  = feat_test["value"].values
y_reload = loaded["model"].predict(X_test_)
print(f"  Reload verification MAE: {mean_absolute_error(y_test_, y_reload):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualization
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Panel 1: Raw time series
ax = axes[0, 0]
ax.plot(dates, values, color="#2196F3", linewidth=0.8, alpha=0.8)
ax.set_title("Synthetic Time Series", fontsize=11)
ax.set_xlabel("Date"); ax.set_ylabel("Value"); ax.grid(alpha=0.3)

# Panel 2: Walk-forward CV results (per-fold MAE)
ax = axes[0, 1]
folds_r = [r["fold"] for r in ridge_results]
maes_r  = [r["mae"]  for r in ridge_results]
folds_g = [r["fold"] for r in gbm_results]
maes_g  = [r["mae"]  for r in gbm_results]
x_pos   = np.arange(len(folds_r))
w       = 0.35
ax.bar(x_pos - w/2, maes_r, w, label="Ridge",    color="#2196F3")
ax.bar(x_pos + w/2, maes_g, w, label="GBM",      color="#4CAF50")
ax.set_xticks(x_pos); ax.set_xticklabels([f"Fold {f}" for f in folds_r])
ax.set_ylabel("Test MAE"); ax.set_title("Walk-Forward CV — Per-Fold MAE", fontsize=11)
ax.legend(); ax.grid(alpha=0.3, axis="y")

# Panel 3: Fold 1 prediction plot
ax = axes[1, 0]
last_fold = gbm_results[-1]
ax.plot(last_fold["dates"], last_fold["y_true"], label="Actual", linewidth=1.5, color="#2196F3")
ax.plot(last_fold["dates"], last_fold["y_pred"], label="Forecast", linewidth=1.5,
        color="#FF5722", linestyle="--")
ax.set_title(f"Last Fold — GBM Forecast vs. Actual", fontsize=11)
ax.legend(); ax.grid(alpha=0.3); ax.tick_params(axis="x", rotation=30)

# Panel 4: Leakage comparison
ax = axes[1, 1]
methods  = ["Leaky pipeline\n(scaler fit on full data)", "Correct pipeline\n(scaler fit on train only)"]
maes_cmp = [mae_leaky, mae_correct]
colors   = ["#F44336", "#4CAF50"]
bars     = ax.bar(methods, maes_cmp, color=colors, width=0.5)
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=11)
ax.set_ylabel("Test MAE"); ax.set_title("Leakage vs. Correct Pipeline", fontsize=11)
ax.grid(alpha=0.3, axis="y")
ax.text(0.5, max(maes_cmp) * 0.5, f"Bias: {mae_correct-mae_leaky:.4f}",
        ha="center", fontsize=11, color="gray")

plt.suptitle("Production TS Pipeline — Walk-Forward CV + Leakage Demo",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("pipeline_walkforward_cv.png", dpi=150, bbox_inches="tight")
plt.show()

# Summary table
print("\n" + "="*55)
print("CV SUMMARY")
print("="*55)
for name, results in [("Ridge", ridge_results), ("GBM", gbm_results)]:
    maes  = [r["mae"]  for r in results]
    rmses = [r["rmse"] for r in results]
    print(f"\n{name}:")
    print(f"  MAE:  {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"  RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
