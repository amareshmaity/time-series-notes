"""
03_hyperparameter_tuning.py
============================
Module 04 — ML for Time Series
Topic   : Hyperparameter Tuning for TS ML Models

Covers:
  - Optuna-based Bayesian optimization for LightGBM
  - Time-safe objective function (walk-forward CV inside each trial)
  - Visualization: optimization history, parameter importance, parallel plot
  - Before/after tuning comparison
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)
from sklearn.model_selection import TimeSeriesSplit

plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED = "#2C7BB6", "#D7191C"
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA + FEATURES
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
n = 1096
idx = pd.date_range("2021-01-01", periods=n, freq="D")
t = np.arange(n)

sales = (
    5000 + 20*t
    + 1000*np.sin(2*np.pi*t/365.25)
    + 500*np.sin(2*np.pi*t/7)
    + 400*(idx.dayofweek==5).astype(float)
    - 200*(idx.dayofweek==6).astype(float)
    + np.random.normal(0, 400, n)
).clip(min=0)

df = pd.DataFrame({"sales": sales}, index=idx)

def build_features(df, target="sales"):
    df = df.copy()
    s = df[target]
    for lag in [1, 7, 14, 28, 91, 365]:
        df[f"lag_{lag}"] = s.shift(lag)
    for w in [7, 30, 90]:
        df[f"roll{w}_mean"] = s.shift(1).rolling(w, min_periods=1).mean()
        df[f"roll{w}_std"]  = s.shift(1).rolling(w, min_periods=1).std()
    df["trend"]     = np.arange(len(df))
    df["month_sin"] = np.sin(2*np.pi*df.index.month/12)
    df["month_cos"] = np.cos(2*np.pi*df.index.month/12)
    df["dow_sin"]   = np.sin(2*np.pi*df.index.dayofweek/7)
    df["dow_cos"]   = np.cos(2*np.pi*df.index.dayofweek/7)
    df["is_weekend"]= (df.index.dayofweek>=5).astype(int)
    return df.dropna()

df_feat = build_features(df)
feature_cols = [c for c in df_feat.columns if c != "sales"]
X = df_feat[feature_cols]
y = df_feat["sales"]

split_date = "2023-07-01"
X_train, X_test = X[X.index < split_date], X[X.index >= split_date]
y_train, y_test = y[y.index < split_date], y[y.index >= split_date]
print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {X.shape[1]}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DEFAULT MODEL BASELINE
# ─────────────────────────────────────────────────────────────────────────────

default_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.1,
                                   num_leaves=31, verbose=-1, random_state=42)
default_model.fit(X_train, y_train)
default_rmse = np.sqrt(((y_test - default_model.predict(X_test))**2).mean())
print(f"\nDefault LightGBM RMSE: {default_rmse:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. OPTUNA OBJECTIVE — TIME-SAFE CV INSIDE EACH TRIAL
# ─────────────────────────────────────────────────────────────────────────────

tscv = TimeSeriesSplit(n_splits=5, test_size=30)

def objective(trial: optuna.Trial) -> float:
    """Time-safe CV objective for Optuna."""
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 10, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "subsample":         trial.suggest_float("subsample", 0.4, 1.0),
        "subsample_freq":    1,
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "objective":         "regression",
        "verbose":           -1,
        "random_state":      42,
    }

    fold_rmses = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        pred  = model.predict(X_val)
        rmse  = np.sqrt(((y_val.values - pred)**2).mean())
        fold_rmses.append(rmse)

    return float(np.mean(fold_rmses))


# ─────────────────────────────────────────────────────────────────────────────
# 4. RUN OPTUNA STUDY
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Running Optuna (50 trials) ---")
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=50, timeout=300, show_progress_bar=True)

print(f"\nBest trial: #{study.best_trial.number}")
print(f"Best CV RMSE: {study.best_value:.4f}")
print("Best hyperparameters:")
for k, v in study.best_params.items():
    print(f"  {k:<25}: {v}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN TUNED MODEL
# ─────────────────────────────────────────────────────────────────────────────

best_params = {**study.best_params, "objective": "regression", "verbose": -1, "random_state": 42}
tuned_model = lgb.LGBMRegressor(**best_params)
tuned_model.fit(X_train, y_train)

tuned_rmse = np.sqrt(((y_test - tuned_model.predict(X_test))**2).mean())
print(f"\nDefault RMSE: {default_rmse:.2f}")
print(f"Tuned RMSE:   {tuned_rmse:.2f}")
print(f"Improvement:  {(default_rmse - tuned_rmse)/default_rmse*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 6. OPTIMIZATION HISTORY PLOT
# ─────────────────────────────────────────────────────────────────────────────

trial_values = [t.value for t in study.trials if t.value is not None]
best_so_far  = np.minimum.accumulate(trial_values)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(range(1, len(trial_values)+1), trial_values,
             color="lightgray", alpha=0.6, marker="o", markersize=3, label="Trial RMSE")
axes[0].plot(range(1, len(best_so_far)+1), best_so_far,
             color=RED, linewidth=2, label="Best so far")
axes[0].axhline(default_rmse, color=BLUE, linewidth=1.5, linestyle="--", label="Default RMSE")
axes[0].set_xlabel("Trial number")
axes[0].set_ylabel("CV RMSE")
axes[0].set_title("Optimization History")
axes[0].legend(fontsize=9)

# Before/After forecast
pred_default = default_model.predict(X_test)
pred_tuned   = tuned_model.predict(X_test)

axes[1].plot(y_test.values, color="black", linewidth=2, label="Actual")
axes[1].plot(pred_default, color=BLUE, linewidth=1.5, linestyle="--",
             label=f"Default (RMSE={default_rmse:.0f})")
axes[1].plot(pred_tuned,   color=RED,  linewidth=1.5, linestyle=":",
             label=f"Tuned (RMSE={tuned_rmse:.0f})")
axes[1].set_title("Forecast: Default vs. Tuned LightGBM")
axes[1].legend(fontsize=9)

plt.suptitle("Optuna Hyperparameter Tuning Results", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("01_optuna_results.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7. PARAMETER IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

param_imp = optuna.importance.get_param_importances(study)
print("\nParameter Importance:")
for param, imp in sorted(param_imp.items(), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {param:<25} {imp:.4f}  {bar}")

fig, ax = plt.subplots(figsize=(8, 5))
params = list(param_imp.keys())
values = list(param_imp.values())
ax.barh(params, values, color=BLUE, edgecolor="white")
ax.set_title("Optuna Hyperparameter Importance")
ax.set_xlabel("Importance (FAnova)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("02_param_importance.png", bbox_inches="tight")
plt.show()

print("\n✅ Hyperparameter tuning demo complete.")
print(f"   Optuna improved RMSE by {(default_rmse - tuned_rmse)/default_rmse*100:.1f}% with {len(study.trials)} trials")
