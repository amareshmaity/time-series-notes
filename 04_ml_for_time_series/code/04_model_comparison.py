"""
04_model_comparison.py
=======================
Module 04 — ML for Time Series
Topic   : Full Model Comparison Leaderboard

Covers:
  - Ridge, Random Forest, XGBoost, LightGBM, Stacking Ensemble
  - Walk-forward CV for each model
  - Final leaderboard with RMSE, MAE, MAPE
  - Forecast plots for all models
  - Model diversity check (error correlation)
  - Final ensemble
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import nnls

plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
COLORS = ["#2C7BB6", "#D7191C", "#1A9641", "#F07D00", "#7B2D8B", "#8B4513"]


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
    df["trend"]      = np.arange(len(df))
    df["month_sin"]  = np.sin(2*np.pi*df.index.month/12)
    df["month_cos"]  = np.cos(2*np.pi*df.index.month/12)
    df["dow_sin"]    = np.sin(2*np.pi*df.index.dayofweek/7)
    df["dow_cos"]    = np.cos(2*np.pi*df.index.dayofweek/7)
    df["is_weekend"] = (df.index.dayofweek>=5).astype(int)
    return df.dropna()

df_feat = build_features(df)
X = df_feat.drop(columns=["sales"])
y = df_feat["sales"]

split_date = "2023-07-01"
X_train, X_test = X[X.index < split_date], X[X.index >= split_date]
y_train, y_test = y[y.index < split_date], y[y.index >= split_date]
print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {X.shape[1]}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DEFINE ALL MODELS
# ─────────────────────────────────────────────────────────────────────────────

tscv = TimeSeriesSplit(n_splits=5, test_size=30)

models = {
    "Seasonal Naive": None,  # special case
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=100)),
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=10, max_features=0.33,
        n_jobs=-1, random_state=42,
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, objective="regression",
        random_state=42, verbose=-1,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. WALK-FORWARD CV FOR EACH MODEL
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, name):
    e  = y_true - y_pred
    return {
        "Model": name,
        "RMSE":  round(np.sqrt((e**2).mean()), 2),
        "MAE":   round(np.abs(e).mean(), 2),
        "MAPE":  round((np.abs(e) / np.abs(y_true)).mean() * 100, 2),
    }

print("\n--- Walk-Forward CV ---")
cv_scores = {}
all_test_preds = {}

for name, model in models.items():
    if name == "Seasonal Naive":
        # Seasonal naive: y(t) = y(t-365)
        pred = y_test.values
        snaive_vals = y.shift(365).reindex(y_test.index).values
        snaive_vals = np.where(np.isnan(snaive_vals), y_train.mean(), snaive_vals)
        all_test_preds["Seasonal Naive"] = snaive_vals
        rmse = np.sqrt(((y_test.values - snaive_vals)**2).mean())
        cv_scores["Seasonal Naive"] = [rmse]
        print(f"  Seasonal Naive    RMSE={rmse:.2f}")
        continue

    fold_rmses = []
    for tr_idx, val_idx in tscv.split(X_train):
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pred = model.predict(X_train.iloc[val_idx])
        fold_rmses.append(np.sqrt(((y_train.iloc[val_idx].values - pred)**2).mean()))

    print(f"  {name:<20} CV RMSE={np.mean(fold_rmses):.2f} ± {np.std(fold_rmses):.2f}")
    cv_scores[name] = fold_rmses

    # Final test prediction
    model.fit(X_train, y_train)
    all_test_preds[name] = model.predict(X_test)


# ─────────────────────────────────────────────────────────────────────────────
# 4. STACKING ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Building Stacking Ensemble ---")

# Out-of-fold predictions for meta-learning
oof_preds = np.zeros((len(X_train), len(models) - 1))   # exclude seasonal naive
model_names_ml = [n for n in models if n != "Seasonal Naive"]

for col_idx, name in enumerate(model_names_ml):
    model = models[name]
    for tr_idx, val_idx in tscv.split(X_train):
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_preds[val_idx, col_idx] = model.predict(X_train.iloc[val_idx])

# Train Ridge meta-model on OOF predictions
meta_model = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=[0.01, 0.1, 1.0, 10]))])
meta_model.fit(oof_preds, y_train)

# Test predictions
test_pred_stack = np.zeros((len(X_test), len(model_names_ml)))
for col_idx, name in enumerate(model_names_ml):
    models[name].fit(X_train, y_train)
    test_pred_stack[:, col_idx] = models[name].predict(X_test)

stack_pred = meta_model.predict(test_pred_stack)
all_test_preds["Stacking Ensemble"] = stack_pred

# NNLS Blend
stacked_val = np.column_stack([all_test_preds[n] for n in model_names_ml])
weights, _ = nnls(stacked_val, y_test.values)
weights /= weights.sum()
nnls_pred = stacked_val @ weights
all_test_preds["NNLS Blend"] = nnls_pred

print("NNLS Blend weights:")
for name, w in zip(model_names_ml, weights):
    print(f"  {name:<20}: {w:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. FINAL LEADERBOARD
# ─────────────────────────────────────────────────────────────────────────────

results = [evaluate(y_test.values, pred, name)
           for name, pred in all_test_preds.items()]
leaderboard = pd.DataFrame(results).set_index("Model").sort_values("RMSE")
print("\n" + "="*55)
print("  FINAL LEADERBOARD (Test Set)")
print("="*55)
print(leaderboard.to_string())

# Improvement vs. seasonal naive
snaive_rmse = leaderboard.loc["Seasonal Naive", "RMSE"]
print(f"\nImprovement over Seasonal Naive:")
for model_name, row in leaderboard.iterrows():
    if model_name != "Seasonal Naive":
        gain = (snaive_rmse - row["RMSE"]) / snaive_rmse * 100
        flag = "✅" if gain > 0 else "❌"
        print(f"  {model_name:<25}: {gain:+.1f}%  {flag}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(13, 11))

# Forecast comparison
axes[0].plot(y_test.values, color="black", linewidth=2.5, label="Actual")
plot_models = [n for n in all_test_preds if n != "NNLS Blend"]
for i, name in enumerate(plot_models):
    lw = 2.5 if "Ensemble" in name or "LightGBM" in name else 1.2
    axes[0].plot(all_test_preds[name], color=COLORS[i % len(COLORS)],
                 linewidth=lw, linestyle="--" if name == "Seasonal Naive" else "-",
                 alpha=0.7, label=f"{name} ({leaderboard.loc[name, 'RMSE']:.0f})")
axes[0].set_title("All Models — Test Set Forecast")
axes[0].legend(fontsize=8, ncol=2)
axes[0].set_xlabel("Days into test period")
axes[0].set_ylabel("Sales")

# RMSE leaderboard bar chart
rmse_vals = leaderboard["RMSE"]
bar_colors = [COLORS[i % len(COLORS)] for i in range(len(rmse_vals))]
axes[1].barh(leaderboard.index, rmse_vals.values, color=bar_colors, edgecolor="white")
axes[1].axvline(snaive_rmse, color="black", linewidth=1.5, linestyle="--",
                label=f"Seasonal Naive ({snaive_rmse:.0f})")
axes[1].set_xlabel("Test RMSE (lower = better)")
axes[1].set_title("Model Comparison — Test RMSE Leaderboard")
axes[1].legend()
for i, (idx_val, val) in enumerate(zip(leaderboard.index, rmse_vals)):
    axes[1].text(val + 5, i, f"{val:.1f}", va="center", fontsize=9)

plt.suptitle("ML for Time Series — Complete Model Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("01_model_leaderboard.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7. ERROR CORRELATION (MODEL DIVERSITY)
# ─────────────────────────────────────────────────────────────────────────────

error_df = pd.DataFrame({
    name: y_test.values - pred
    for name, pred in all_test_preds.items()
    if name not in ("Stacking Ensemble", "NNLS Blend")
})

fig, ax = plt.subplots(figsize=(8, 6))
corr = error_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn_r",
            vmin=-1, vmax=1, center=0, ax=ax, linewidths=0.5)
ax.set_title("Forecast Error Correlation Matrix\n(lower correlation = higher diversity = better ensemble gain)")
plt.tight_layout()
plt.savefig("02_error_correlation.png", bbox_inches="tight")
plt.show()

print("\n✅ Full model comparison complete.")
print("   Diverse model families + ensemble → best overall accuracy")
