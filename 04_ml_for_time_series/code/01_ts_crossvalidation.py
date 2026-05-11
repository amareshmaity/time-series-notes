"""
01_ts_crossvalidation.py
========================
Module 04 — ML for Time Series
Topic   : Time Series Cross-Validation

Covers:
  - Walk-forward validation with rolling origin
  - Expanding window vs. sliding window CV
  - TimeSeriesSplit visualization
  - Effect of gap on CV scores
  - Comparing CV strategies
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE = "#2C7BB6", "#D7191C", "#1A9641", "#F07D00"


# ─────────────────────────────────────────────────────────────────────────────
# 1. CREATE SYNTHETIC DAILY DATASET
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
n = 730   # 2 years of daily data
idx = pd.date_range("2022-01-01", periods=n, freq="D")
t = np.arange(n)

sales = (
    3000
    + 15 * t
    + 600 * np.sin(2 * np.pi * t / 365.25)
    + 300 * np.sin(2 * np.pi * t / 7)
    + 200 * (idx.dayofweek == 5).astype(float)
    - 150 * (idx.dayofweek == 6).astype(float)
    + np.random.normal(0, 200, n)
).clip(min=0)

df = pd.DataFrame({"sales": sales}, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD FEATURES (leakage-free)
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df, target_col="sales"):
    df = df.copy()
    t_col = df[target_col]
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = t_col.shift(lag)
    for w in [7, 14, 30]:
        df[f"roll{w}_mean"] = t_col.shift(1).rolling(w, min_periods=1).mean()
        df[f"roll{w}_std"]  = t_col.shift(1).rolling(w, min_periods=1).std()
    idx = df.index
    df["month_sin"]  = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * idx.month / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["trend"]      = np.arange(len(df))
    return df.dropna()

df_feat = build_features(df)
X = df_feat.drop(columns=["sales"])
y = df_feat["sales"]
print(f"Feature matrix: {X.shape} | Target: {len(y)} rows")


# ─────────────────────────────────────────────────────────────────────────────
# 3. VISUALIZE TIMESERIESSPLIT STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def plot_cv_splits(X, cv_splits, title, ax):
    """Visualize CV splits on an axis."""
    all_train, all_test = [], []
    for i, (train_idx, test_idx) in enumerate(cv_splits):
        ax.scatter(train_idx, [i] * len(train_idx), c=BLUE,
                   marker="_", linewidth=3, alpha=0.5, s=100)
        ax.scatter(test_idx, [i] * len(test_idx), c=RED,
                   marker="_", linewidth=3, alpha=0.9, s=100)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Fold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Train"), Patch(color=RED, label="Test")],
              fontsize=9, loc="lower right")

fig, axes = plt.subplots(3, 1, figsize=(13, 9))

# Expanding window
cv_expanding = TimeSeriesSplit(n_splits=5, test_size=30)
plot_cv_splits(X, cv_expanding.split(X), "Expanding Window CV (test_size=30)", axes[0])

# Sliding window (max_train_size)
cv_sliding = TimeSeriesSplit(n_splits=5, test_size=30, max_train_size=200)
plot_cv_splits(X, cv_sliding.split(X), "Sliding Window CV (train_window=200, test_size=30)", axes[1])

# With gap
cv_gap = TimeSeriesSplit(n_splits=5, test_size=30, gap=14)
plot_cv_splits(X, cv_gap.split(X), "Expanding CV with Gap=14 (avoids short-term leakage)", axes[2])

plt.suptitle("Time Series Cross-Validation Strategies", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("01_cv_strategies.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 4. COMPARE CV SCORE DISTRIBUTIONS ACROSS STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Comparing CV strategies with LightGBM ---")

def run_cv(X, y, cv, label):
    rmses = []
    for train_idx, val_idx in cv.split(X):
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                               num_leaves=31, min_child_samples=10,
                               subsample=0.8, colsample_bytree=0.8, verbose=-1)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = m.predict(X.iloc[val_idx])
        rmse = np.sqrt(((y.iloc[val_idx].values - pred)**2).mean())
        rmses.append(rmse)
    print(f"  {label:<45} RMSE: {np.mean(rmses):.2f} ± {np.std(rmses):.2f}")
    return rmses

strategies = {
    "Expanding Window (test=30)":          TimeSeriesSplit(n_splits=5, test_size=30),
    "Sliding Window (train=200, test=30)": TimeSeriesSplit(n_splits=5, test_size=30, max_train_size=200),
    "Expanding + Gap=14":                  TimeSeriesSplit(n_splits=5, test_size=30, gap=14),
    "Expanding (test=60)":                 TimeSeriesSplit(n_splits=5, test_size=60),
}

all_scores = {}
for label, cv in strategies.items():
    all_scores[label] = run_cv(X, y, cv, label)

# Boxplot comparison
fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot(list(all_scores.values()), labels=[s[:30] for s in all_scores.keys()])
ax.set_title("CV RMSE Distribution by Strategy")
ax.set_ylabel("RMSE")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("02_cv_comparison.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5. DEMONSTRATE RANDOM KFOLD IS WRONG
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Demonstrating why K-Fold is wrong ---")

from sklearn.model_selection import KFold

kfold_rmses = []
for train_idx, val_idx in KFold(n_splits=5, shuffle=True, random_state=42).split(X):
    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, verbose=-1)
    m.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = m.predict(X.iloc[val_idx])
    kfold_rmses.append(np.sqrt(((y.iloc[val_idx].values - pred)**2).mean()))

ts_rmses = run_cv(X, y, TimeSeriesSplit(n_splits=5, test_size=30), "TimeSeriesSplit (correct)")

print(f"\n  K-Fold (WRONG for TS):  RMSE={np.mean(kfold_rmses):.2f} ← OPTIMISTICALLY BIASED")
print(f"  TimeSeriesSplit (right): RMSE={np.mean(ts_rmses):.2f} ← TRUE expected error")
print(f"\n  K-Fold underestimates error by: {(np.mean(ts_rmses) - np.mean(kfold_rmses)) / np.mean(ts_rmses) * 100:.1f}%")
print("  This 'free lunch' disappears when the model is deployed on real future data!")


# ─────────────────────────────────────────────────────────────────────────────
# 6. WALK-FORWARD VALIDATION WITH FULL REFIT
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Walk-forward validation (rolling origin) ---")

n_folds  = 6
horizon  = 30
results  = []

for fold in range(n_folds):
    train_end   = len(df_feat) - (n_folds - fold) * horizon
    test_start  = train_end
    test_end    = test_start + horizon
    if test_end > len(df_feat):
        break

    X_tr = X.iloc[:train_end]
    y_tr = y.iloc[:train_end]
    X_te = X.iloc[test_start:test_end]
    y_te = y.iloc[test_start:test_end]

    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                           num_leaves=31, min_child_samples=10, verbose=-1)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    rmse = np.sqrt(((y_te.values - pred)**2).mean())
    results.append({"fold": fold+1, "train_size": train_end,
                    "origin": X_tr.index[-1].date(), "RMSE": rmse})

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
print(f"\nMean RMSE: {df_results['RMSE'].mean():.2f} ± {df_results['RMSE'].std():.2f}")

# Plot RMSE by origin (check for drift — increasing error = model degrading)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_results["fold"], df_results["RMSE"], marker="o", color=RED, linewidth=2)
ax.fill_between(df_results["fold"],
                df_results["RMSE"] - df_results["RMSE"].std(),
                df_results["RMSE"] + df_results["RMSE"].std(),
                alpha=0.2, color=RED)
ax.set_xlabel("Fold (origin)")
ax.set_ylabel("RMSE")
ax.set_title("Walk-Forward Validation: RMSE by Origin\n(stable = model generalizes well; increasing = model drift)")
plt.tight_layout()
plt.savefig("03_walkforward_rmse.png", bbox_inches="tight")
plt.show()

print("\n✅ Time Series CV demo complete.")
print("   Key takeaway: K-Fold on TS is wrong — always use TimeSeriesSplit or walk-forward validation.")
