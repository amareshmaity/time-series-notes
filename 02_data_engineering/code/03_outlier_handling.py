"""
03_outlier_handling.py
======================
Module 02 — Data Engineering for Time Series
Topic   : Outlier Detection and Treatment

Covers:
  - Z-score and modified Z-score (MAD-based)
  - IQR method
  - Rolling Z-score (time-aware)
  - STL residual-based detection
  - Isolation Forest with temporal features
  - Treatment: remove+interpolate, Winsorize, replace with rolling median
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest
from scipy.stats import mstats

plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE = "#2C7BB6", "#D7191C", "#1A9641", "#F07D00"


# ─────────────────────────────────────────────────────────────────────────────
# 1. CREATE SYNTHETIC SERIES WITH KNOWN OUTLIERS
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
n = 365 * 3   # 3 years of daily data
idx = pd.date_range("2021-01-01", periods=n, freq="D")
t = np.arange(n)

# Clean underlying series
clean = (
    100
    + 0.05 * t
    + 15 * np.sin(2 * np.pi * t / 365.25)   # yearly seasonality
    + 8  * np.sin(2 * np.pi * t / 7)         # weekly seasonality
    + np.random.normal(0, 3, n)
)

# Inject known outliers
series = pd.Series(clean.copy(), index=idx, name="value")
outlier_positions = [50, 200, 400, 700, 900]
series.iloc[outlier_positions] += [80, -60, 100, -90, 70]   # spike outliers

# Inject a level shift (to show its different nature)
series.iloc[600:650] += 50   # transient jump

print(f"Series: {len(series)} observations | {series.index[0].date()} → {series.index[-1].date()}")
print(f"Known outlier positions: {outlier_positions}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DETECTION METHODS
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Running outlier detection ---")

# ── Method 1: Z-score ───────────────────────────────────────────────────────
def zscore_detection(s: pd.Series, threshold: float = 3.0) -> pd.Series:
    z = (s - s.mean()) / s.std()
    return z.abs() > threshold

mask_zscore = zscore_detection(series, threshold=3.0)
print(f"Z-score        : {mask_zscore.sum()} outliers")

# ── Method 2: Modified Z-score (MAD-based) ──────────────────────────────────
def modified_zscore_detection(s: pd.Series, threshold: float = 3.5) -> pd.Series:
    median = s.median()
    mad = (s - median).abs().median()
    mz = 0.6745 * (s - median) / (mad + 1e-8)
    return mz.abs() > threshold

mask_mad = modified_zscore_detection(series, threshold=3.5)
print(f"Modified Z-score (MAD): {mask_mad.sum()} outliers")

# ── Method 3: IQR ───────────────────────────────────────────────────────────
def iqr_detection(s: pd.Series, factor: float = 1.5) -> pd.Series:
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    return (s < Q1 - factor * IQR) | (s > Q3 + factor * IQR)

mask_iqr = iqr_detection(series, factor=1.5)
print(f"IQR (factor=1.5)      : {mask_iqr.sum()} outliers")

# ── Method 4: Rolling Z-score ───────────────────────────────────────────────
def rolling_zscore_detection(s: pd.Series, window: int = 30, threshold: float = 3.0) -> pd.Series:
    roll_mean = s.rolling(window=window, center=True, min_periods=window // 2).mean()
    roll_std  = s.rolling(window=window, center=True, min_periods=window // 2).std()
    z = (s - roll_mean) / (roll_std + 1e-8)
    return z.abs() > threshold

mask_rolling = rolling_zscore_detection(series, window=30, threshold=3.0)
print(f"Rolling Z-score (w=30): {mask_rolling.sum()} outliers")

# ── Method 5: STL Residual ──────────────────────────────────────────────────
def stl_detection(s: pd.Series, period: int = 365, threshold: float = 3.0) -> pd.Series:
    stl = STL(s, period=period, robust=True)
    result = stl.fit()
    resid = result.resid
    median_r = resid.median()
    mad = (resid - median_r).abs().median()
    mz = 0.6745 * (resid - median_r) / (mad + 1e-8)
    return mz.abs() > threshold

mask_stl = stl_detection(series, period=365, threshold=3.0)
print(f"STL residual          : {mask_stl.sum()} outliers")

# ── Method 6: Isolation Forest ──────────────────────────────────────────────
def isolation_forest_detection(s: pd.Series, contamination: float = 0.01, window: int = 7) -> pd.Series:
    df_feat = pd.DataFrame({"value": s})
    for lag in range(1, window + 1):
        df_feat[f"lag_{lag}"] = s.shift(lag)
    df_feat["roll_mean"] = s.shift(1).rolling(window).mean()
    df_feat["roll_std"]  = s.shift(1).rolling(window).std()
    df_feat = df_feat.dropna()

    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    preds = clf.fit_predict(df_feat)
    mask = pd.Series(preds == -1, index=df_feat.index)
    return mask.reindex(s.index, fill_value=False)

mask_if = isolation_forest_detection(series, contamination=0.01)
print(f"Isolation Forest      : {mask_if.sum()} outliers")


# ─────────────────────────────────────────────────────────────────────────────
# 3. DETECTION COMPARISON PLOT
# ─────────────────────────────────────────────────────────────────────────────

masks = {
    "Z-score":       mask_zscore,
    "Modified Z (MAD)": mask_mad,
    "IQR":           mask_iqr,
    "Rolling Z-score": mask_rolling,
    "STL Residual":  mask_stl,
    "Isolation Forest": mask_if,
}

fig, axes = plt.subplots(len(masks), 1, figsize=(14, 14), sharex=True)
for ax, (name, mask) in zip(axes, masks.items()):
    ax.plot(series, color="gray", linewidth=0.8, alpha=0.6)
    ax.scatter(series.index[mask], series[mask], color=RED, s=30, zorder=5, label=f"Outliers ({mask.sum()})")
    for pos in outlier_positions:
        ax.axvline(series.index[pos], color="blue", linewidth=1, linestyle="--", alpha=0.4)
    ax.set_title(f"{name} ({mask.sum()} flagged)")
    ax.legend(fontsize=8)

plt.suptitle("Outlier Detection Method Comparison\n(blue dashed = known outlier positions)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("01_detection_comparison.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 4. TREATMENT STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

# Use STL-based mask (best method for seasonal data)
outlier_mask = mask_stl

# Treatment 1: Remove and linear interpolate
series_interp = series.copy()
series_interp[outlier_mask] = np.nan
series_interp = series_interp.interpolate(method="linear")

# Treatment 2: Winsorize (clip to 1st–99th percentile)
lower = series.quantile(0.01)
upper = series.quantile(0.99)
series_winsorized = series.clip(lower=lower, upper=upper)

# Treatment 3: Replace with rolling median
rolling_med = series.rolling(window=7, center=True, min_periods=1).median()
series_roll_med = series.copy()
series_roll_med[outlier_mask] = rolling_med[outlier_mask]

# Treatment 4: Keep + flag (don't modify the value)
df_flagged = pd.DataFrame({
    "value":      series,
    "is_outlier": outlier_mask.astype(int),
})

# Plot treatments
fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

axes[0].plot(series, color=BLUE, linewidth=0.8, label="Original")
axes[0].scatter(series.index[outlier_mask], series[outlier_mask],
                color=RED, s=40, zorder=5, label="Detected Outliers")
axes[0].set_title("Original Series (with outliers highlighted)")
axes[0].legend()

axes[1].plot(series_interp, color=GREEN, linewidth=1, label="Remove + Interpolate")
axes[1].set_title("Treatment 1: Remove Outlier → Linear Interpolation")
axes[1].legend()

axes[2].plot(series_winsorized, color=ORANGE, linewidth=1, label="Winsorized (clip to 1–99%)")
axes[2].set_title("Treatment 2: Winsorizing (clip to 1st–99th percentile)")
axes[2].legend()

axes[3].plot(series_roll_med, color="purple", linewidth=1, label="Rolling Median Replace")
axes[3].set_title("Treatment 3: Replace with Rolling Median (window=7)")
axes[3].legend()

plt.suptitle("Outlier Treatment Strategy Comparison (STL-detected outliers)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("02_treatment_comparison.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5. IMPACT ON STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

print("\nImpact of Treatment on Summary Statistics:")
print(f"\n{'Metric':<12} {'Original':>12} {'Interpolated':>14} {'Winsorized':>12} {'Roll Median':>12}")
print("-" * 64)
for metric in ["mean", "std", "min", "max"]:
    orig = getattr(series, metric)()
    interp = getattr(series_interp, metric)()
    wins = getattr(series_winsorized, metric)()
    roll = getattr(series_roll_med, metric)()
    print(f"{metric:<12} {orig:>12.3f} {interp:>14.3f} {wins:>12.3f} {roll:>12.3f}")

print("\n✅ Outlier detection and treatment demo complete.")
print("\nRecommendations:")
print("  Seasonal data → STL residual detection (most context-aware)")
print("  Single point errors → Remove + interpolate")
print("  Many outliers → Winsorize (preserves all observations)")
print("  Important extreme events → Keep + add is_outlier flag feature")
