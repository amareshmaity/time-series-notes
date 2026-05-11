"""
02_missing_values.py
====================
Module 02 — Data Engineering for Time Series
Topic   : Handling Missing Values

Covers:
  - Diagnosing missing value patterns and gap lengths
  - Forward fill, backward fill
  - Linear, polynomial, spline, and time-based interpolation
  - Seasonal mean fill
  - STL-based imputation
  - Side-by-side strategy comparison
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf

plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE = "#2C7BB6", "#D7191C", "#1A9641", "#F07D00"


# ─────────────────────────────────────────────────────────────────────────────
# 1. CREATE SYNTHETIC SERIES WITH KNOWN GAPS
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(0)
n = 240   # 20 years of monthly data
idx = pd.date_range("2004-01-01", periods=n, freq="MS")
t = np.arange(n)

# True underlying series
true_series = pd.Series(
    200 + 0.8 * t
    + 40 * np.sin(2 * np.pi * t / 12)
    + np.random.normal(0, 10, n),
    index=idx,
    name="temperature_anomaly",
)

# Introduce gaps of different types
series_with_gaps = true_series.copy()
series_with_gaps.iloc[10:12]  = np.nan   # Short gap (2 months)
series_with_gaps.iloc[50:58]  = np.nan   # Medium gap (8 months)
series_with_gaps.iloc[120:124]= np.nan   # Seasonal gap (4 months)
# Random scattered missing
rng_idx = np.random.choice(range(140, 220), size=10, replace=False)
series_with_gaps.iloc[rng_idx] = np.nan

total_missing = series_with_gaps.isna().sum()
print(f"Total missing: {total_missing} / {n} ({total_missing/n*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DIAGNOSE MISSING VALUE PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

def describe_gaps(series: pd.Series) -> pd.DataFrame:
    """Return DataFrame describing each contiguous gap."""
    is_null = series.isna()
    gaps = []
    in_gap = False
    start = None
    for i, (dt, val) in enumerate(series.items()):
        if pd.isna(val) and not in_gap:
            in_gap = True
            start = dt
        elif not pd.isna(val) and in_gap:
            in_gap = False
            gaps.append({"start": start, "end": series.index[i-1],
                         "length": (series.index[i-1] - start).days // 30 + 1})
    if in_gap:
        gaps.append({"start": start, "end": series.index[-1],
                     "length": series.isna()[::-1].idxmax()})
    return pd.DataFrame(gaps)

gap_df = describe_gaps(series_with_gaps)
print("\nGap Analysis:")
print(gap_df)

# Visualize missing pattern
fig, axes = plt.subplots(2, 1, figsize=(13, 7))
axes[0].plot(true_series, color=BLUE, alpha=0.3, linewidth=1, label="True (complete)")
axes[0].plot(series_with_gaps, color=BLUE, linewidth=1.5, label="Observed (with gaps)")
axes[0].set_title("Series with Introduced Missing Values")
axes[0].legend()

axes[1].fill_between(series_with_gaps.index,
                     series_with_gaps.isna().astype(int),
                     color=RED, alpha=0.6, label="Missing (=1)")
axes[1].set_title("Missing Value Timeline")
axes[1].set_ylabel("Is Missing")
axes[1].legend()

plt.suptitle("Missing Value Pattern Diagnosis", fontweight="bold")
plt.tight_layout()
plt.savefig("01_missing_pattern.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 3. APPLY MULTIPLE IMPUTATION STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

strategies = {}

# Forward fill
strategies["ffill"] = series_with_gaps.ffill()

# Backward fill
strategies["bfill"] = series_with_gaps.bfill()

# Linear interpolation
strategies["linear"] = series_with_gaps.interpolate(method="linear")

# Cubic spline interpolation
strategies["cubic_spline"] = series_with_gaps.interpolate(method="spline", order=3)

# Seasonal mean fill (same month mean)
def seasonal_fill(s, period="month"):
    s = s.copy()
    if period == "month":
        group_key = s.index.month
    means = s.groupby(group_key).transform("mean")
    return s.fillna(means)

strategies["seasonal_mean"] = seasonal_fill(series_with_gaps, "month")

# STL-based imputation
def stl_impute(s: pd.Series, period: int = 12, n_iter: int = 5) -> pd.Series:
    s_temp = s.interpolate(method="linear").ffill().bfill()
    nan_mask = s.isna()
    for _ in range(n_iter):
        result = STL(s_temp, period=period, robust=True).fit()
        s_temp[nan_mask] = (result.trend + result.seasonal)[nan_mask]
    return s_temp

strategies["stl_impute"] = stl_impute(series_with_gaps, period=12)


# ─────────────────────────────────────────────────────────────────────────────
# 4. COMPARE STRATEGIES VISUALLY
# ─────────────────────────────────────────────────────────────────────────────

nan_mask = series_with_gaps.isna()
colors = [BLUE, RED, GREEN, ORANGE, "purple", "brown"]
labels = list(strategies.keys())

fig, axes = plt.subplots(3, 2, figsize=(14, 11))

for ax, (name, imputed), color in zip(axes.flatten(), strategies.items(), colors):
    ax.plot(true_series, color="lightgray", linewidth=2, label="True (complete)", zorder=1)
    ax.plot(series_with_gaps, color="black", linewidth=1, label="Observed", zorder=2)
    ax.plot(imputed[nan_mask], color=color, linewidth=0, marker="o",
            markersize=4, label=f"{name} (imputed)", zorder=3)
    ax.set_title(f"Strategy: {name}")
    ax.legend(fontsize=7)

plt.suptitle("Imputation Strategy Comparison (gray = true series, dots = imputed values)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("02_strategy_comparison.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5. QUANTITATIVE COMPARISON — RMSE AT MISSING POSITIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\nImputation Error (RMSE at missing positions vs. true values):")
print(f"{'Strategy':<20} {'RMSE':>10} {'MAE':>10}")
print("-" * 42)

for name, imputed in strategies.items():
    errors = imputed[nan_mask] - true_series[nan_mask]
    rmse = np.sqrt((errors**2).mean())
    mae  = errors.abs().mean()
    print(f"{name:<20} {rmse:>10.3f} {mae:>10.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ACF PRESERVATION CHECK
# ─────────────────────────────────────────────────────────────────────────────

print("\nChecking ACF structure preservation...")

fig, axes = plt.subplots(len(strategies) + 1, 1, figsize=(12, 2.5 * (len(strategies) + 1)))

plot_acf(true_series, lags=30, ax=axes[0], alpha=0.05)
axes[0].set_title("ACF — True Series (target)")

for i, (name, imputed) in enumerate(strategies.items()):
    plot_acf(imputed, lags=30, ax=axes[i+1], alpha=0.05)
    axes[i+1].set_title(f"ACF — {name}")

plt.suptitle("ACF Structure Preservation After Imputation", fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("03_acf_preservation.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n✅ Missing value imputation demo complete.")
print("\nRecommendation summary:")
print("  Short gaps (1-3 steps)   → forward fill or linear interpolation")
print("  Seasonal data            → seasonal mean fill or STL imputation")
print("  Smooth measurements      → cubic spline or linear interpolation")
print("  General recommendation   → STL imputation (best RMSE on seasonal data)")
