"""
01_resampling_demo.py
=====================
Module 02 — Data Engineering for Time Series
Topic   : Resampling and Frequency Conversion

Covers:
  - Downsampling with different aggregation functions
  - Upsampling with ffill, bfill, and interpolation
  - OHLC aggregation for financial data
  - Handling gaps vs. resampling
  - Multi-column resampling with different rules
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN = "#2C7BB6", "#D7191C", "#1A9641"


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD HOURLY DATA (simulated electricity demand)
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
n_hours = 24 * 7 * 52   # 52 weeks of hourly data
idx_hourly = pd.date_range("2023-01-02", periods=n_hours, freq="h")
t = np.arange(n_hours)

demand = (
    500
    + 0.01 * t                                           # slight upward trend
    + 80 * np.sin(2 * np.pi * t / 24)                   # daily seasonality
    + 40 * np.sin(2 * np.pi * t / (24 * 7))             # weekly seasonality
    + np.random.normal(0, 15, n_hours)                   # noise
)

series_hourly = pd.Series(demand, index=idx_hourly, name="electricity_demand_mw")
print(f"Loaded hourly series: {len(series_hourly)} observations")
print(f"Range: {series_hourly.index[0]} → {series_hourly.index[-1]}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DOWNSAMPLING — Hourly → Daily → Weekly → Monthly
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Downsampling ---")

# Electricity demand: sum = total energy consumed (kWh if MW × 1h)
daily_sum    = series_hourly.resample("D").sum()
weekly_sum   = series_hourly.resample("W").sum()
monthly_sum  = series_hourly.resample("MS").sum()

# Mean for average load
daily_mean   = series_hourly.resample("D").mean()
daily_max    = series_hourly.resample("D").max()
daily_min    = series_hourly.resample("D").min()

print(f"Hourly → Daily  : {len(series_hourly)} → {len(daily_sum)} observations")
print(f"Hourly → Weekly : {len(series_hourly)} → {len(weekly_sum)} observations")
print(f"Hourly → Monthly: {len(series_hourly)} → {len(monthly_sum)} observations")

# Plot comparison
fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=False)
axes[0].plot(series_hourly[:24*14], color=BLUE, linewidth=0.8)
axes[0].set_title("Hourly (first 2 weeks)")

axes[1].plot(daily_mean[:30], color=RED, linewidth=1.5, marker="o", markersize=3)
axes[1].set_title("Daily Mean")

axes[2].plot(weekly_sum, color=GREEN, linewidth=2, marker="s", markersize=4)
axes[2].set_title("Weekly Sum")

axes[3].bar(monthly_sum.index, monthly_sum.values, width=20, color=BLUE, alpha=0.7)
axes[3].set_title("Monthly Sum")

plt.suptitle("Downsampling: Hourly → Daily → Weekly → Monthly", fontweight="bold")
plt.tight_layout()
plt.savefig("01_downsampling.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 3. MULTI-COLUMN RESAMPLING (Different Rules Per Column)
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Multi-column resampling with different rules ---")

# Simulate a multi-column daily dataset
idx_daily = pd.date_range("2023-01-01", periods=365, freq="D")
df = pd.DataFrame({
    "sales":     np.random.poisson(lam=500, size=365),
    "price":     50 + np.random.randn(365) * 2,
    "units":     np.random.randint(80, 120, size=365),
    "in_stock":  np.random.randint(0, 2, size=365),   # binary state
}, index=idx_daily)

# Different aggregation per column
df_monthly = df.resample("MS").agg({
    "sales":    "sum",    # total monthly sales
    "price":    "mean",   # average monthly price
    "units":    "sum",    # total units sold
    "in_stock": "last",   # state at end of month
})
print(df_monthly)


# ─────────────────────────────────────────────────────────────────────────────
# 4. UPSAMPLING — Monthly → Daily
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Upsampling: Monthly → Daily ---")

monthly_budget = pd.Series(
    [100_000, 105_000, 98_000, 110_000, 115_000, 120_000],
    index=pd.date_range("2023-01-01", periods=6, freq="MS"),
    name="monthly_budget",
)

# Different strategies
budget_ffill   = monthly_budget.resample("D").ffill()
budget_bfill   = monthly_budget.resample("D").bfill()
budget_interp  = monthly_budget.resample("D").interpolate(method="linear")

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

for ax, series_up, label, color in [
    (axes[0], budget_ffill,  "Forward Fill (step function)",    BLUE),
    (axes[1], budget_bfill,  "Backward Fill",                   RED),
    (axes[2], budget_interp, "Linear Interpolation (smooth)",   GREEN),
]:
    ax.plot(series_up, color=color, linewidth=1.5, label=label)
    ax.scatter(monthly_budget.index, monthly_budget.values,
               color="black", s=50, zorder=5, label="Monthly observations")
    ax.set_title(label)
    ax.legend(fontsize=9)

plt.suptitle("Upsampling Strategies: Monthly Budget → Daily", fontweight="bold")
plt.tight_layout()
plt.savefig("02_upsampling.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5. OHLC AGGREGATION — Financial Data
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- OHLC Aggregation ---")

try:
    # Download 1-minute Apple stock data (recent 5 days)
    ticker = yf.Ticker("AAPL")
    df_min = ticker.history(period="5d", interval="1m")
    price_min = df_min["Close"].dropna()
    print(f"Loaded {len(price_min)} minute-level price observations")

    # Aggregate minute → hourly OHLC
    ohlc_hourly = price_min.resample("h").ohlc()
    ohlc_hourly["volume"] = df_min["Volume"].resample("h").sum()
    print("\nHourly OHLC bars:")
    print(ohlc_hourly.head(8))

    # Plot candlestick-style (simplified)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(ohlc_hourly.index, ohlc_hourly["high"] - ohlc_hourly["low"],
           bottom=ohlc_hourly["low"], width=0.03, color="steelblue", alpha=0.4, label="High-Low range")
    ax.plot(ohlc_hourly.index, ohlc_hourly["close"], color=RED, linewidth=1.5, label="Close")
    ax.set_title("AAPL — Minute Data Resampled to Hourly OHLC")
    ax.legend()
    plt.tight_layout()
    plt.savefig("03_ohlc.png", bbox_inches="tight")
    plt.show()

except Exception as e:
    print(f"Could not fetch live data: {e}")
    print("Skipping OHLC demo — run when network is available")


# ─────────────────────────────────────────────────────────────────────────────
# 6. GAP DETECTION AND FILLING
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Gap Detection and Filling ---")

# Introduce artificial gaps into the daily series
series_with_gaps = daily_mean.copy()
gap_indices = np.random.choice(len(series_with_gaps), size=15, replace=False)
series_with_gaps.iloc[gap_indices] = np.nan

# Detect gaps
expected_idx = pd.date_range(
    start=series_with_gaps.index.min(),
    end=series_with_gaps.index.max(),
    freq="D",
)
missing_dates = expected_idx[~expected_idx.isin(series_with_gaps.dropna().index)]
print(f"Gaps found: {series_with_gaps.isna().sum()} missing values")

# Fill gaps
series_gap_filled = series_with_gaps.reindex(expected_idx).interpolate(method="linear")

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
axes[0].plot(series_with_gaps, color=BLUE, linewidth=1.2)
axes[0].scatter(series_with_gaps.index[series_with_gaps.isna()],
                [series_with_gaps.mean()] * series_with_gaps.isna().sum(),
                color=RED, s=40, zorder=5, label="Missing values")
axes[0].set_title("Series with Gaps")
axes[0].legend()

axes[1].plot(series_gap_filled, color=GREEN, linewidth=1.2)
axes[1].set_title("After Linear Interpolation to Fill Gaps")

plt.suptitle("Gap Detection and Filling", fontweight="bold")
plt.tight_layout()
plt.savefig("04_gap_filling.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n✅ Resampling demo complete.")
print("\nKey points covered:")
print("  1. Downsampling: sum for energy, mean for temperature, last for state variables")
print("  2. Multi-column: each column can have a different aggregation rule")
print("  3. Upsampling: ffill for states, linear interpolation for smooth quantities")
print("  4. OHLC: standard for financial tick/minute → hourly/daily aggregation")
print("  5. Gaps vs resampling: gaps need reindex → impute workflow")
