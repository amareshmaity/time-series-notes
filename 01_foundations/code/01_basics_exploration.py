"""
01_basics_exploration.py
========================
Module 01 — Foundations of Time Series
Topic   : Loading, exploring, and diagnosing a time series

Covers:
  - Loading a time series dataset with proper datetime parsing
  - Setting a DatetimeIndex and resampling
  - Rolling statistics (mean, std)
  - Stationarity tests: ADF and KPSS
  - ACF and PACF plots
  - Lag plots

Dataset: Monthly airline passengers (Box & Jenkins, 1976)
         Available from seaborn or online
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from pandas.plotting import lag_plot

# ── Plotting defaults ────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
BLUE  = "#2C7BB6"
RED   = "#D7191C"
GRAY  = "#888888"


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_airline_data() -> pd.Series:
    """Load the classic monthly airline passengers dataset."""
    url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets"
        "/master/airline-passengers.csv"
    )
    try:
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        series = df.squeeze()           # DataFrame → Series
        series.index.freq = "MS"        # Monthly Start frequency
        series.name = "Passengers"
        print(f"Loaded {len(series)} monthly observations: "
              f"{series.index[0].date()} → {series.index[-1].date()}")
        return series
    except Exception:
        # Fallback: use seaborn built-in dataset
        df = sns.load_dataset("flights")
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str), format="%Y-%B"
        )
        series = df.set_index("date")["passengers"].sort_index()
        series.index.freq = "MS"
        series.name = "Passengers"
        return series


series = load_airline_data()


# ─────────────────────────────────────────────────────────────────────────────
# 2. RAW SERIES PLOT
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(series.index, series.values, color=BLUE, linewidth=1.8, label="Monthly Passengers")
ax.set_title("Monthly Airline Passengers (1949–1960)", fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Passengers (thousands)")
ax.legend()
plt.tight_layout()
plt.savefig("01_raw_series.png", bbox_inches="tight")
plt.show()

print("\nBasic Statistics:")
print(series.describe().round(2))


# ─────────────────────────────────────────────────────────────────────────────
# 3. RESAMPLING
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Resampling ---")

# Quarterly: sum of monthly passengers
quarterly = series.resample("QS").sum()
print(f"\nMonthly → Quarterly (sum): {len(quarterly)} observations")
print(quarterly.head(4))

# Annual: mean of monthly passengers
annual = series.resample("AS").mean()
print(f"\nMonthly → Annual (mean): {len(annual)} observations")
print(annual)

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
axes[0].plot(series, color=BLUE, label="Monthly");       axes[0].set_title("Monthly")
axes[1].bar(quarterly.index, quarterly.values, width=80, color=BLUE, alpha=0.7); axes[1].set_title("Quarterly (sum)")
axes[2].bar(annual.index, annual.values, width=300, color=RED, alpha=0.7);       axes[2].set_title("Annual (mean)")
for ax in axes:
    ax.legend()
plt.suptitle("Resampling Demonstration", fontweight="bold")
plt.tight_layout()
plt.savefig("02_resampling.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 4. ROLLING STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

window = 12   # 12-month rolling window

rolling_mean = series.rolling(window=window).mean()
rolling_std  = series.rolling(window=window).std()

fig, axes = plt.subplots(2, 1, figsize=(13, 7))

axes[0].plot(series, color=BLUE, alpha=0.6, label="Original")
axes[0].plot(rolling_mean, color=RED, linewidth=2, label=f"Rolling Mean ({window}m)")
axes[0].set_title("Rolling Mean — Drift = Trend or Non-stationarity")
axes[0].legend()

axes[1].plot(rolling_std, color=RED, linewidth=2, label=f"Rolling Std ({window}m)")
axes[1].set_title("Rolling Std — Growing = Heteroskedasticity (variance non-constant)")
axes[1].legend()

plt.suptitle("Rolling Statistics Diagnostic", fontweight="bold")
plt.tight_layout()
plt.savefig("03_rolling_stats.png", bbox_inches="tight")
plt.show()

print("\nObservation:")
print("  Rolling mean is drifting upward → NON-STATIONARY (trend + variance growth)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. STATIONARITY TESTS — ADF & KPSS
# ─────────────────────────────────────────────────────────────────────────────

def run_stationarity_tests(s: pd.Series, label: str = "") -> None:
    """Run ADF and KPSS tests and print a combined report."""
    print(f"\n{'='*55}")
    print(f"  Stationarity Tests{': ' + label if label else ''}")
    print(f"{'='*55}")

    # ADF test
    adf_stat, adf_p, adf_lags, adf_nobs, adf_cv, _ = adfuller(s.dropna(), autolag="AIC")
    print(f"\n  ADF Test:")
    print(f"    Statistic : {adf_stat:.4f}")
    print(f"    p-value   : {adf_p:.4f}")
    print(f"    Lags used : {adf_lags}")
    print(f"    Critical values: 1%={adf_cv['1%']:.3f} | 5%={adf_cv['5%']:.3f} | 10%={adf_cv['10%']:.3f}")
    adf_conclusion = "STATIONARY" if adf_p < 0.05 else "NON-STATIONARY"
    print(f"    → {adf_conclusion} (p {'<' if adf_p < 0.05 else '≥'} 0.05)")

    # KPSS test
    kpss_stat, kpss_p, kpss_lags, kpss_cv = kpss(s.dropna(), regression="c", nlags="auto")
    print(f"\n  KPSS Test:")
    print(f"    Statistic : {kpss_stat:.4f}")
    print(f"    p-value   : {kpss_p:.4f}")
    print(f"    Lags used : {kpss_lags}")
    kpss_conclusion = "NON-STATIONARY" if kpss_p < 0.05 else "STATIONARY"
    print(f"    → {kpss_conclusion} (p {'<' if kpss_p < 0.05 else '≥'} 0.05)")

    # Combined conclusion
    print(f"\n  Combined Conclusion:")
    if adf_p < 0.05 and kpss_p >= 0.05:
        print("    ✅ STATIONARY — both tests agree")
    elif adf_p >= 0.05 and kpss_p < 0.05:
        print("    ❌ NON-STATIONARY — both tests agree → apply differencing")
    elif adf_p < 0.05 and kpss_p < 0.05:
        print("    ⚠️  TREND STATIONARY — stationary around a trend")
    else:
        print("    ⚠️  DIFFERENCE STATIONARY — has unit root → first difference")
    print(f"{'='*55}")


# Test the raw series
run_stationarity_tests(series, label="Raw Series")

# Log transform to stabilize variance
series_log = np.log(series)
run_stationarity_tests(series_log, label="Log-Transformed Series")

# First difference of log (remove trend)
series_log_diff = series_log.diff().dropna()
run_stationarity_tests(series_log_diff, label="Log + First Difference")

# Log + first diff + seasonal diff (remove seasonality)
series_log_diff_sdiff = series_log_diff.diff(12).dropna()
run_stationarity_tests(series_log_diff_sdiff, label="Log + First Diff + Seasonal Diff(12)")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ACF AND PACF PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_acf_pacf(s: pd.Series, title: str, lags: int = 40, save_name: str = None):
    """Plot ACF and PACF side by side."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    plot_acf(s.dropna(), lags=lags, alpha=0.05, ax=axes[0])
    plot_pacf(s.dropna(), lags=lags, method="ywm", alpha=0.05, ax=axes[1])
    axes[0].set_title(f"ACF — {title}")
    axes[1].set_title(f"PACF — {title}")
    plt.suptitle(title, fontweight="bold")
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, bbox_inches="tight")
    plt.show()


plot_acf_pacf(series, "Raw Series (Non-Stationary)", save_name="04_acf_pacf_raw.png")
plot_acf_pacf(series_log_diff_sdiff, "After Log + Diff + Seasonal Diff", save_name="05_acf_pacf_transformed.png")

print("\nACF/PACF Reading:")
print("  Raw series: ACF decays very slowly → clear non-stationarity")
print("  Transformed: ACF/PACF show sharp patterns → use to identify ARIMA orders")


# ─────────────────────────────────────────────────────────────────────────────
# 7. LJUNG-BOX TEST ON RESIDUALS
# ─────────────────────────────────────────────────────────────────────────────

# Quick white noise check on the final transformed series
lb_result = acorr_ljungbox(series_log_diff_sdiff, lags=[10, 20, 30], return_df=True)
print("\nLjung-Box Test on Transformed Series:")
print(lb_result.to_string())
print("\nIf all p-values > 0.05 → transformed series is close to white noise")


# ─────────────────────────────────────────────────────────────────────────────
# 8. LAG PLOTS
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, ax in enumerate(axes.flatten()):
    lag_plot(series, lag=i + 1, ax=ax, c=BLUE, alpha=0.5, s=15)
    ax.set_title(f"Lag {i+1}", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")

plt.suptitle("Lag Plots — Raw Series (tight linear clusters = high autocorrelation)",
             fontweight="bold")
plt.tight_layout()
plt.savefig("06_lag_plots.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 9. SUMMARY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 2, figsize=(14, 11))

# Raw series
axes[0, 0].plot(series, color=BLUE, linewidth=1.5)
axes[0, 0].set_title("Raw Series")

# Rolling mean and std
axes[0, 1].plot(series, color=BLUE, alpha=0.4, label="Original")
axes[0, 1].plot(rolling_mean, color=RED, label="Rolling Mean")
axes[0, 1].plot(rolling_std * 5, color="orange", label="Rolling Std ×5")
axes[0, 1].legend(fontsize=8)
axes[0, 1].set_title("Rolling Statistics")

# Log-transformed
axes[1, 0].plot(series_log, color=BLUE, linewidth=1.5)
axes[1, 0].set_title("Log-Transformed")

# First differenced log
axes[1, 1].plot(series_log_diff, color=RED, linewidth=1.2)
axes[1, 1].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[1, 1].set_title("Log + First Difference")

# After all transformations
axes[2, 0].plot(series_log_diff_sdiff, color="green", linewidth=1.2)
axes[2, 0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[2, 0].set_title("Log + First Diff + Seasonal Diff(12)")

# ACF of final series
from statsmodels.graphics.tsaplots import plot_acf as _plot_acf
_plot_acf(series_log_diff_sdiff, lags=30, ax=axes[2, 1], alpha=0.05)
axes[2, 1].set_title("ACF of Final Stationary Series")

plt.suptitle("Time Series Exploration Dashboard", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("07_dashboard.png", bbox_inches="tight")
plt.show()

print("\n✅ Exploration complete. Key findings:")
print("  - Raw series: trended + growing variance → NON-STATIONARY")
print("  - Fix: log transform → first diff → seasonal diff(12)")
print("  - Result: stationary series ready for SARIMA modeling")
print("  - ACF/PACF of final series: use to identify ARIMA(p,d,q)(P,D,Q,12) orders")
