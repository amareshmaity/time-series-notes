"""
04_var_models.py
================
Module 03 — Statistical Models
Topic   : Vector AutoRegression (VAR)

Covers:
  - Simulating a VAR(2) system
  - Stationarity checks for all variables
  - Lag order selection (AIC/BIC)
  - VAR fitting and forecasting
  - Granger causality testing
  - Impulse Response Functions (IRF)
  - FEVD — Forecast Error Variance Decomposition
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import seaborn as sns

plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN = "#2C7BB6", "#D7191C", "#1A9641"


# ─────────────────────────────────────────────────────────────────────────────
# 1. CREATE SYNTHETIC MULTIVARIATE DATASET (Macro indicators)
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
n = 200   # 200 quarters (~50 years)
idx = pd.date_range("1970-01-01", periods=n, freq="QS")

# VAR(1) data generating process
# GDP growth → unemployment (lagged), inflation → GDP (lagged)
gdp        = np.zeros(n)
unemp      = np.zeros(n)
inflation  = np.zeros(n)

# Initialize
gdp[0], unemp[0], inflation[0] = 2.5, 5.5, 2.0

for t in range(1, n):
    gdp[t]       = 0.5 + 0.7  * gdp[t-1]   - 0.3 * unemp[t-1]   + 0.2 * inflation[t-1]  + np.random.randn() * 0.8
    unemp[t]     = 0.3 - 0.4  * gdp[t-1]   + 0.6 * unemp[t-1]   + 0.1 * inflation[t-1]  + np.random.randn() * 0.4
    inflation[t] = 0.4 + 0.2  * gdp[t-1]   - 0.1 * unemp[t-1]   + 0.5 * inflation[t-1]  + np.random.randn() * 0.5

df = pd.DataFrame({
    "gdp_growth":   gdp,
    "unemployment": unemp,
    "inflation":    inflation,
}, index=idx)

print(f"Simulated macro dataset: {df.shape}")
print(df.describe().round(2))


# ─────────────────────────────────────────────────────────────────────────────
# 2. VISUALIZE THE SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
colors = [BLUE, RED, GREEN]
labels = ["GDP Growth (%)", "Unemployment (%)", "Inflation (%)"]
for i, (col, color, label) in enumerate(zip(df.columns, colors, labels)):
    axes[i].plot(df[col], color=color, linewidth=1.5)
    axes[i].axhline(df[col].mean(), color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[i].set_ylabel(label)

plt.suptitle("Simulated Macroeconomic System (VAR Demo)", fontweight="bold")
plt.tight_layout()
plt.savefig("01_var_system.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 3. STATIONARITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Stationarity Check ---")
for col in df.columns:
    p = adfuller(df[col], autolag="AIC")[1]
    print(f"  {col:<20} ADF p={p:.4f}  → {'✅ Stationary' if p < 0.05 else '❌ Non-stationary'}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

train_df = df.iloc[:-16]    # hold out last 16 quarters
test_df  = df.iloc[-16:]
print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. LAG ORDER SELECTION
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Lag Order Selection ---")
model = VAR(train_df)
lag_selection = model.select_order(maxlags=6)
print(lag_selection.summary())

optimal_lag = lag_selection.aic   # use AIC
print(f"\nOptimal lag by AIC: {optimal_lag}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIT VAR
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Fitting VAR ---")
model = VAR(train_df)
fitted = model.fit(maxlags=optimal_lag, ic=None)
print(fitted.summary())

# Check residual autocorrelation
print("\nDurbin-Watson statistics per variable (should be ~2.0):")
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(fitted.resid)
for col, d in zip(df.columns, dw):
    print(f"  {col}: {d:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. FORECAST
# ─────────────────────────────────────────────────────────────────────────────

h = len(test_df)
lag_input = train_df.values[-optimal_lag:]
forecast_arr = fitted.forecast(y=lag_input, steps=h)
forecast_df = pd.DataFrame(forecast_arr, index=test_df.index, columns=df.columns)

# Compute RMSE per variable
print("\n--- Forecast RMSE ---")
for col in df.columns:
    rmse = np.sqrt(((test_df[col].values - forecast_df[col].values)**2).mean())
    print(f"  {col}: RMSE={rmse:.4f}")

# Plot forecasts
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for i, (col, color) in enumerate(zip(df.columns, colors)):
    axes[i].plot(train_df[col][-40:], color=color, linewidth=1.5, label="Historical")
    axes[i].plot(test_df[col],         color="black", linewidth=2, label="Actual")
    axes[i].plot(forecast_df[col],     color=color, linewidth=2, linestyle="--",
                 label=f"VAR({optimal_lag}) Forecast")
    axes[i].axvline(train_df.index[-1], color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    axes[i].set_title(col.replace("_", " ").title())
    axes[i].legend(fontsize=8)

plt.suptitle(f"VAR({optimal_lag}) Forecasts — Macro System", fontweight="bold")
plt.tight_layout()
plt.savefig("02_var_forecasts.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 8. GRANGER CAUSALITY
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Granger Causality Tests ---")
print(f"{'Cause → Effect':<35} {'p-value':>10} {'Significant':>12}")
print("-" * 60)

for cause in df.columns:
    for effect in df.columns:
        if cause == effect:
            continue
        try:
            res = grangercausalitytests(
                train_df[[effect, cause]], maxlag=optimal_lag, verbose=False
            )
            min_p = min(res[lag][0]["ssr_ftest"][1] for lag in range(1, optimal_lag + 1))
            sig = "✅ Yes" if min_p < 0.05 else "❌ No"
            print(f"  {cause:<20} → {effect:<12}  p={min_p:.4f}   {sig}")
        except Exception:
            pass

# Granger causality heatmap
variables = df.columns.tolist()
cmat = pd.DataFrame(np.nan, index=variables, columns=variables)
for cause in variables:
    for effect in variables:
        if cause == effect:
            continue
        try:
            res = grangercausalitytests(train_df[[effect, cause]], maxlag=optimal_lag, verbose=False)
            cmat.loc[cause, effect] = min(res[l][0]["ssr_ftest"][1] for l in range(1, optimal_lag+1))
        except Exception:
            pass

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cmat.astype(float), annot=True, fmt=".3f", cmap="RdYlGn_r",
            vmin=0, vmax=0.1, ax=ax, linewidths=0.5)
ax.set_title("Granger Causality p-values\n(row→col; green<0.05 = significant)")
plt.tight_layout()
plt.savefig("03_granger_heatmap.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 9. IMPULSE RESPONSE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Impulse Response Functions ---")
irf = fitted.irf(periods=16)

# Plot: shock to gdp_growth → all variables
irf.plot(impulse="gdp_growth", figsize=(13, 8))
plt.suptitle("IRF: Shock to GDP Growth → All Variables", fontweight="bold")
plt.tight_layout()
plt.savefig("04_irf_gdp.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 10. FORECAST ERROR VARIANCE DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- FEVD ---")
fevd = fitted.fevd(periods=16)
fevd.plot(figsize=(13, 8))
plt.suptitle("Forecast Error Variance Decomposition", fontweight="bold")
plt.tight_layout()
plt.savefig("05_fevd.png", bbox_inches="tight")
plt.show()

print("\n✅ VAR model demo complete.")
print("   GDP growth Granger-causes unemployment and inflation — consistent with economic theory")
