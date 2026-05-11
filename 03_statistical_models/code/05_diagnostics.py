"""
05_diagnostics.py
=================
Module 03 — Statistical Models
Topic   : Model Selection and Diagnostics

Covers:
  - AIC/BIC grid search across ARIMA orders
  - Full residual diagnostic dashboard
  - Ljung-Box test
  - Walk-forward validation
  - Diebold-Mariano test
  - Model comparison leaderboard
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE = "#2C7BB6", "#D7191C", "#1A9641", "#F07D00"


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_airline():
    try:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        s = df.squeeze(); s.index.freq = "MS"
    except Exception:
        df = sns.load_dataset("flights")
        df["date"] = pd.to_datetime(df["year"].astype(str)+"-"+df["month"].astype(str), format="%Y-%B")
        s = df.set_index("date")["passengers"].sort_index(); s.index.freq = "MS"
    s.name = "Passengers"
    return s

series = load_airline()
log_series = np.log(series)
train = log_series[:-24]
test  = log_series[-24:]
h = len(test)


# ─────────────────────────────────────────────────────────────────────────────
# 2. AIC/BIC GRID SEARCH
# ─────────────────────────────────────────────────────────────────────────────

print("="*60)
print("  AIC/BIC Grid Search — SARIMA variants")
print("="*60)

candidates = []
search_space = [
    ((0, 1, 1), (0, 1, 1, 12), "SARIMA(0,1,1)(0,1,1)[12] — Airline"),
    ((1, 1, 1), (1, 1, 1, 12), "SARIMA(1,1,1)(1,1,1)[12]"),
    ((2, 1, 0), (1, 1, 0, 12), "SARIMA(2,1,0)(1,1,0)[12]"),
    ((0, 1, 2), (0, 1, 1, 12), "SARIMA(0,1,2)(0,1,1)[12]"),
    ((1, 1, 0), (0, 1, 1, 12), "SARIMA(1,1,0)(0,1,1)[12]"),
    ((2, 1, 2), (1, 1, 1, 12), "SARIMA(2,1,2)(1,1,1)[12] — Complex"),
]

for order, seasonal_order, label in search_space:
    try:
        m = SARIMAX(train, order=order, seasonal_order=seasonal_order).fit(disp=False, maxiter=200)
        candidates.append({"Model": label, "AIC": m.aic, "BIC": m.bic, "fitted": m})
        print(f"  {label:<45} AIC={m.aic:.2f}  BIC={m.bic:.2f}")
    except Exception as e:
        print(f"  {label:<45} ERROR: {e}")

df_cands = pd.DataFrame([{k: v for k, v in c.items() if k != "fitted"} for c in candidates])
df_cands = df_cands.sort_values("AIC")
print(f"\nBest by AIC: {df_cands.iloc[0]['Model']}")
print(f"Best by BIC: {df_cands.sort_values('BIC').iloc[0]['Model']}")

# Select best model
best = next(c for c in candidates if c["Model"] == df_cands.iloc[0]["Model"])
best_fitted = best["fitted"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. FULL RESIDUAL DIAGNOSTIC DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def residual_dashboard(fitted_model, model_name: str):
    residuals = fitted_model.resid.dropna()

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"Residual Diagnostics — {model_name}", fontsize=13, fontweight="bold")

    # 1. Residual time series
    axes[0, 0].plot(residuals, color=BLUE, linewidth=0.8)
    axes[0, 0].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[0, 0].set_title("Residuals vs. Time")

    # 2. Histogram + Normal curve
    axes[0, 1].hist(residuals, bins=25, color=BLUE, edgecolor="white", density=True, alpha=0.7)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()),
                    color=RED, linewidth=2, label="Normal")
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].legend()

    # 3. Q-Q Plot
    stats.probplot(residuals, plot=axes[0, 2])
    axes[0, 2].set_title("Q-Q Plot (Normal)")

    # 4. ACF
    plot_acf(residuals, lags=30, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title("ACF of Residuals")

    # 5. PACF
    plot_pacf(residuals, lags=30, ax=axes[1, 1], alpha=0.05, method="ywm")
    axes[1, 1].set_title("PACF of Residuals")

    # 6. Squared residuals (heteroskedasticity check)
    axes[1, 2].plot(residuals**2, color=ORANGE, linewidth=0.8)
    axes[1, 2].set_title("Squared Residuals (Variance Stability)")

    plt.tight_layout()
    plt.savefig(f"diagnostics_{model_name.replace(' ','_').replace('/','')}.png", bbox_inches="tight")
    plt.show()

    # Numerical summary
    print(f"\nResidual Statistics — {model_name}")
    print(f"  Mean:         {residuals.mean():.6f}  (should be ≈ 0)")
    print(f"  Std:          {residuals.std():.6f}")
    print(f"  Skewness:     {stats.skew(residuals):.4f}  (|<0.5| = good)")
    print(f"  Excess Kurt:  {stats.kurtosis(residuals):.4f}  (≈ 0 = normal tails)")
    _, p_norm = stats.normaltest(residuals)
    print(f"  Normal test:  p={p_norm:.4f}  ({'✅' if p_norm > 0.05 else '⚠️ Non-normal tails'})")


residual_dashboard(best_fitted, best["Model"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. LJUNG-BOX TEST
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Ljung-Box Test on Best Model Residuals ---")
lb_result = acorr_ljungbox(
    best_fitted.resid.dropna(),
    lags=[5, 10, 15, 20, 30],
    return_df=True,
    model_df=2,
)
print(lb_result.to_string())
all_pass = (lb_result["lb_pvalue"] > 0.05).all()
print(f"\nAll lags pass (p > 0.05)? {'✅ YES — residuals are white noise' if all_pass else '❌ NO — model under-specified'}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Walk-Forward Validation ---")

n_origins = 8
origin_size = len(test) // n_origins
wf_results = []

for i in range(n_origins):
    train_end = len(train) + i * origin_size
    actual_end = train_end + origin_size
    if actual_end > len(log_series):
        break

    wf_train  = log_series.iloc[:train_end]
    wf_actual = log_series.iloc[train_end:actual_end]

    try:
        m = SARIMAX(wf_train, order=best["fitted"].model.order,
                    seasonal_order=best["fitted"].model.seasonal_order).fit(disp=False, maxiter=200)
        fc = m.get_forecast(steps=len(wf_actual)).predicted_mean
        errors = wf_actual.values - fc.values[:len(wf_actual)]
        rmse = np.sqrt((errors**2).mean())
        mae  = np.abs(errors).mean()
        wf_results.append({"origin": wf_train.index[-1].date(), "RMSE": rmse, "MAE": mae})
        print(f"  Origin {i+1}: {wf_train.index[-1].date()} | RMSE={rmse:.4f} | MAE={mae:.4f}")
    except Exception as e:
        print(f"  Origin {i+1}: ERROR — {e}")

wf_df = pd.DataFrame(wf_results)
print(f"\nMean RMSE across {len(wf_df)} origins: {wf_df['RMSE'].mean():.4f} (log scale)")


# ─────────────────────────────────────────────────────────────────────────────
# 6. DIEBOLD-MARIANO TEST — SARIMA vs ETS
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Diebold-Mariano Test: SARIMA vs. HW-Multiplicative ---")

# Fit competing ETS model
hw_fit = ExponentialSmoothing(
    np.exp(train), trend="add", seasonal="mul", seasonal_periods=12,
    initialization_method="estimated"
).fit(optimized=True, disp=False)

# Forecasts on test set (original scale)
sarima_fc = np.exp(best_fitted.get_forecast(h).predicted_mean.values)
hw_fc     = hw_fit.forecast(h).values
actual    = np.exp(test.values)

# DM test (MSE loss)
e1 = actual - sarima_fc
e2 = actual - hw_fc
d  = e1**2 - e2**2
d_bar = d.mean()
d_var = ((d - d_bar)**2).sum() / (len(d) * (len(d) - 1))
dm_stat = d_bar / np.sqrt(d_var)
p_val   = 2 * (1 - stats.t.cdf(abs(dm_stat), df=len(d)-1))

print(f"  SARIMA  RMSE: {np.sqrt((e1**2).mean()):.2f}")
print(f"  HW-Mul  RMSE: {np.sqrt((e2**2).mean()):.2f}")
print(f"  DM statistic: {dm_stat:.4f}")
print(f"  p-value:      {p_val:.4f}")
if p_val < 0.05:
    better = "SARIMA" if dm_stat > 0 else "HW-Multiplicative"
    print(f"  Conclusion: Significant difference — {better} is better ✅")
else:
    print("  Conclusion: No significant difference in accuracy (p > 0.05) — models are equivalent")


# ─────────────────────────────────────────────────────────────────────────────
# 7. FINAL LEADERBOARD VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# AIC comparison
ax = axes[0]
df_plot = df_cands.copy()
df_plot["Short"] = df_plot["Model"].str.extract(r"(SARIMA\(\d,\d,\d\)\(\d,\d,\d\)\[12\])")
ax.barh(df_plot["Short"].values, df_plot["AIC"].values, color=BLUE, edgecolor="white")
ax.set_xlabel("AIC")
ax.set_title("Model Comparison: AIC (lower = better)")
ax.axvline(df_plot["AIC"].min(), color=RED, linewidth=1, linestyle="--")

# RMSE comparison
model_rmse = {
    "SARIMA(0,1,1)(0,1,1)": np.sqrt(((actual - sarima_fc)**2).mean()),
    "HW-Multiplicative":    np.sqrt(((actual - hw_fc)**2).mean()),
    "Seasonal Naive":       np.sqrt(((actual - np.exp(train.values[-12:].tolist() * 2))**2).mean()),
}
ax2 = axes[1]
names  = list(model_rmse.keys())
values = list(model_rmse.values())
ax2.barh(names, values, color=[RED, GREEN, ORANGE], edgecolor="white")
ax2.set_xlabel("RMSE (original scale)")
ax2.set_title("Out-of-Sample RMSE (lower = better)")
for i, v in enumerate(values):
    ax2.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=10)

plt.suptitle("Model Selection Dashboard", fontweight="bold")
plt.tight_layout()
plt.savefig("01_model_selection.png", bbox_inches="tight")
plt.show()

print("\n✅ Diagnostics demo complete.")
print("   Steps demonstrated: AIC/BIC selection → residual diagnostics → Ljung-Box → walk-forward → DM test")
