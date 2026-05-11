"""
03_ar_ma_arma_arima_sarima.py
==============================
Module 03 — Statistical Models
Topic   : AR, MA, ARMA, ARIMA, SARIMA family

Covers:
  - AR(p), MA(q), ARMA(p,q) fitting
  - ARIMA(p,d,q) — Box-Jenkins methodology
  - SARIMA(p,d,q)(P,D,Q,s) — full seasonal model
  - auto_arima for automatic order selection
  - ACF/PACF guided identification
  - Residual diagnostics
  - Forecast with prediction intervals
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pmdarima as pm

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN = "#2C7BB6", "#D7191C", "#1A9641"


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD AND PREPARE DATA
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
train = series[:-24]
test  = series[-24:]
h = len(test)

# Log transform (stabilizes variance for multiplicative series)
log_series = np.log(series)
log_train  = np.log(train)
log_test   = np.log(test)


# ─────────────────────────────────────────────────────────────────────────────
# 2. STATIONARITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("="*55)
print("  Stationarity Analysis")
print("="*55)

def adf_kpss(s, label):
    adf_p = adfuller(s.dropna(), autolag="AIC")[1]
    kpss_p = kpss(s.dropna(), regression="c", nlags="auto")[1]
    adf_str  = "✅ Stationary" if adf_p < 0.05 else "❌ Non-stationary"
    kpss_str = "✅ Stationary" if kpss_p > 0.05 else "❌ Non-stationary"
    print(f"\n  {label}:")
    print(f"    ADF  p={adf_p:.4f}  → {adf_str}")
    print(f"    KPSS p={kpss_p:.4f}  → {kpss_str}")

adf_kpss(log_train,                          "log(Passengers)")
adf_kpss(log_train.diff().dropna(),          "log + first diff")
adf_kpss(log_train.diff().diff(12).dropna(), "log + first diff + seasonal diff(12)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ACF / PACF ON STATIONARY SERIES
# ─────────────────────────────────────────────────────────────────────────────

stationary = log_train.diff().diff(12).dropna()

fig, axes = plt.subplots(2, 1, figsize=(13, 8))
plot_acf(stationary,  lags=40, ax=axes[0], alpha=0.05)
plot_pacf(stationary, lags=40, ax=axes[1], alpha=0.05, method="ywm")
axes[0].set_title("ACF — after log + diff(1) + diff(12)")
axes[1].set_title("PACF — after log + diff(1) + diff(12)")
plt.suptitle("ACF/PACF for SARIMA Order Identification", fontweight="bold")
plt.tight_layout()
plt.savefig("01_acf_pacf_stationary.png", bbox_inches="tight")
plt.show()

print("\nReading the ACF/PACF:")
print("  ACF  : Large spike at lag 12 → SMA(1) term (Q=1)")
print("  PACF : Spike at lag 12 + decay → SAR(1) term (P=1)")
print("  Suggested starting model: SARIMA(0,1,1)(0,1,1)[12]  ← the 'airline model'")


# ─────────────────────────────────────────────────────────────────────────────
# 4. FIT AR(2), MA(2), ARMA(1,1)
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Fitting AR, MA, ARMA on stationary series ---")

ar2  = ARIMA(stationary, order=(2, 0, 0)).fit()
ma2  = ARIMA(stationary, order=(0, 0, 2)).fit()
arma = ARIMA(stationary, order=(1, 0, 1)).fit()

for name, m in [("AR(2)", ar2), ("MA(2)", ma2), ("ARMA(1,1)", arma)]:
    print(f"  {name}:  AIC={m.aic:.2f}  BIC={m.bic:.2f}")
    print(f"    Coefficients: {dict(zip(m.param_names, m.params.round(4)))}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. FIT ARIMA(p,d,q) — BOX-JENKINS
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- ARIMA Grid Search (log-scale, d=1) ---")

grid_results = []
for p in range(3):
    for q in range(3):
        try:
            m = ARIMA(log_train, order=(p, 1, q), trend="c").fit()
            grid_results.append({"p": p, "q": q, "AIC": m.aic, "BIC": m.bic})
        except Exception:
            pass

df_grid = pd.DataFrame(grid_results).sort_values("AIC")
print(df_grid.head(5).to_string(index=False))
best_p = df_grid.iloc[0]["p"]
best_q = df_grid.iloc[0]["q"]
print(f"\nBest ARIMA: ({int(best_p)}, 1, {int(best_q)})")

arima_fit = ARIMA(log_train, order=(int(best_p), 1, int(best_q)), trend="c").fit()
arima_fc_log  = arima_fit.get_forecast(steps=h).predicted_mean
arima_fc      = np.exp(arima_fc_log)
arima_ci_log  = arima_fit.get_forecast(steps=h).conf_int(alpha=0.05)
arima_ci_lo   = np.exp(arima_ci_log.iloc[:, 0])
arima_ci_hi   = np.exp(arima_ci_log.iloc[:, 1])


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIT SARIMA
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- SARIMA(0,1,1)(0,1,1)[12] — Airline Model ---")

sarima_fit = SARIMAX(
    log_train,
    order=(0, 1, 1),
    seasonal_order=(0, 1, 1, 12),
    trend="c"
).fit(disp=False, maxiter=200)

print(sarima_fit.summary())

sarima_fc_log = sarima_fit.get_forecast(steps=h).predicted_mean
sarima_fc     = np.exp(sarima_fc_log)
sarima_ci_log = sarima_fit.get_forecast(steps=h).conf_int(alpha=0.05)
sarima_ci_lo  = np.exp(sarima_ci_log.iloc[:, 0])
sarima_ci_hi  = np.exp(sarima_ci_log.iloc[:, 1])

# Residual diagnostics
lb = acorr_ljungbox(sarima_fit.resid, lags=[10, 20], return_df=True, model_df=2)
print("\nLjung-Box on SARIMA residuals:")
print(lb.to_string())
print("All p > 0.05?", (lb["lb_pvalue"] > 0.05).all())


# ─────────────────────────────────────────────────────────────────────────────
# 7. AUTO-ARIMA
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- auto_arima ---")
auto = pm.auto_arima(
    log_train, seasonal=True, m=12,
    start_p=0, max_p=3, start_q=0, max_q=3,
    start_P=0, max_P=2, start_Q=0, max_Q=2,
    d=None, D=None,
    information_criterion="aic",
    stepwise=True, trace=True,
    error_action="ignore", suppress_warnings=True,
)
print(f"\nBest model: ARIMA{auto.order} × {auto.seasonal_order}")
print(f"AIC: {auto.aic():.2f}")
auto_fc_log = pd.Series(auto.predict(n_periods=h), index=test.index)
auto_fc     = np.exp(auto_fc_log)


# ─────────────────────────────────────────────────────────────────────────────
# 8. COMPARISON PLOT
# ─────────────────────────────────────────────────────────────────────────────

def rmse(act, pred):
    return np.sqrt(((act.values - pred.values[:len(act)])**2).mean())

print("\n--- Forecast Comparison ---")
for name, fc in [("ARIMA", arima_fc), ("SARIMA", sarima_fc), ("Auto-ARIMA", auto_fc)]:
    print(f"  {name}: RMSE={rmse(test, fc):.2f}")

fig, axes = plt.subplots(2, 1, figsize=(13, 10))

# ARIMA vs SARIMA
axes[0].plot(train[-36:], color="gray", linewidth=1.2, label="Train")
axes[0].plot(test, color="black", linewidth=2.5, label="Actual")
axes[0].plot(arima_fc,  color=BLUE, linewidth=1.8, linestyle="--", label=f"ARIMA RMSE={rmse(test,arima_fc):.0f}")
axes[0].plot(sarima_fc, color=RED,  linewidth=1.8, linestyle="--", label=f"SARIMA RMSE={rmse(test,sarima_fc):.0f}")
axes[0].fill_between(test.index, sarima_ci_lo, sarima_ci_hi, color=RED, alpha=0.12, label="SARIMA 95% CI")
axes[0].axvline(train.index[-1], color="black", linewidth=0.8, linestyle=":", alpha=0.5)
axes[0].legend(fontsize=9)
axes[0].set_title("ARIMA vs. SARIMA Forecasts (original scale)")

# Residual diagnostics for SARIMA
sarima_fit.plot_diagnostics(axes=None, figsize=(13, 8))
plt.tight_layout()
plt.savefig("03_sarima_diagnostics.png", bbox_inches="tight")
plt.show()

fig2, ax = plt.subplots(figsize=(13, 5))
ax.plot(train[-36:], color="gray", linewidth=1.2, label="Train")
ax.plot(test, color="black", linewidth=2.5, label="Actual")
ax.plot(sarima_fc, color=RED, linewidth=2, linestyle="--", label="SARIMA(0,1,1)(0,1,1)[12]")
ax.fill_between(test.index, sarima_ci_lo, sarima_ci_hi, color=RED, alpha=0.15, label="95% Prediction Interval")
ax.axvline(train.index[-1], color="black", linewidth=0.8, linestyle=":", alpha=0.5)
ax.legend()
ax.set_title("SARIMA — Airline Passengers Forecast with 95% Prediction Intervals")
plt.tight_layout()
plt.savefig("02_sarima_forecast.png", bbox_inches="tight")
plt.show()

print("\n✅ ARIMA family demo complete.")
print("   SARIMA(0,1,1)(0,1,1)[12] = 'Airline Model' — classic and very effective for monthly seasonal data")
