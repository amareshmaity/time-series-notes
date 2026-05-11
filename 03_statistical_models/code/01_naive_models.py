"""
01_naive_models.py
==================
Module 03 — Statistical Models
Topic   : Naive Baseline Models

Covers:
  - Mean forecast
  - Naive forecast (random walk)
  - Drift forecast
  - Seasonal naive forecast
  - Comparison with metrics and visualization
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE, PURPLE = "#2C7BB6", "#D7191C", "#1A9641", "#F07D00", "#7B2D8B"


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_airline():
    try:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        s = df.squeeze()
        s.index.freq = "MS"
    except Exception:
        s = sns.load_dataset("flights")
        s["date"] = pd.to_datetime(s["year"].astype(str) + "-" + s["month"].astype(str), format="%Y-%B")
        s = s.set_index("date")["passengers"].sort_index()
        s.index.freq = "MS"
    s.name = "Passengers"
    return s

series = load_airline()
train = series[:-24]   # hold out last 24 months for testing
test  = series[-24:]

print(f"Series: {len(series)} monthly observations")
print(f"Train: {len(train)} | Test: {len(test)}")
print(f"Test period: {test.index[0].date()} → {test.index[-1].date()}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. IMPLEMENT BASELINE MODELS
# ─────────────────────────────────────────────────────────────────────────────

def make_forecast_index(train, h):
    return pd.date_range(start=train.index[-1] + train.index.freq, periods=h, freq=train.index.freq)

h = len(test)
fc_idx = make_forecast_index(train, h)

# Mean forecast
fc_mean = pd.Series(train.mean(), index=fc_idx, name="Mean")

# Naive forecast
fc_naive = pd.Series(train.iloc[-1], index=fc_idx, name="Naive")

# Drift forecast
drift = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
fc_drift = pd.Series([train.iloc[-1] + i * drift for i in range(1, h + 1)], index=fc_idx, name="Drift")

# Seasonal naive (m=12 for monthly)
m = 12
last_season = train.iloc[-m:].values
fc_snaive = pd.Series([last_season[i % m] for i in range(h)], index=fc_idx, name="Seasonal Naive")


# ─────────────────────────────────────────────────────────────────────────────
# 3. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def metrics(actual, predicted, name):
    e = actual.values - predicted.values[:len(actual)]
    return {"Model": name,
            "MAE":  round(np.abs(e).mean(), 2),
            "RMSE": round(np.sqrt((e**2).mean()), 2),
            "MAPE": round((np.abs(e) / np.abs(actual.values)).mean() * 100, 2)}

results = pd.DataFrame([
    metrics(test, fc_mean,   "Mean"),
    metrics(test, fc_naive,  "Naive"),
    metrics(test, fc_drift,  "Drift"),
    metrics(test, fc_snaive, "Seasonal Naive"),
]).set_index("Model").sort_values("RMSE")

print("\nBaseline Leaderboard:")
print(results.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(train[-36:], color="gray", linewidth=1.2, label="Train (last 3 years)")
ax.plot(test, color="black", linewidth=2.5, label="Actual Test")

for fcst, color in [(fc_mean, BLUE), (fc_naive, RED), (fc_drift, GREEN), (fc_snaive, ORANGE)]:
    ax.plot(fcst, color=color, linewidth=1.8, linestyle="--", label=fcst.name)

ax.axvline(train.index[-1], color="black", linewidth=1, linestyle=":", alpha=0.5)
ax.legend(fontsize=9)
ax.set_title("Baseline Forecasts vs. Actual (Airline Passengers)")
ax.set_ylabel("Passengers (thousands)")
plt.tight_layout()
plt.savefig("01_baseline_forecasts.png", bbox_inches="tight")
plt.show()

# Bar chart of RMSE
fig, ax = plt.subplots(figsize=(8, 4))
colors_bar = [BLUE, RED, GREEN, ORANGE]
ax.barh(results.index, results["RMSE"], color=colors_bar[:len(results)], edgecolor="white")
ax.set_xlabel("RMSE")
ax.set_title("Baseline Models: RMSE Comparison")
for i, (idx, row) in enumerate(results.iterrows()):
    ax.text(row["RMSE"] + 1, i, f"{row['RMSE']:.1f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("02_baseline_rmse.png", bbox_inches="tight")
plt.show()

print("\nKey Finding:")
print(f"  Seasonal Naive RMSE: {results.loc['Seasonal Naive', 'RMSE']:.2f}")
print("  Any production model must beat this threshold to justify its complexity!")
