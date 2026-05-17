# 02 — Energy Demand Forecasting

> **Module**: 13 Projects & Case Studies | **Project**: 2 of 5
> **Domain**: Energy | **Problem**: Multi-step probabilistic hierarchical forecasting
>
> Energy demand forecasting underpins grid stability, dispatch scheduling, and renewable integration. The signal has rich multi-scale seasonality (hourly/daily/weekly/yearly), holiday effects, and weather dependence — making it an ideal TS learning environment.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Data & EDA](#2-data--eda)
3. [Feature Engineering for Energy](#3-feature-engineering-for-energy)
4. [Models: SARIMA → Prophet → LightGBM → N-BEATS](#4-models-sarima--prophet--lightgbm--n-beats)
5. [Hierarchical Forecasting & Reconciliation](#5-hierarchical-forecasting--reconciliation)
6. [Probabilistic Forecasting](#6-probabilistic-forecasting)
7. [Production Design](#7-production-design)
8. [Key Lessons](#8-key-lessons)

---

## 1. Problem Definition

```
Business goal:
  Forecast hourly electricity demand at national and zone levels
  for the next 24h, 48h, and 7-day horizons.

Stakeholders:
  Grid operator:   24h ahead for dispatch scheduling
  Traders:         48h ahead for day-ahead market bids
  Planning team:   7-day ahead for maintenance scheduling

KPIs:
  MAPE < 2% at 24h horizon (industry standard for mature markets)
  95% PI coverage ≥ 93%
  Hierarchical coherence: zone forecasts sum to national forecast

Dataset options:
  - Open Power System Data (OPSD): hourly load for DE/FR/GB (free)
  - UCI Electricity Load: 370 households, 15-minute intervals
  - PJM Interconnection: 21 zones, hourly, 2001-present (US)
```

---

## 2. Data & EDA

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_energy_data(
    n_days: int = 730,
    n_zones: int = 3,
    freq: str = "H",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic energy demand data.
    Includes: daily + weekly + annual seasonality, weather effect, holidays.
    """
    np.random.seed(seed)
    dates  = pd.date_range("2022-01-01", periods=n_days*24, freq="H")
    n      = len(dates)
    t      = np.arange(n)

    results = []
    for zone in range(n_zones):
        base  = 5000 + zone * 1500   # zone base load

        # Multi-scale seasonality
        hourly   = 0.25 * np.sin(2*np.pi * t / 24 - np.pi/2)       # daily peak ~13:00
        weekly   = 0.10 * np.cos(2*np.pi * t / (24*7))              # weekend dip
        annual   = 0.20 * np.sin(2*np.pi * t / (24*365) + np.pi)    # winter peak

        # Weather (temperature-driven): correlated across zones
        temp_base = 15 * np.sin(2*np.pi * t / (24*365) + np.pi) + np.random.normal(0, 3, n)
        weather_effect = -0.008 * (temp_base - 18)**2   # U-shaped: peak demand at cold or hot

        # Holiday effect (approx bank holidays — 0 on certain dates)
        is_holiday = np.zeros(n)
        for h_date in pd.bdate_range("2022-01-01", "2023-12-31", freq="C",
                                      holidays=["2022-12-25","2023-12-25","2022-01-01","2023-01-01"]):
            mask = (dates.date == h_date.date())
            is_holiday[mask] = 1
        holiday_effect = -0.15 * is_holiday

        demand = base * (1 + hourly + weekly + annual + weather_effect + holiday_effect)
        noise  = np.random.normal(0, base * 0.02, n)

        results.append(pd.DataFrame({
            "timestamp": dates,
            "zone":      f"Zone_{zone+1}",
            "demand":    np.maximum(demand + noise, 0),
            "temperature": temp_base,
        }))

    df = pd.concat(results, ignore_index=True)
    # National total
    national = df.groupby("timestamp")["demand"].sum().reset_index()
    national["zone"] = "National"; national["temperature"] = np.nan
    df = pd.concat([df, national], ignore_index=True)
    return df


def energy_eda(df: pd.DataFrame, zone: str = "National") -> None:
    """EDA for one zone's demand series."""
    sub   = df[df["zone"] == zone].set_index("timestamp")["demand"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Full time series
    axes[0,0].plot(sub.index, sub.values, color="#2196F3", linewidth=0.5, alpha=0.8)
    axes[0,0].set_title(f"{zone} Hourly Demand (full period)"); axes[0,0].grid(alpha=0.3)

    # Average hourly profile
    hourly_avg = sub.groupby(sub.index.hour).mean()
    axes[0,1].bar(hourly_avg.index, hourly_avg.values, color="#4CAF50")
    axes[0,1].set_title("Average Hourly Profile"); axes[0,1].set_xlabel("Hour of Day")
    axes[0,1].grid(alpha=0.3, axis="y")

    # Weekly profile
    weekly_avg = sub.groupby(sub.index.dayofweek).mean()
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    axes[0,2].bar(days, weekly_avg.values, color="#FF9800")
    axes[0,2].set_title("Average Weekly Profile"); axes[0,2].grid(alpha=0.3, axis="y")

    # Monthly seasonality
    monthly = sub.resample("ME").mean()
    axes[1,0].plot(monthly.index, monthly.values, "o-", color="#9C27B0", linewidth=2)
    axes[1,0].set_title("Monthly Average (annual seasonality)"); axes[1,0].grid(alpha=0.3)

    # Decomposition (STL)
    from statsmodels.tsa.seasonal import STL
    daily = sub.resample("D").mean().dropna()
    stl   = STL(daily, period=7, robust=True).fit()
    axes[1,1].plot(daily.index, stl.trend, color="#F44336", linewidth=1.5, label="Trend")
    axes[1,1].set_title("STL Trend Component (daily)"); axes[1,1].grid(alpha=0.3)

    # ACF
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(sub.iloc[:168*4], lags=48, ax=axes[1,2],  # 4 weeks
             title="ACF (lags 0-48h)", alpha=0.05)

    plt.suptitle(f"Energy Demand EDA — {zone}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
```

---

## 3. Feature Engineering for Energy

```python
def energy_features(df: pd.DataFrame, zone: str = "National") -> pd.DataFrame:
    """
    Rich feature set for energy demand forecasting.
    All features are backward-looking and available at prediction time.
    """
    sub  = df[df["zone"] == zone].set_index("timestamp").sort_index()
    demand = sub["demand"].astype(float)
    feat = pd.DataFrame(index=sub.index)

    # Target: next-step demand (h=1)
    feat["target"] = demand.shift(-1)

    # Lag features (stationary — auto-correlated)
    for lag in [1, 2, 3, 24, 25, 26, 48, 168]:
        feat[f"lag_{lag}"] = demand.shift(lag)

    # Rolling statistics (shifted to avoid leakage)
    for w in [24, 48, 168]:
        rol = demand.shift(1).rolling(w, min_periods=w//4)
        feat[f"roll_mean_{w}h"] = rol.mean()
        feat[f"roll_std_{w}h"]  = rol.std()

    # Calendar features (fully available at prediction time)
    idx = sub.index
    feat["hour"]        = idx.hour.astype(float)
    feat["day_of_week"] = idx.dayofweek.astype(float)
    feat["month"]       = idx.month.astype(float)
    feat["is_weekend"]  = (idx.dayofweek >= 5).astype(float)
    feat["is_monday"]   = (idx.dayofweek == 0).astype(float)

    # Fourier features for annual seasonality
    days_in_year = 365.25 * 24
    for k in [1, 2, 3]:
        feat[f"sin_annual_{k}"] = np.sin(2*np.pi*k * np.arange(len(feat)) / days_in_year)
        feat[f"cos_annual_{k}"] = np.cos(2*np.pi*k * np.arange(len(feat)) / days_in_year)

    # Fourier features for daily cycle
    for k in [1, 2, 3]:
        feat[f"sin_daily_{k}"] = np.sin(2*np.pi*k * idx.hour / 24)
        feat[f"cos_daily_{k}"] = np.cos(2*np.pi*k * idx.hour / 24)

    # Temperature (if available)
    if "temperature" in sub.columns:
        feat["temp"]          = sub["temperature"].values
        feat["temp_sq"]       = sub["temperature"].values**2   # U-shaped effect

    return feat.dropna()
```

---

## 4. Models: SARIMA → Prophet → LightGBM → N-BEATS

```python
def run_sarima(series: pd.Series, test_size: int = 168) -> dict:
    """SARIMA for hourly data (24-period seasonality)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_absolute_percentage_error as mape_fn

    train = series[:-test_size]; test = series[-test_size:]
    model = SARIMAX(train, order=(2,1,2), seasonal_order=(1,0,1,24),
                    enforce_stationarity=False)
    res   = model.fit(disp=False)
    preds = res.forecast(steps=test_size)
    mape_ = float(mape_fn(test, preds) * 100)
    print(f"SARIMA(2,1,2)(1,0,1,24): MAPE={mape_:.2f}%")
    return {"mape": mape_, "preds": preds, "model": res}


def run_prophet_energy(series: pd.Series, test_size: int = 168) -> dict:
    """Prophet with multiple seasonality for energy data."""
    try:
        from prophet import Prophet
        df_p = series.reset_index()
        df_p.columns = ["ds", "y"]
        train = df_p.iloc[:-test_size]; test = df_p.iloc[-test_size:]

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05,
        )
        m.fit(train)
        fc     = m.predict(test[["ds"]])
        preds  = fc["yhat"].values
        actual = test["y"].values
        from sklearn.metrics import mean_absolute_percentage_error as mape_fn
        mape_  = float(mape_fn(actual, preds) * 100)
        print(f"Prophet: MAPE={mape_:.2f}%")
        return {"mape": mape_, "preds": preds}
    except ImportError:
        print("Prophet not installed: pip install prophet")
        return {}
```

---

## 5. Hierarchical Forecasting & Reconciliation

```python
def minT_reconciliation(
    Y_hat: np.ndarray,
    S: np.ndarray,
    residuals: np.ndarray,
) -> np.ndarray:
    """
    Minimum Trace (MinT) hierarchical reconciliation.

    Reconciles base forecasts so that all levels are coherent:
      zone_1 + zone_2 + zone_3 = national

    Parameters
    ----------
    Y_hat     : (n_series, H) base forecasts (rows: national, then zones)
    S         : (n_series, n_bottom) summing matrix
    residuals : (n_series, T) in-sample residuals for covariance estimation

    Returns
    -------
    Y_tilde : (n_series, H) reconciled forecasts
    """
    # Estimate residual covariance (shrinkage estimator)
    T, n  = residuals.T.shape
    W     = np.cov(residuals) + 1e-6 * np.eye(n)

    # MinT formula: Ỹ = S(S'W⁻¹S)⁻¹ S'W⁻¹ Ŷ
    W_inv = np.linalg.inv(W)
    SWS   = S.T @ W_inv @ S
    SWS_inv = np.linalg.inv(SWS + 1e-8 * np.eye(SWS.shape[0]))
    P     = SWS_inv @ S.T @ W_inv
    return (S @ P @ Y_hat).astype(float)


def build_summing_matrix(n_zones: int) -> np.ndarray:
    """
    Summing matrix S for 2-level hierarchy: national → zones.
    S maps bottom-level (zones) to all levels (national + zones).

    Returns S of shape (n_zones + 1, n_zones).
    """
    # Row 0: national = sum of all zones
    top = np.ones((1, n_zones))
    # Rows 1..n: identity (each zone is itself)
    bot = np.eye(n_zones)
    return np.vstack([top, bot])
```

---

## 6. Probabilistic Forecasting

```python
def quantile_lgbm_energy(
    feat_df: pd.DataFrame,
    quantiles: list = None,
    n_splits: int = 5,
    test_size: int = 168,
) -> dict:
    """
    Quantile regression with LightGBM for probabilistic energy forecasts.
    Trains one model per quantile (or uses multi-output).
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not available"); return {}

    quantiles = quantiles or [0.025, 0.1, 0.5, 0.9, 0.975]
    feature_cols = [c for c in feat_df.columns if c != "target"]
    X = feat_df[feature_cols].values
    y = feat_df["target"].values
    n = len(X)

    models = {}
    for q in quantiles:
        params = {
            "objective": "quantile", "alpha": q,
            "learning_rate": 0.05, "n_estimators": 300,
            "max_depth": 5, "num_leaves": 31,
            "verbosity": -1, "random_state": 42,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X[:n - test_size], y[:n - test_size])
        models[q] = model

    X_test = X[-test_size:]
    y_test = y[-test_size:]
    preds  = {q: m.predict(X_test) for q, m in models.items()}

    # Coverage
    coverage_90 = float(((y_test >= preds[0.05]) & (y_test <= preds[0.95])).mean()
                         if 0.05 in preds and 0.95 in preds else
                         ((y_test >= preds[0.025]) & (y_test <= preds[0.975])).mean())

    median_mape = float(abs((y_test - preds[0.5]) / (abs(y_test) + 1e-8)).mean() * 100)

    print(f"Quantile LightGBM: Median MAPE={median_mape:.2f}%, 95% Coverage={100*coverage_90:.1f}%")
    return {"models": models, "preds": preds, "y_test": y_test,
            "median_mape": median_mape, "coverage_95": coverage_90}
```

---

## 7. Production Design

```
Production pipeline for 24h-ahead energy forecasting:

  06:00 daily: Trigger retraining job
    → Pull last 2 years of demand data
    → Pull weather forecast for next 7 days (NWP provider)
    → Run walk-forward validation → compare to incumbent
    → If MAE improves > 2% → promote new model
    
  00:00, 06:00, 12:00, 18:00: Run inference
    → Fetch latest demand (actual vs. forecast residuals)
    → Compute 24h probabilistic forecast (5th, 25th, 50th, 75th, 95th percentiles)
    → Post to dashboard + send alert if upper 95th > grid capacity

  Monitoring:
    → Rolling 7-day MAPE vs. 4-week baseline
    → Feature drift: temperature distribution, holiday calendar accuracy
    → Alert if rolling MAPE > 3% (degraded model)

Serving:
    → Pre-compute 24h/48h/7-day forecasts each run
    → Store in Redis with 6h TTL
    → FastAPI endpoint for on-demand queries
```

---

## 8. Key Lessons

```
LESSON 1: Multiple seasonality is the main challenge.
  Simple ARIMA misses daily AND weekly AND annual patterns.
  → Use SARIMA (24-period) + Fourier features for annual
    OR Prophet's multi-seasonality framework.

LESSON 2: Temperature is the strongest exogenous predictor.
  Include it — but use forecast temperature (not actual) at prediction time.
  Treating actual temperature as a "feature" at serve time requires
  integration with an NWP weather forecast provider.

LESSON 3: Hierarchy must be enforced explicitly.
  Unconstrained zone forecasts rarely sum to the national forecast.
  MinT reconciliation guarantees coherence across hierarchy levels.

LESSON 4: Calibrated intervals are critical for grid operators.
  An 80% PI that only covers 60% of actuals will cause capacity crises.
  Always report empirical coverage alongside forecast intervals.

LESSON 5: Weekends and holidays need special treatment.
  Binary is_weekend and is_holiday features matter more than
  any sophisticated model architecture change.

LESSON 6: MAPE is misleading at near-zero demand (overnight).
  Use MAE or RMSE at low-demand hours;
  MAPE only for reporting to non-technical stakeholders.
```

---

*← [01 — Stock](./01_stock_price_forecasting.md) | [Module README](./README.md) | Next: [03 — Retail](./03_retail_sales_forecasting.md) →*
