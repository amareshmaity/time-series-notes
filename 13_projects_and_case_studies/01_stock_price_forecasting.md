# 01 — Stock Price Forecasting

> **Module**: 13 Projects & Case Studies | **Project**: 1 of 5
> **Domain**: Finance | **Problem**: Forecasting + Uncertainty Quantification
>
> Stock prices are among the hardest signals to forecast — they are near-random-walks, subject to sudden regime changes, and driven by unobservable information. This project demonstrates a realistic, honest pipeline: from raw data → EDA → multiple models → proper backtesting → prediction intervals.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Data Collection & EDA](#2-data-collection--eda)
3. [Preprocessing & Feature Engineering](#3-preprocessing--feature-engineering)
4. [Baseline Models](#4-baseline-models)
5. [Advanced Models](#5-advanced-models)
6. [Backtesting & Model Comparison](#6-backtesting--model-comparison)
7. [Prediction Intervals](#7-prediction-intervals)
8. [Key Lessons](#8-key-lessons)

---

## 1. Problem Definition

### 1.1 Business Context

```
Goal: Forecast daily closing price of a stock over next H = 5, 10, 21 trading days.

Stakeholders:
  Portfolio manager: needs directional forecast + downside risk estimate.
  Risk team:         needs 95% prediction interval.
  Trading desk:      needs point forecast + confidence for position sizing.

KPIs:
  Primary:   RMSE on walk-forward test set (5 folds × 21-day windows)
  Secondary: Prediction interval coverage rate (should be ~95%)
  Business:  Sharpe ratio of a simple forecast-based strategy

Key constraints:
  ✓ No future data leakage (strict walk-forward)
  ✓ Prediction intervals required (uncertainty matters)
  ✓ Model must be re-fit daily in < 2 minutes
  ✓ Baseline: random walk (tomorrow = today)
```

### 1.2 Why Stock Forecasting Is Hard

```
EFFICIENT MARKET HYPOTHESIS (EMH):
  In a semi-strong efficient market, prices reflect all public information.
  → No systematic profit from public data alone.
  → Does NOT mean prices are unpredictable (short-horizon predictability exists).

What IS predictable:
  ✓ Volatility clustering (GARCH — large moves tend to cluster)
  ✓ Short-term momentum (1-4 week horizon)
  ✓ Mean reversion at very short horizons (microstructure)
  ✓ Seasonal patterns in returns (turn-of-month, January effect)

What is NOT reliably predictable:
  ✗ Long-horizon direction (>3 months, near-random)
  ✗ Exact timing of trend reversals
  ✗ Impact of unobservable events (news, earnings surprises)

Honest expectation:
  A well-calibrated forecast with ±5-10% MAPE at 1-5 day horizon
  is the realistic upper bound for most public-data-only models.
```

---

## 2. Data Collection & EDA

### 2.1 Data Pipeline

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Load OHLCV data from Yahoo Finance.
    pip install yfinance
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        df.columns = [c.lower() for c in df.columns]
        df.index   = pd.to_datetime(df.index)
        df.dropna(inplace=True)
        print(f"Loaded {ticker}: {len(df)} trading days, "
              f"{df.index[0].date()} → {df.index[-1].date()}")
        return df
    except ImportError:
        # Fallback: generate synthetic OHLCV
        print("yfinance not installed — generating synthetic stock data")
        return _synthetic_stock(start, end, ticker)


def _synthetic_stock(start: str, end: str, seed: int = 42) -> pd.DataFrame:
    """Geometric Brownian Motion stock price simulator."""
    np.random.seed(42)
    dates  = pd.bdate_range(start=start, end=end)
    n      = len(dates)
    mu     = 0.0003      # daily drift
    sigma  = 0.015       # daily volatility
    S0     = 150.0
    
    returns = np.random.normal(mu, sigma, n)
    prices  = S0 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "open":   prices * np.random.uniform(0.995, 1.005, n),
        "high":   prices * np.random.uniform(1.001, 1.015, n),
        "low":    prices * np.random.uniform(0.985, 0.999, n),
        "close":  prices,
        "volume": np.random.randint(5_000_000, 20_000_000, n),
    }, index=dates)
    return df


def stock_eda(df: pd.DataFrame, ticker: str = "TICKER") -> dict:
    """
    Comprehensive EDA for stock time series.
    Returns summary statistics relevant for model selection.
    """
    from statsmodels.tsa.stattools import adfuller, acf

    close = df["close"]
    ret   = close.pct_change().dropna()

    # Stationarity
    adf_price   = adfuller(close)[1]
    adf_returns = adfuller(ret)[1]

    # Return distribution
    mean_ret = float(ret.mean())
    std_ret  = float(ret.std())
    skew     = float(ret.skew())
    kurt     = float(ret.kurtosis())

    # Autocorrelation of returns and |returns|
    acf_ret  = acf(ret, nlags=10, fft=True)[1:]
    acf_abs  = acf(ret.abs(), nlags=10, fft=True)[1:]

    summary = {
        "n_observations": len(df),
        "price_range":    (round(float(close.min()), 2), round(float(close.max()), 2)),
        "adf_price_pval": round(adf_price, 5),
        "adf_return_pval":round(adf_returns, 5),
        "daily_return_mean_pct": round(mean_ret * 100, 4),
        "daily_return_std_pct":  round(std_ret * 100, 4),
        "skewness":   round(skew, 4),
        "excess_kurtosis": round(kurt, 4),  # > 0 = fat tails (common in stocks)
        "max_drawdown_pct": round(float(_max_drawdown(close)) * 100, 2),
        "acf_ret_lag1": round(float(acf_ret[0]), 4),
        "acf_abs_lag1": round(float(acf_abs[0]), 4),  # volatility clustering
    }

    print(f"\n=== {ticker} EDA ===")
    for k, v in summary.items():
        print(f"  {k:30s}: {v}")

    # Key observations for model selection
    print("\nKey observations:")
    if summary["adf_price_pval"] > 0.05:
        print("  ✓ Price is NON-stationary (I(1)) → use log-returns for modeling")
    if abs(summary["acf_abs_lag1"]) > 0.1:
        print("  ✓ Volatility clustering detected → GARCH may improve intervals")
    if abs(summary["acf_ret_lag1"]) > 0.05:
        print("  ✓ Return autocorrelation → momentum features useful")

    return summary


def _max_drawdown(prices: pd.Series) -> float:
    peak  = prices.cummax()
    dd    = (prices - peak) / peak
    return float(dd.min())
```

### 2.2 Key EDA Plots

```python
def plot_stock_eda(df: pd.DataFrame, ticker: str = "TICKER") -> None:
    close = df["close"]
    ret   = close.pct_change().dropna()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Price
    axes[0,0].plot(close.index, close, color="#2196F3", linewidth=1)
    axes[0,0].set_title(f"{ticker} Closing Price"); axes[0,0].grid(alpha=0.3)

    # Log returns
    log_ret = np.log(close).diff().dropna()
    axes[0,1].plot(log_ret.index, log_ret, color="#FF5722", linewidth=0.7, alpha=0.8)
    axes[0,1].set_title("Log Returns"); axes[0,1].grid(alpha=0.3)

    # Return distribution
    axes[0,2].hist(ret, bins=60, color="#4CAF50", edgecolor="white", alpha=0.8, density=True)
    x = np.linspace(ret.min(), ret.max(), 200)
    from scipy.stats import norm
    axes[0,2].plot(x, norm.pdf(x, ret.mean(), ret.std()), "r-", linewidth=2, label="Normal")
    axes[0,2].set_title("Return Distribution vs. Normal"); axes[0,2].legend(); axes[0,2].grid(alpha=0.3)

    # Rolling volatility (21-day)
    vol = ret.rolling(21).std() * np.sqrt(252)
    axes[1,0].plot(vol.index, vol, color="#9C27B0", linewidth=1)
    axes[1,0].set_title("Rolling 21d Annualized Volatility"); axes[1,0].grid(alpha=0.3)

    # ACF of returns
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    plot_acf(ret, lags=20, ax=axes[1,1], title="ACF of Returns", alpha=0.05)

    # ACF of |returns| (volatility clustering)
    plot_acf(ret.abs(), lags=20, ax=axes[1,2], title="ACF of |Returns| (Vol Clustering)", alpha=0.05)

    plt.suptitle(f"{ticker} — Stock EDA", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{ticker.lower()}_eda.png", dpi=150, bbox_inches="tight")
    plt.show()
```

---

## 3. Preprocessing & Feature Engineering

```python
def engineer_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature set for stock price forecasting.
    All features are strictly backward-looking (no leakage).

    Features:
      Price lags, return lags, moving averages, RSI, Bollinger Bands,
      MACD, volume features, calendar features.
    """
    c = df["close"].copy()
    v = df["volume"].copy()
    feat = pd.DataFrame(index=df.index)

    # Target: next-day log return
    feat["target"] = np.log(c).diff().shift(-1)   # ← shift(-1): tomorrow's return

    # Price lags (in log-return space — stationary)
    log_ret = np.log(c).diff()
    for lag in [1, 2, 3, 5, 10, 21]:
        feat[f"ret_lag_{lag}"] = log_ret.shift(lag)

    # Trend features: price relative to moving averages
    for w in [5, 10, 21, 63]:
        ma = c.rolling(w).mean()
        feat[f"price_vs_ma{w}"] = (c / ma - 1)   # above/below MA

    # Volatility features (realized vol)
    for w in [5, 10, 21]:
        feat[f"realized_vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

    # RSI (14-period)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    feat["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-12)))

    # Bollinger Band position
    ma20   = c.rolling(20).mean()
    std20  = c.rolling(20).std()
    feat["bb_position"] = (c - ma20) / (2 * std20 + 1e-12)

    # MACD (12-26-9)
    ema12  = c.ewm(span=12).mean()
    ema26  = c.ewm(span=26).mean()
    macd   = ema12 - ema26
    feat["macd"] = macd
    feat["macd_signal"] = macd.ewm(span=9).mean()
    feat["macd_hist"]   = macd - feat["macd_signal"]

    # Volume features
    feat["volume_vs_ma20"] = v / (v.rolling(20).mean() + 1e-12)

    # Calendar features
    feat["day_of_week"] = df.index.dayofweek.astype(float)
    feat["month"]       = df.index.month.astype(float)

    return feat.dropna()
```

---

## 4. Baseline Models

```python
def random_walk_baseline(close: pd.Series, horizon: int = 5) -> dict:
    """
    Random walk: tomorrow = today.
    This is the single most important benchmark for stock forecasting.
    If your model can't beat this → it has no predictive value.
    """
    from sklearn.metrics import mean_absolute_error
    n      = len(close)
    splits = walk_forward_splits(n, n_splits=5, test_size=21)
    maes   = []
    for tr_end, te_start, te_end in splits:
        y_true = close.iloc[te_start:te_end].values[1:]   # t+1
        y_pred = close.iloc[te_start:te_end-1].values     # t (random walk)
        maes.append(mean_absolute_error(y_true, y_pred))
    print(f"Random Walk baseline: MAE = {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    return {"mae": np.mean(maes), "std": np.std(maes)}


def walk_forward_splits(n, n_splits=5, test_size=21, gap=1):
    step = (n - test_size * n_splits) // (n_splits + 1)
    splits = []
    for i in range(n_splits):
        tr_end   = step + i * step + step
        te_start = tr_end + gap
        te_end   = te_start + test_size
        if te_end <= n:
            splits.append((tr_end, te_start, te_end))
    return splits
```

---

## 5. Advanced Models

### 5.1 Model Comparison Plan

```
Tier 1 — Baselines (must beat):
  ✓ Random Walk: y_{t+1} = y_t
  ✓ Historical Mean: y_{t+h} = mean(y_{t-252:t})

Tier 2 — Statistical:
  ✓ ARIMA(p,1,q): fit on log-returns, select via AIC
  ✓ GARCH(1,1):   volatility model for prediction intervals

Tier 3 — ML:
  ✓ LightGBM:     rich feature set, walk-forward CV
  ✓ Ridge Regression: simpler, less overfitting risk

Tier 4 — DL / Foundation:
  ✓ Chronos (zero-shot): Amazon foundation model
  ✓ TFT (if sufficient data ≥ 3 years)

Model selection criterion:
  Primary:   lowest RMSE on 5-fold walk-forward test
  Tiebreaker: 95% PI coverage rate
```

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_lgbm_stock(feat_df: pd.DataFrame) -> dict:
    """Walk-forward LightGBM with time series features."""
    try:
        import lightgbm as lgb
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor as lgb_fallback
        print("Using sklearn GBM fallback")
        return {}

    feature_cols = [c for c in feat_df.columns if c != "target"]
    X = feat_df[feature_cols].values
    y = feat_df["target"].values
    n = len(X)
    splits = walk_forward_splits(n, n_splits=5, test_size=21)

    all_preds, all_true = [], []
    for tr_end, te_start, te_end in splits:
        X_tr, y_tr = X[:tr_end], y[:tr_end]
        X_te, y_te = X[te_start:te_end], y[te_start:te_end]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                               num_leaves=31, random_state=42, verbose=-1)
        m.fit(X_tr_s, y_tr)
        all_preds.extend(m.predict(X_te_s)); all_true.extend(y_te)

    mae  = mean_absolute_error(all_true, all_preds)
    rmse = float(np.sqrt(((np.array(all_true) - np.array(all_preds))**2).mean()))
    print(f"LightGBM stock: MAE={mae:.6f}, RMSE={rmse:.6f}")
    return {"mae": mae, "rmse": rmse}
```

---

## 6. Backtesting & Model Comparison

```python
def stock_backtest_comparison(
    df: pd.DataFrame,
    feat_df: pd.DataFrame,
    ticker: str = "TICKER",
) -> pd.DataFrame:
    """
    Run all models with identical walk-forward splits.
    Returns comparison DataFrame.
    """
    from sklearn.metrics import mean_absolute_error

    close = df["close"]
    feature_cols = [c for c in feat_df.columns if c != "target"]
    X = feat_df[feature_cols].values
    y = feat_df["target"].values   # log returns
    n = len(X)

    splits  = walk_forward_splits(n, n_splits=5, test_size=21)
    results = {}

    # Model 1: Random Walk
    rw_maes = []
    for tr_end, te_start, te_end in splits:
        y_true = y[te_start:te_end]
        y_pred = np.zeros_like(y_true)   # log return = 0 → price unchanged
        rw_maes.append(mean_absolute_error(y_true, y_pred))
    results["Random Walk"] = {"MAE": np.mean(rw_maes), "std": np.std(rw_maes)}

    # Model 2: Ridge Regression
    ridge_maes = []
    for tr_end, te_start, te_end in splits:
        X_tr, y_tr = X[:tr_end], y[:tr_end]
        X_te, y_te = X[te_start:te_end], y[te_start:te_end]
        pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
        pipe.fit(X_tr, y_tr)
        ridge_maes.append(mean_absolute_error(y_te, pipe.predict(X_te)))
    results["Ridge"] = {"MAE": np.mean(ridge_maes), "std": np.std(ridge_maes)}

    # Print table
    print(f"\n{'='*55}\n{ticker} Backtest Results (5-fold walk-forward)\n{'='*55}")
    print(f"{'Model':<20} {'MAE':>12} {'± Std':>10}")
    print("-"*45)
    for name, r in sorted(results.items(), key=lambda x: x[1]["MAE"]):
        print(f"  {name:<18} {r['MAE']:>12.6f} {r['std']:>10.6f}")

    return pd.DataFrame({k: {"MAE": v["MAE"], "Std": v["std"]} for k,v in results.items()}).T
```

---

## 7. Prediction Intervals

```python
def bootstrap_prediction_interval(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
) -> dict:
    """
    Residual bootstrap prediction intervals.

    Algorithm:
      1. Fit model on training data
      2. Compute training residuals
      3. For each test point, sample residuals and add to point forecast
      4. Take α/2 and 1-α/2 quantiles as interval bounds
    """
    model.fit(X_train, y_train)
    residuals = y_train - model.predict(X_train)
    y_hat     = model.predict(X_test)

    boot_preds = np.array([
        y_hat + np.random.choice(residuals, size=len(X_test), replace=True)
        for _ in range(n_bootstrap)
    ])

    lower = np.percentile(boot_preds, 100 * alpha/2, axis=0)
    upper = np.percentile(boot_preds, 100 * (1-alpha/2), axis=0)

    return {"point": y_hat, "lower": lower, "upper": upper, "alpha": alpha}


def compute_coverage(y_true, lower, upper) -> float:
    """Fraction of true values that fall within the prediction interval."""
    return float(((y_true >= lower) & (y_true <= upper)).mean())
```

---

## 8. Key Lessons

```
LESSON 1: Random walk is hard to beat for stock prices.
  Most ML models perform similarly to or worse than naive forecasts
  on a proper walk-forward backtest. The notebook accuracy is not the metric.

LESSON 2: Predicting volatility is easier than predicting direction.
  GARCH models for volatility → well-calibrated prediction intervals.
  Direction prediction → near-random for most models.

LESSON 3: Feature engineering for returns, not prices.
  Always model log-returns (stationary) — not raw prices.
  Transform back to price-level for reporting.

LESSON 4: Multiple horizons behave differently.
  1-day ahead: high autocorrelation, momentum features help.
  21-day ahead: near-random walk, model uncertainty dominates.

LESSON 5: Walk-forward backtesting is non-negotiable.
  Any split that uses future data for training is invalid.
  Report results across multiple folds, not just one split.

LESSON 6: Communicate uncertainty honestly.
  95% prediction intervals covering 80% of actuals is a
  poorly-calibrated model — report coverage explicitly.
```

---

*← [Module README](./README.md) | Next: [02 — Energy Demand](./02_energy_demand_forecasting.md) →*
