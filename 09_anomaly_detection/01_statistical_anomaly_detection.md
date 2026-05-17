# 01 — Statistical Anomaly Detection

> **Module**: 09 Anomaly Detection | **File**: 1 of 6
>
> Statistical methods are the fastest, most interpretable anomaly detectors. They require no training data labeling and often outperform complex models when the signal is strong. This note covers Z-score, IQR, CUSUM, Bollinger Bands, and STL residual-based detection.

---

## Table of Contents

1. [Anomaly Types in Time Series](#1-anomaly-types-in-time-series)
2. [Z-Score Detection](#2-z-score-detection)
3. [IQR and Tukey Fences](#3-iqr-and-tukey-fences)
4. [Rolling Statistics Detection](#4-rolling-statistics-detection)
5. [CUSUM — Cumulative Sum Control Chart](#5-cusum--cumulative-sum-control-chart)
6. [Bollinger Bands](#6-bollinger-bands)
7. [STL Residual-Based Detection](#7-stl-residual-based-detection)
8. [Threshold Calibration](#8-threshold-calibration)
9. [Production Implementation](#9-production-implementation)

---

## 1. Anomaly Types in Time Series

### 1.1 Taxonomy

```
Point Anomaly:      A single observation is anomalous
  │
  │  Normal:  ─────────────────────────────
  │  Anomaly: ───────────────┼────────────
  │                         spike

Contextual Anomaly: Value is normal globally, anomalous in its context
  │
  │  Example: Temperature of 35°C is normal in July (India),
  │           anomalous in January (Norway)

Collective Anomaly: A subsequence is anomalous (no single point is)
  │
  │  Normal:  ~~~~~~~~~~~~~~~~~
  │  Anomaly: ───────────────── (flatline in ECG = cardiac arrest)

Seasonal Anomaly:   Value deviates from expected seasonal pattern
  │
  │  Sales usually spike at Christmas; not doing so = anomaly
```

### 1.2 Detection Approaches Overview

| Approach                | Supervised | Type Detected        | Complexity |
|-------------------------|-----------|----------------------|------------|
| Z-score / IQR           | ❌         | Point                | Low        |
| Rolling statistics      | ❌         | Point, Contextual    | Low        |
| CUSUM                   | ❌         | Shift in mean        | Low        |
| Bollinger Bands         | ❌         | Point, Contextual    | Low        |
| STL residuals           | ❌         | Point, Seasonal      | Low-Medium |
| Isolation Forest        | ❌         | Point, Contextual    | Medium     |
| Autoencoder             | Semi       | All types            | High       |
| LSTM-AD                 | Semi       | Collective           | High       |

---

## 2. Z-Score Detection

### 2.1 Formula

```
z_score(xₜ) = (xₜ - μ) / σ

Where:
  μ = mean of the training/reference window
  σ = standard deviation of the training/reference window

Flag as anomaly: |z_score| > threshold (typically 3.0)
```

### 2.2 Global vs. Rolling Z-Score

```
Global Z-score:
  Uses mean and std of the ENTIRE series
  ✅ Simple, stable
  ❌ Fails with trend (entire distribution shifts)
  ❌ Misses contextual anomalies (seasonal context ignored)

Rolling Z-score:
  Uses mean and std of a sliding window of length W
  ✅ Adapts to local level and variance
  ✅ Catches contextual anomalies
  ❌ Window length W is a hyperparameter
  ❌ Edge effects at series boundaries
```

### 2.3 Implementation

```python
import numpy as np
import pandas as pd

def zscore_anomaly_detector(
    series: np.ndarray,
    window: int = None,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Z-score anomaly detector — global or rolling.

    Parameters
    ----------
    series    : 1D time series array
    window    : rolling window size (None = global z-score)
    threshold : |z| > threshold → anomaly

    Returns
    -------
    DataFrame with columns: value, z_score, anomaly
    """
    s = pd.Series(series, dtype=float)

    if window is None:
        mu    = s.mean()
        sigma = s.std()
    else:
        mu    = s.rolling(window, min_periods=window // 2, center=False).mean()
        sigma = s.rolling(window, min_periods=window // 2, center=False).std()

    z_score = (s - mu) / (sigma + 1e-12)

    return pd.DataFrame({
        "value":   s.values,
        "z_score": z_score.values,
        "anomaly": (np.abs(z_score) > threshold).values,
        "upper":   (mu + threshold * sigma).values if hasattr(mu, 'values') else float(mu + threshold * sigma),
        "lower":   (mu - threshold * sigma).values if hasattr(mu, 'values') else float(mu - threshold * sigma),
    })
```

### 2.4 Limitations

- Assumes approximately normal distribution
- Sensitive to the presence of anomalies in the reference window (they inflate σ)
- **Solution**: use a **robust Z-score** with median and MAD:

```python
def robust_zscore(series: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """
    Modified Z-score using median and Median Absolute Deviation (MAD).
    More robust than standard z-score when anomalies are present.

    Reference: Iglewicz & Hoaglin (1993)
    """
    series = np.asarray(series, dtype=float)
    median = np.median(series)
    mad    = np.median(np.abs(series - median))
    if mad == 0:
        mad = np.mean(np.abs(series - median)) + 1e-12
    modified_z = 0.6745 * (series - median) / mad
    return np.abs(modified_z) > threshold
```

---

## 3. IQR and Tukey Fences

### 3.1 Formula

```
Q1, Q3 = 25th and 75th percentiles of series
IQR    = Q3 - Q1

Tukey fences (inner):  [Q1 - 1.5·IQR,  Q3 + 1.5·IQR]
Tukey fences (outer):  [Q1 - 3.0·IQR,  Q3 + 3.0·IQR]

Inner fence: mild outliers
Outer fence: extreme outliers
```

### 3.2 Rolling IQR

```python
def iqr_anomaly_detector(
    series: np.ndarray,
    window: int = None,
    k: float = 1.5,
) -> pd.DataFrame:
    """
    IQR-based anomaly detector (Tukey fences).

    Parameters
    ----------
    window : rolling window size (None = global IQR)
    k      : fence multiplier (1.5 = mild, 3.0 = extreme)
    """
    s = pd.Series(series, dtype=float)

    if window is None:
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
    else:
        q1 = s.rolling(window, min_periods=window//2).quantile(0.25)
        q3 = s.rolling(window, min_periods=window//2).quantile(0.75)

    iqr   = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    anomaly = (s < lower) | (s > upper)

    return pd.DataFrame({
        "value":   s.values,
        "lower":   lower.values if hasattr(lower, 'values') else lower,
        "upper":   upper.values if hasattr(upper, 'values') else upper,
        "anomaly": anomaly.values,
    })
```

---

## 4. Rolling Statistics Detection

### 4.1 Rolling Mean + Std Bands

Detects anomalies as deviations from a **locally expected range**. Unlike global Z-score, this adapts to trend and seasonal level changes:

```python
def rolling_band_detector(
    series: np.ndarray,
    window: int = 30,
    n_sigma: float = 3.0,
) -> pd.DataFrame:
    """
    Rolling mean ± n_sigma·std bands — contextual anomaly detection.

    Key advantage: captures local level shifts (trend, seasonality).
    The window should span at least one full seasonal cycle.
    """
    s     = pd.Series(series, dtype=float)
    mu    = s.rolling(window, min_periods=window // 2, center=False).mean()
    sigma = s.rolling(window, min_periods=window // 2, center=False).std()

    upper   = mu + n_sigma * sigma
    lower   = mu - n_sigma * sigma
    anomaly = (s > upper) | (s < lower)

    return pd.DataFrame({
        "value":   s.values,
        "mu":      mu.values,
        "upper":   upper.values,
        "lower":   lower.values,
        "anomaly": anomaly.values,
        "score":   np.abs((s - mu) / (sigma + 1e-12)).values,
    })
```

### 4.2 Rolling Min/Max Range

Flags values that fall outside the observed historical range within the window — useful for **hard constraint** violations (e.g., sensor physically cannot exceed a value):

```python
def rolling_range_detector(
    series: np.ndarray,
    window: int = 30,
    margin: float = 0.1,
) -> np.ndarray:
    """
    Anomaly if value is outside [rolling_min - margin, rolling_max + margin].
    margin: fractional tolerance (0.1 = 10% beyond historical range is allowed).
    """
    s   = pd.Series(series, dtype=float)
    lo  = s.rolling(window, min_periods=2).min()
    hi  = s.rolling(window, min_periods=2).max()
    rng = hi - lo
    return ((s < lo - margin * rng) | (s > hi + margin * rng)).values
```

---

## 5. CUSUM — Cumulative Sum Control Chart

### 5.1 Concept

CUSUM detects **persistent shifts in the process mean**. Unlike Z-score (which reacts to a single point), CUSUM accumulates evidence over time and is ideal for detecting gradual drift:

```
Standard CUSUM:
  S₊ₜ = max(0, S₊ₜ₋₁ + (xₜ - μ₀ - k))   (detects upward shift)
  S⁻ₜ = max(0, S⁻ₜ₋₁ + (μ₀ - k - xₜ))   (detects downward shift)

Where:
  μ₀ = target (in-control) mean
  k  = allowance (typically 0.5 × shift to detect, in units of σ)
  h  = control limit (threshold — signal if S > h)

Alert when: S₊ₜ > h  or  S⁻ₜ > h
```

### 5.2 Choosing Parameters

```
k (allowance):
  k = δ/2  where δ = shift to detect (in σ units)
  To detect a 1σ shift: k = 0.5
  To detect a 2σ shift: k = 1.0

h (control limit):
  h = 4σ  → ARL₀ ≈ 168 (false alarm every ~168 points under no shift)
  h = 5σ  → ARL₀ ≈ 465

ARL₀ = Average Run Length when no anomaly present (higher = fewer false alarms)
```

### 5.3 Implementation

```python
import numpy as np
import pandas as pd

def cusum_detector(
    series: np.ndarray,
    target_mean: float = None,
    k: float = 0.5,
    h: float = 5.0,
    sigma: float = None,
) -> pd.DataFrame:
    """
    CUSUM control chart for detecting persistent mean shifts.

    Parameters
    ----------
    series      : 1D time series (standardised or raw)
    target_mean : in-control target (default: mean of first 20% of series)
    k           : allowance (0.5 = detect 1σ shift; 1.0 = detect 2σ shift)
    h           : control limit (signal threshold for S+/S-)
    sigma       : process standard deviation (default: std of first 20%)

    Returns
    -------
    DataFrame with CUSUM statistics and alert flags
    """
    series = np.asarray(series, dtype=float)
    n_ref  = max(int(0.2 * len(series)), 5)

    if target_mean is None:
        target_mean = series[:n_ref].mean()
    if sigma is None:
        sigma = series[:n_ref].std() + 1e-12

    # Normalize by sigma so k and h are in sigma units
    x_norm = (series - target_mean) / sigma

    S_pos = np.zeros(len(series))
    S_neg = np.zeros(len(series))

    for t in range(1, len(series)):
        S_pos[t] = max(0.0, S_pos[t-1] + x_norm[t] - k)
        S_neg[t] = max(0.0, S_neg[t-1] - x_norm[t] - k)

    return pd.DataFrame({
        "value":   series,
        "S_pos":   S_pos,
        "S_neg":   S_neg,
        "alert":   (S_pos > h) | (S_neg > h),
        "upper":   h,
        "lower":   -h,   # for plotting reference
    })
```

### 5.4 CUSUM vs. Z-Score Comparison

```
Scenario: Mean shifts from 10 → 12 at t=100, noise σ=1.

Z-score (threshold 3σ):
  Individual points: |z| = (12-10)/1 = 2 < 3 → misses the shift!
  Detects only extreme individual spikes

CUSUM (k=0.5, h=5):
  Accumulates evidence: detects shift by t≈108
  → Much more sensitive to persistent mean shifts
  
Use Z-score for: spike/impulse anomalies
Use CUSUM for:   drift/level shift anomalies
```

---

## 6. Bollinger Bands

### 6.1 Concept

Bollinger Bands (developed for financial markets, widely used in monitoring):

```
Middle Band: SMA(t, W) = (1/W) Σ xₜ₋ⱼ   (W-period simple moving average)
Upper Band:  SMA(t, W) + n · σ(t, W)
Lower Band:  SMA(t, W) - n · σ(t, W)

Where σ(t, W) = rolling standard deviation
Standard parameters: W=20, n=2

Signal: xₜ > Upper → high anomaly score
        xₜ < Lower → low anomaly score
```

### 6.2 Implementation

```python
def bollinger_bands(
    series: np.ndarray,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands anomaly detector.

    %B statistic: (x - lower) / (upper - lower)
      %B > 1.0 → above upper band (high anomaly)
      %B < 0.0 → below lower band (low anomaly)
      %B = 0.5 → at middle band (normal)
    """
    s   = pd.Series(series, dtype=float)
    sma = s.rolling(window, min_periods=window // 2).mean()
    std = s.rolling(window, min_periods=window // 2).std()

    upper = sma + n_std * std
    lower = sma - n_std * std
    band_width = upper - lower

    pct_b = (s - lower) / (band_width + 1e-12)

    return pd.DataFrame({
        "value":      s.values,
        "sma":        sma.values,
        "upper":      upper.values,
        "lower":      lower.values,
        "pct_b":      pct_b.values,
        "anomaly":    ((s > upper) | (s < lower)).values,
        "band_width": band_width.values,
    })
```

---

## 7. STL Residual-Based Detection

### 7.1 Why STL?

STL (Seasonal-Trend decomposition using LOESS) separates the series into trend, seasonality, and residual. Anomalies appear in the **residual component** — stripped of trend and seasonal effects, anomalies stand out more clearly:

```
xₜ = Trendₜ + Seasonalₜ + Residualₜ

If xₜ has a seasonal spike at Christmas:
  → Residualₜ is small (seasonal component explains it) → NOT anomaly

If xₜ spikes randomly in March:
  → Residualₜ is large (trend + season don't explain it) → ANOMALY

→ STL residuals give a much cleaner anomaly signal than raw z-scores
```

### 7.2 Implementation

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

def stl_anomaly_detector(
    series: np.ndarray,
    period: int = 12,
    threshold_sigma: float = 3.0,
    robust: bool = True,
) -> pd.DataFrame:
    """
    STL decomposition + residual Z-score anomaly detection.

    Parameters
    ----------
    series          : 1D time series (must be at least 2 full seasons)
    period          : seasonal period (12 = monthly, 7 = weekly, 24 = hourly)
    threshold_sigma : flag residuals with |z| > this as anomalies
    robust          : use robust LOESS fitting (resistant to outliers in STL fit)

    Returns
    -------
    DataFrame with STL components and anomaly flags
    """
    series = np.asarray(series, dtype=float)

    stl    = STL(series, period=period, robust=robust)
    result = stl.fit()

    residual  = result.resid
    trend     = result.trend
    seasonal  = result.seasonal

    # Anomaly score from residuals
    res_mu    = np.nanmedian(residual)                  # use median (robust)
    res_mad   = np.nanmedian(np.abs(residual - res_mu))
    res_sigma = res_mad / 0.6745 + 1e-12               # convert MAD to σ equivalent

    z_residual = (residual - res_mu) / res_sigma
    anomaly    = np.abs(z_residual) > threshold_sigma

    return pd.DataFrame({
        "value":      series,
        "trend":      trend,
        "seasonal":   seasonal,
        "residual":   residual,
        "z_residual": z_residual,
        "anomaly":    anomaly,
    })
```

### 7.3 STL Anomaly Workflow

```
1. Fit STL(series, period=S, robust=True)
2. Extract residuals r = x - trend - seasonal
3. Compute robust statistics: median(r), MAD(r)
4. Flag: |r - median| / MAD > 3.5 as anomaly
5. Optionally: re-fit STL with anomalous points replaced → cleaner decomposition
```

---

## 8. Threshold Calibration

### 8.1 The Threshold Problem

All statistical detectors require a threshold. Setting it:
- **Too low** → many false positives (alert fatigue)
- **Too high** → many missed detections (dangerous in safety-critical systems)

### 8.2 Calibration Strategies

```python
def calibrate_threshold(
    anomaly_scores: np.ndarray,
    method: str = "percentile",
    target_fpr: float = 0.01,
    percentile: float = 99.0,
    n_sigma: float = 3.0,
) -> float:
    """
    Calibrate anomaly detection threshold from anomaly scores.

    Parameters
    ----------
    anomaly_scores : array of anomaly scores (higher = more anomalous)
    method         : 'percentile', 'sigma', or 'fpr'
    target_fpr     : target false positive rate (for 'fpr' method)
    percentile     : percentile to use (for 'percentile' method)
    n_sigma        : sigma multiplier (for 'sigma' method)

    Returns
    -------
    threshold : float — flag score > threshold as anomaly
    """
    scores = np.asarray(anomaly_scores, dtype=float)
    scores = scores[~np.isnan(scores)]

    if method == "percentile":
        # Top `100 - percentile`% flagged as anomalies
        return float(np.percentile(scores, percentile))

    elif method == "sigma":
        mu, sigma = scores.mean(), scores.std()
        return float(mu + n_sigma * sigma)

    elif method == "fpr":
        # Set threshold so only target_fpr fraction are flagged
        return float(np.quantile(scores, 1 - target_fpr))

    raise ValueError(f"Unknown method: {method}")
```

---

## 9. Production Implementation

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

class StatisticalAnomalyDetector:
    """
    Production-ready statistical anomaly detector combining
    STL decomposition + CUSUM for drift + rolling z-score for spikes.

    Fits on normal data; scores new observations in real time.
    """

    def __init__(
        self,
        period: int = 12,
        z_threshold: float = 3.0,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
    ):
        self.period      = period
        self.z_threshold = z_threshold
        self.cusum_k     = cusum_k
        self.cusum_h     = cusum_h
        self._fitted     = False

    def fit(self, train_series: np.ndarray) -> "StatisticalAnomalyDetector":
        """Fit on normal (anomaly-free) training data."""
        self._train     = np.asarray(train_series, dtype=float)
        self._mu        = np.nanmean(self._train)
        self._sigma     = np.nanstd(self._train)
        self._fitted    = True
        self._S_pos     = 0.0   # CUSUM state
        self._S_neg     = 0.0
        return self

    def score(self, x: float) -> dict:
        """
        Score a single new observation in real time.

        Returns dict with z_score, CUSUM states, and anomaly flag.
        """
        assert self._fitted, "Call .fit() first."

        z   = (x - self._mu) / (self._sigma + 1e-12)
        z_n = (x - self._mu) / self._sigma     # for CUSUM

        self._S_pos = max(0.0, self._S_pos + z_n - self.cusum_k)
        self._S_neg = max(0.0, self._S_neg - z_n - self.cusum_k)

        anomaly_z     = abs(z) > self.z_threshold
        anomaly_cusum = (self._S_pos > self.cusum_h) or (self._S_neg > self.cusum_h)

        return {
            "value":        x,
            "z_score":      float(z),
            "S_pos":        float(self._S_pos),
            "S_neg":        float(self._S_neg),
            "anomaly_spike": bool(anomaly_z),
            "anomaly_drift": bool(anomaly_cusum),
            "anomaly":      bool(anomaly_z or anomaly_cusum),
        }

    def score_batch(self, series: np.ndarray) -> pd.DataFrame:
        """Score a batch of observations; resets CUSUM state."""
        self._S_pos = 0.0
        self._S_neg = 0.0
        return pd.DataFrame([self.score(x) for x in series])
```

---

*← [Module README](./README.md) | Next: [02 — Isolation Forest](./02_isolation_forest_for_ts.md) →*
