"""
code/01_statistical_methods.py
================================
Module 09 — Anomaly Detection
Practical: Z-score, rolling statistics, CUSUM, Bollinger Bands, STL residuals.

Demonstrates:
  - Global vs. rolling Z-score detection
  - Robust Z-score (MAD-based) for outlier-resistant baselines
  - CUSUM control chart for drift detection
  - Bollinger Bands with %B statistic
  - STL decomposition + residual z-score
  - Comparison of all methods on a synthetic anomalous series
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Series with Injected Anomalies
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
N = 365   # 1 year of daily data

t        = np.arange(N)
trend    = 0.05 * t
seasonal = 10 * np.sin(2 * np.pi * t / 52)   # 52-week cycle
noise    = np.random.normal(0, 1.5, N)
series   = 50 + trend + seasonal + noise

# Inject anomalies
anomaly_indices = [50, 120, 180, 240, 300]
anomaly_types   = ["spike_up", "spike_down", "drift_start", "spike_up", "spike_down"]

series_anomalous  = series.copy()
series_anomalous[50]  += 25       # large spike up
series_anomalous[120] -= 20       # large spike down
series_anomalous[180:200] += 8    # gradual drift (CUSUM target)
series_anomalous[240] += 30       # extreme spike
series_anomalous[300] -= 22       # another spike

true_anomalies = np.zeros(N, dtype=bool)
true_anomalies[[50, 120, 240, 300]] = True
true_anomalies[180:200] = True

dates  = pd.date_range("2023-01-01", periods=N, freq="D")
s_orig = pd.Series(series_anomalous, index=dates)

print(f"Series: N={N}, mean={series_anomalous.mean():.1f}, std={series_anomalous.std():.1f}")
print(f"True anomaly count: {true_anomalies.sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Detector Implementations
# ─────────────────────────────────────────────────────────────────────────────

def zscore_detector(s, window=None, threshold=3.0):
    """Global or rolling Z-score detector."""
    s = pd.Series(s, dtype=float)
    if window is None:
        mu, sigma = s.mean(), s.std()
    else:
        mu    = s.rolling(window, min_periods=window//2).mean()
        sigma = s.rolling(window, min_periods=window//2).std()
    z       = (s - mu) / (sigma + 1e-12)
    upper   = mu + threshold * sigma
    lower   = mu - threshold * sigma
    return pd.DataFrame({"value": s, "z": z, "upper": upper, "lower": lower,
                          "anomaly": (np.abs(z) > threshold)})


def robust_zscore_detector(s, threshold=3.5):
    """Modified Z-score using median and MAD (Iglewicz & Hoaglin)."""
    s      = np.asarray(s, dtype=float)
    median = np.median(s)
    mad    = np.median(np.abs(s - median))
    if mad == 0:
        mad = np.mean(np.abs(s - median)) + 1e-12
    mz = 0.6745 * (s - median) / mad
    return pd.Series(np.abs(mz) > threshold)


def cusum_detector(s, k=0.5, h=5.0, n_ref=50):
    """CUSUM control chart for detecting persistent mean shifts."""
    s      = np.asarray(s, dtype=float)
    mu0    = s[:n_ref].mean()
    sigma  = s[:n_ref].std() + 1e-12
    x_norm = (s - mu0) / sigma
    S_pos  = np.zeros(len(s))
    S_neg  = np.zeros(len(s))
    for t in range(1, len(s)):
        S_pos[t] = max(0, S_pos[t-1] + x_norm[t] - k)
        S_neg[t] = max(0, S_neg[t-1] - x_norm[t] - k)
    alert = (S_pos > h) | (S_neg > h)
    return pd.DataFrame({"S_pos": S_pos, "S_neg": S_neg, "alert": alert})


def bollinger_detector(s, window=20, n_std=2.0):
    """Bollinger Bands anomaly detector with %B statistic."""
    s    = pd.Series(s, dtype=float)
    sma  = s.rolling(window, min_periods=window//2).mean()
    std  = s.rolling(window, min_periods=window//2).std()
    up   = sma + n_std * std
    lo   = sma - n_std * std
    pct  = (s - lo) / (up - lo + 1e-12)
    return pd.DataFrame({"value": s, "sma": sma, "upper": up, "lower": lo,
                          "pct_b": pct, "anomaly": (s > up) | (s < lo)})


def stl_detector(s, period=52, threshold=3.5, robust=True):
    """STL decomposition + MAD-based residual anomaly detection."""
    s      = np.asarray(s, dtype=float)
    result = STL(s, period=period, robust=robust).fit()
    resid  = result.resid
    median = np.nanmedian(resid)
    mad    = np.nanmedian(np.abs(resid - median))
    sigma  = mad / 0.6745 + 1e-12
    z      = (resid - median) / sigma
    return pd.DataFrame({
        "value":    s,
        "trend":    result.trend,
        "seasonal": result.seasonal,
        "residual": resid,
        "z_resid":  z,
        "anomaly":  np.abs(z) > threshold,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 3. Run All Detectors
# ─────────────────────────────────────────────────────────────────────────────

print("\nRunning all statistical detectors...")

res_global  = zscore_detector(series_anomalous, window=None, threshold=3.0)
res_rolling = zscore_detector(series_anomalous, window=30,   threshold=3.0)
res_robust  = robust_zscore_detector(series_anomalous, threshold=3.5)
res_cusum   = cusum_detector(series_anomalous, k=0.5, h=5.0, n_ref=30)
res_boll    = bollinger_detector(series_anomalous, window=20, n_std=2.0)
res_stl     = stl_detector(series_anomalous, period=52, threshold=3.5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_detector(pred_anomalies, true_anomalies, name):
    """Compute precision, recall, F1 for binary anomaly detection."""
    pred  = np.asarray(pred_anomalies, dtype=bool)
    true_ = np.asarray(true_anomalies, dtype=bool)
    tp    = (pred & true_).sum()
    fp    = (pred & ~true_).sum()
    fn    = (~pred & true_).sum()
    prec  = tp / (tp + fp + 1e-12)
    rec   = tp / (tp + fn + 1e-12)
    f1    = 2 * prec * rec / (prec + rec + 1e-12)
    return {
        "Method":    name,
        "TP":        int(tp),
        "FP":        int(fp),
        "FN":        int(fn),
        "Precision": round(prec, 3),
        "Recall":    round(rec, 3),
        "F1":        round(f1, 3),
        "Flagged":   int(pred.sum()),
    }

rows = [
    evaluate_detector(res_global["anomaly"],         true_anomalies, "Global Z-score"),
    evaluate_detector(res_rolling["anomaly"],        true_anomalies, "Rolling Z-score (W=30)"),
    evaluate_detector(res_robust,                    true_anomalies, "Robust Z-score (MAD)"),
    evaluate_detector(res_cusum["alert"],            true_anomalies, "CUSUM"),
    evaluate_detector(res_boll["anomaly"],           true_anomalies, "Bollinger Bands"),
    evaluate_detector(res_stl["anomaly"],            true_anomalies, "STL Residuals"),
]
eval_df = pd.DataFrame(rows).set_index("Method")
print("\n" + "="*70)
print("DETECTOR COMPARISON")
print("="*70)
print(eval_df.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 5. Visualization
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)

# Panel 1: Rolling Z-score
ax = axes[0]
ax.plot(dates, series_anomalous, color="steelblue", linewidth=1.2, label="Series")
ax.fill_between(dates, res_rolling["lower"], res_rolling["upper"],
                color="steelblue", alpha=0.15, label="Rolling ±3σ")
anomaly_mask = res_rolling["anomaly"].values
ax.scatter(dates[anomaly_mask], series_anomalous[anomaly_mask],
           color="red", s=50, zorder=5, label="Detected anomalies")
ax.scatter(dates[true_anomalies], series_anomalous[true_anomalies],
           color="orange", s=30, marker="^", zorder=4, label="True anomalies")
ax.set_title("Rolling Z-score (window=30, threshold=3σ)", fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

# Panel 2: CUSUM
ax = axes[1]
ax.plot(dates, res_cusum["S_pos"], color="green",  linewidth=1.5, label="S+")
ax.plot(dates, res_cusum["S_neg"], color="purple", linewidth=1.5, label="S−")
ax.axhline(5.0, color="red", linestyle="--", linewidth=1.5, label="h=5 threshold")
ax.fill_between(dates, 0, 5, where=res_cusum["alert"].values,
                color="red", alpha=0.2, label="Alert zone")
ax.set_title("CUSUM Control Chart (k=0.5, h=5.0)", fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

# Panel 3: Bollinger Bands
ax = axes[2]
ax.plot(dates, series_anomalous, color="steelblue", linewidth=1.2)
ax.plot(dates, res_boll["sma"],   color="gray",  linewidth=1.2, linestyle="--", label="SMA")
ax.fill_between(dates, res_boll["lower"], res_boll["upper"],
                color="gold", alpha=0.3, label="Bollinger Band ±2σ")
boll_mask = res_boll["anomaly"].values
ax.scatter(dates[boll_mask], series_anomalous[boll_mask],
           color="red", s=50, zorder=5, label="Outside band")
ax.set_title("Bollinger Bands (window=20, n=2.0)", fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

# Panel 4: STL Residuals
ax = axes[3]
ax.plot(dates, res_stl["residual"], color="darkorange", linewidth=1.2, label="Residual")
ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
stl_thr = 3.5 * (np.nanmedian(np.abs(res_stl["residual"])) / 0.6745)
ax.axhline(+stl_thr, color="red", linestyle="--", label=f"±{3.5:.1f}·MAD/0.6745")
ax.axhline(-stl_thr, color="red", linestyle="--")
stl_mask = res_stl["anomaly"].values
ax.scatter(dates[stl_mask], res_stl["residual"][stl_mask],
           color="red", s=50, zorder=5, label="Detected")
ax.set_title("STL Residuals (period=52, threshold=3.5·MAD)", fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

plt.suptitle("Statistical Anomaly Detection Methods Comparison",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("statistical_anomaly_detection.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: statistical_anomaly_detection.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Production Detector Class Demo
# ─────────────────────────────────────────────────────────────────────────────

class StatisticalDetector:
    """Lightweight combined Z-score + CUSUM detector."""
    def __init__(self, z_thresh=3.0, cusum_h=5.0, cusum_k=0.5, n_ref=30):
        self.z_thresh = z_thresh; self.h = cusum_h; self.k = cusum_k
        self._fitted  = False; self._S_pos = self._S_neg = 0.0

    def fit(self, train):
        self._mu = np.mean(train[:30]); self._s = np.std(train[:30]) + 1e-12
        self._fitted = True; return self

    def score(self, x):
        z   = (x - self._mu) / self._s
        xn  = (x - self._mu) / self._s
        self._S_pos = max(0, self._S_pos + xn - self.k)
        self._S_neg = max(0, self._S_neg - xn - self.k)
        return {"z": float(z), "S+": self._S_pos, "S-": self._S_neg,
                "anomaly": abs(z) > self.z_thresh or self._S_pos > self.h or self._S_neg > self.h}

det = StatisticalDetector()
det.fit(series_anomalous[:50])
alerts = [det.score(x) for x in series_anomalous[50:]]
print(f"\nProduction detector — flagged {sum(a['anomaly'] for a in alerts)} anomalies in {len(alerts)} test steps")
