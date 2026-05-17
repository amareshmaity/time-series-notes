"""
code/04_online_detection.py
=============================
Module 09 — Anomaly Detection
Practical: Online/streaming anomaly detection algorithms.

Demonstrates:
  - Welford online Z-score with exponential forgetting
  - ADWIN drift detector
  - Online HBOS (histogram-based scoring)
  - RRCF via river library (with fallback to HBOS if not installed)
  - Streaming pipeline combining spike + drift detection
  - Visualization: real-time score trace and drift events
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Streaming Series — Distribution Shifts + Spikes
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(99)
N = 800

segments = [
    ("Normal 1",   np.random.normal(50, 3, 200)),
    ("Spike zone", np.concatenate([np.random.normal(50, 3, 50),
                                   [90, 85, -10, 92, 88],          # spikes
                                   np.random.normal(50, 3, 45)])),
    ("Drift",      np.random.normal(65, 3, 200)),                   # mean shifted to 65
    ("Normal 2",   np.random.normal(65, 5, 200)),                   # higher variance
    ("Return",     np.random.normal(50, 3, 100)),                   # return to original
]

series  = np.concatenate([s[1] for s in segments])
seg_labels = []
for name, arr in segments:
    seg_labels.extend([name] * len(arr))

true_spikes = np.zeros(len(series), dtype=bool)
for i, x in enumerate(series):
    if abs(x - 50) > 30 or abs(x - 65) > 25:
        true_spikes[i] = True

print(f"Streaming series: N={len(series)}")
print(f"Segment boundaries: {np.cumsum([len(s[1]) for s in segments])}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Online Detectors
# ─────────────────────────────────────────────────────────────────────────────

class OnlineZScore:
    """Welford online Z-score with optional exponential forgetting."""
    def __init__(self, threshold=3.5, forgetting=None):
        self.threshold = threshold
        self.forgetting = forgetting
        self._n = self._mu = self._M = self._w = 0.0

    @property
    def std(self):
        if self._n < 2: return 1.0
        d = (self._w - self._w / self._n) if self.forgetting else (self._n - 1)
        return float(np.sqrt(max(self._M / d, 1e-12)))

    def update(self, x):
        self._n += 1
        if self.forgetting:
            lam = self.forgetting
            self._w = lam * self._w + 1.0
            delta   = x - self._mu
            self._mu += delta / self._w
            self._M   = lam * self._M + delta * (x - self._mu)
        else:
            delta    = x - self._mu
            self._mu += delta / self._n
            self._M   += delta * (x - self._mu)
            self._w    = float(self._n)
        z = (x - self._mu) / (self.std + 1e-12)
        return {"z": float(z), "mu": self._mu, "std": self.std,
                "anomaly": abs(z) > self.threshold}


class ADWIN:
    """Adaptive Windowing drift detector."""
    def __init__(self, delta=0.002):
        self.delta   = delta
        self._window = deque()
        self._n = self._total = self._total2 = 0.0

    @property
    def mean(self): return self._total / self._n if self._n > 0 else 0.0

    @property
    def variance(self):
        if self._n < 2: return 0.0
        return (self._total2 - self._total**2 / self._n) / (self._n - 1)

    def update(self, x):
        self._window.append(x)
        self._n      += 1
        self._total  += x
        self._total2 += x**2
        drift = self._detect()
        return {"drift": drift, "mu": self.mean, "n": int(self._n), "var": self.variance}

    def _detect(self):
        if self._n < 4: return False
        wl = list(self._window)
        splits = range(max(1, int(self._n)//10), int(self._n)-1, max(1, int(self._n)//20))
        for m0 in splits:
            m1 = int(self._n) - m0
            if m0 < 1 or m1 < 1: continue
            mu0, mu1 = np.mean(wl[:m0]), np.mean(wl[m0:])
            eps = np.sqrt((1/(2*m0) + 1/(2*m1)) * np.log(4*self._n**2/self.delta))
            if abs(mu0 - mu1) > eps:
                for _ in range(m0):
                    old = self._window.popleft()
                    self._n -= 1; self._total -= old; self._total2 -= old**2
                return True
        return False


class OnlineHBOS:
    """Online Histogram-Based Outlier Score with exponential updates."""
    def __init__(self, n_bins=20, alpha=0.005):
        self.n_bins = n_bins; self.alpha = alpha
        self._fitted = False

    def fit(self, X_init):
        X_init = np.asarray(X_init, float)
        self._lo = X_init.min() - 1
        self._hi = X_init.max() + 1
        self._edges  = np.linspace(self._lo, self._hi, self.n_bins + 1)
        counts, _    = np.histogram(X_init, bins=self._edges)
        self._counts = counts.astype(float) + 1e-6
        self._fitted = True; return self

    def _bin(self, x):
        return int(np.clip(np.searchsorted(self._edges[1:], x), 0, self.n_bins-1))

    def score(self, x):
        b   = self._bin(float(x))
        w   = self._edges[b+1] - self._edges[b] + 1e-12
        den = self._counts[b] / (self._counts.sum() * w)
        return float(np.log(1 / (den + 1e-12)))

    def update(self, x):
        b = self._bin(float(x))
        self._counts *= (1 - self.alpha)
        self._counts[b] += 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Try river's HalfSpaceTrees (with graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────

RIVER_AVAILABLE = False
try:
    from river import anomaly as river_anomaly
    RIVER_AVAILABLE = True
    print("river library available — using HalfSpaceTrees")
except ImportError:
    print("river not installed — using OnlineHBOS as fallback (pip install river)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Streaming Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class StreamingPipeline:
    """Combined spike + drift streaming anomaly detector."""
    def __init__(self, z_threshold=3.5, forgetting=0.99, adwin_delta=0.002,
                 hbos_bins=20, smooth_n=5, alert_cooldown=10):
        self.z_threshold   = z_threshold
        self.smooth_n      = smooth_n
        self.alert_cooldown = alert_cooldown
        self._z     = OnlineZScore(z_threshold, forgetting)
        self._adwin = ADWIN(adwin_delta)
        self._hbos  = None   # initialized after warmup
        self._buf   = deque(maxlen=smooth_n)
        self._last_alert = -alert_cooldown
        self._t = 0
        self.drift_events_ = []
        self.hbos_bins     = hbos_bins

    def process(self, x):
        self._t += 1
        z_res    = self._z.update(x)
        ad_res   = self._adwin.update(x)

        # Initialize HBOS after 30 observations
        if self._hbos is None and self._t == 30:
            warmup = list(self._adwin._window)
            self._hbos = OnlineHBOS(self.hbos_bins)
            self._hbos.fit(np.array(warmup))

        hbos_score = 0.0
        if self._hbos is not None:
            hbos_score = self._hbos.score(x)
            self._hbos.update(x)

        # Smooth z-score
        self._buf.append(abs(z_res["z"]))
        smoothed = np.mean(self._buf)

        # Alert with cooldown
        alert = (smoothed > self.z_threshold and
                 self._t - self._last_alert >= self.alert_cooldown)
        if alert:
            self._last_alert = self._t

        if ad_res["drift"]:
            self.drift_events_.append({"t": self._t, "new_mu": ad_res["mu"]})

        return {
            "t":          self._t,
            "value":      x,
            "z":          z_res["z"],
            "smoothed_z": smoothed,
            "hbos":       hbos_score,
            "anomaly":    alert,
            "drift":      ad_res["drift"],
            "adwin_mu":   ad_res["mu"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Run on Streaming Series
# ─────────────────────────────────────────────────────────────────────────────

print("\nStreaming anomaly detection...")

pipe    = StreamingPipeline(z_threshold=3.5, forgetting=0.99, adwin_delta=0.002)
records = [pipe.process(float(x)) for x in series]
result  = pd.DataFrame(records)

# river comparison (if available)
river_scores = []
if RIVER_AVAILABLE:
    river_model = river_anomaly.HalfSpaceTrees(n_trees=25, height=15,
                                                window_size=200, seed=42)
    for x in series:
        s = river_model.score_one({"x": x})
        river_model.learn_one({"x": x})
        river_scores.append(s)
    result["river_score"] = river_scores


# ─────────────────────────────────────────────────────────────────────────────
# 6. Summary Statistics
# ─────────────────────────────────────────────────────────────────────────────

n_alerts = result["anomaly"].sum()
n_drifts = result["drift"].sum()
print(f"\nAlerts fired:        {n_alerts}")
print(f"Drift events:        {n_drifts}")
print(f"Drift at timesteps:  {[e['t'] for e in pipe.drift_events_][:10]}")

# Segment-wise anomaly rate
result["segment"] = seg_labels[:len(result)]
seg_rates = result.groupby("segment")["anomaly"].mean().round(3)
print(f"\nAnomaly rate by segment:\n{seg_rates.to_string()}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualization
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
T         = np.arange(len(series))

# Segment backgrounds
seg_colors = {"Normal 1": "#E3F2FD", "Spike zone": "#FFF9C4", "Drift": "#FCE4EC",
              "Normal 2": "#E8F5E9", "Return": "#F3E5F5"}
boundaries = np.cumsum([len(s[1]) for s in segments])
starts     = np.concatenate([[0], boundaries[:-1]])

# Panel 1: Series + anomaly alerts + drift events
ax = axes[0]
for (name, _), s, e in zip(segments, starts, boundaries):
    ax.axvspan(s, e, alpha=0.12, color=seg_colors.get(name, "white"), label=name)
ax.plot(series, color="steelblue", linewidth=1.2, label="Series")
alert_idx = result[result["anomaly"]].index.values
ax.scatter(alert_idx, series[alert_idx], color="red", s=40, zorder=5, label=f"Alerts ({n_alerts})")
for ev in pipe.drift_events_:
    ax.axvline(ev["t"], color="orange", linestyle="--", linewidth=1.5)
ax.set_title("Streaming Series — Alerts (red) and Drift Events (orange dashed)", fontsize=11)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right", ncol=2)
ax.grid(alpha=0.3)

# Panel 2: Online Z-score + smoothed
ax = axes[1]
ax.plot(result["z"], color="gray", linewidth=0.8, alpha=0.5, label="Raw |z|")
ax.plot(result["smoothed_z"].abs(), color="#2196F3", linewidth=1.8, label="Smoothed |z|")
ax.axhline(3.5, color="red", linestyle="--", linewidth=1.5, label="Threshold")
ax.plot(result["adwin_mu"], color="green", linewidth=1.5, linestyle=":", label="ADWIN μ")
ax.set_title("Online Z-Score (Welford + Forgetting) with ADWIN Mean Estimate", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Panel 3: HBOS score
ax = axes[2]
ax.plot(result["hbos"], color="#FF9800", linewidth=1.5, label="HBOS score")
hbos_thr = np.nanpercentile(result["hbos"].values, 99)
ax.axhline(hbos_thr, color="red", linestyle="--", linewidth=1.5, label=f"p99 threshold ({hbos_thr:.1f})")
hbos_alerts = result["hbos"].values > hbos_thr
ax.scatter(T[hbos_alerts], result["hbos"].values[hbos_alerts],
           color="red", s=30, zorder=5, label=f"Alerts ({hbos_alerts.sum()})")
ax.set_title("Online HBOS Score", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_xlabel("Timestep")

plt.suptitle("Online Streaming Anomaly Detection Pipeline",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("online_anomaly_detection.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: online_anomaly_detection.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Detection Latency Analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*50)
print("DETECTION LATENCY ANALYSIS")
print("="*50)
print("(How many steps after the drift boundary was change detected?)")

seg_starts = {seg[0]: starts[i] for i, seg in enumerate(segments)}
drift_start = int(seg_starts["Drift"])
drift_events = [e["t"] for e in pipe.drift_events_ if e["t"] >= drift_start and e["t"] < drift_start + 50]
if drift_events:
    latency = drift_events[0] - drift_start
    print(f"Drift injected at t={drift_start}")
    print(f"First ADWIN detection at t={drift_events[0]}")
    print(f"Detection latency: {latency} steps")
else:
    print("ADWIN did not detect the drift within 50 steps")
