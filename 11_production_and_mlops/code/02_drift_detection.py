"""
code/02_drift_detection.py
============================
Module 11 — Production & MLOps
Practical: Production drift detection and monitoring.

Demonstrates:
  - PSI computation with bin-level breakdown
  - KS test + Wasserstein distance per feature
  - Rolling MAE-based concept drift monitor
  - CUSUM control chart for performance monitoring
  - Alert evaluation against threshold rules
  - Full monitoring dashboard visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ks_2samp, wasserstein_distance
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Dataset — Training + Serving with Drift
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
N_TRAIN   = 500
N_SERVE   = 200
T         = 365   # time steps for performance monitoring

# Training distribution (baseline)
train_data = {
    "lag_1":       np.random.normal(100, 15, N_TRAIN),
    "lag_7":       np.random.normal(100, 18, N_TRAIN),
    "roll_mean_7": np.random.normal(100, 10, N_TRAIN),
    "roll_std_7":  np.random.exponential(5, N_TRAIN),
    "day_of_week": np.random.randint(0, 7, N_TRAIN).astype(float),
}
train_df = pd.DataFrame(train_data)

# Serving — moderate drift in some features
serve_data = {
    "lag_1":       np.random.normal(115, 20, N_SERVE),   # ← mean shifted +15
    "lag_7":       np.random.normal(115, 22, N_SERVE),   # ← mild shift
    "roll_mean_7": np.random.normal(100, 10, N_SERVE),   # ← no drift
    "roll_std_7":  np.random.exponential(10, N_SERVE),   # ← spread doubled
    "day_of_week": np.random.randint(0, 7, N_SERVE).astype(float),  # ← no drift
}
serve_df = pd.DataFrame(serve_data)

feature_cols = list(train_df.columns)
print(f"Reference: {N_TRAIN} samples | Serving: {N_SERVE} samples")
print(f"Intentional drift in: lag_1, lag_7, roll_std_7")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PSI Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_psi(reference, current, n_bins=10, epsilon=1e-6):
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    bin_edges[0] -= 1e-10; bin_edges[-1] += 1e-10

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current,   bins=bin_edges)
    ref_pct = np.where(ref_counts == 0, epsilon, ref_counts / len(reference))
    cur_pct = np.where(cur_counts == 0, epsilon, cur_counts / len(current))

    psi = float(np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct)))
    return psi, ref_pct, cur_pct, bin_edges


def psi_severity(psi):
    if psi < 0.1:   return "✅ Stable",  "green"
    if psi < 0.25:  return "⚠️ Moderate","orange"
    return             "🔴 Major drift", "red"

print("\nPSI Analysis:")
psi_results = []
for col in feature_cols:
    psi_val, _, _, _ = compute_psi(train_df[col].values, serve_df[col].values)
    label, color     = psi_severity(psi_val)
    psi_results.append({"feature": col, "psi": round(psi_val, 5), "status": label})
    print(f"  {col:20s}: PSI={psi_val:.5f}  {label}")

psi_df = pd.DataFrame(psi_results)


# ─────────────────────────────────────────────────────────────────────────────
# 3. KS Test + Wasserstein Distance
# ─────────────────────────────────────────────────────────────────────────────

print("\nKS Test + Wasserstein:")
drift_report = []
for col in feature_cols:
    ref = train_df[col].values
    cur = serve_df[col].values
    ks_stat, ks_pval = ks_2samp(ref, cur)
    wass = wasserstein_distance(ref, cur)
    wass_norm = wass / (ref.std() + 1e-12)

    drift_report.append({
        "feature":      col,
        "psi":          round(compute_psi(ref, cur)[0], 5),
        "ks_stat":      round(ks_stat, 4),
        "ks_pval":      round(ks_pval, 5),
        "wasserstein":  round(wass_norm, 4),
        "drifted":      (compute_psi(ref, cur)[0] >= 0.1) or (ks_pval < 0.05),
        "ref_mean":     round(ref.mean(), 2),
        "cur_mean":     round(cur.mean(), 2),
    })
    print(f"  {col:20s}: KS={ks_stat:.4f}(p={ks_pval:.4f}), "
          f"W={wass_norm:.4f}, drifted={drift_report[-1]['drifted']}")

drift_df = pd.DataFrame(drift_report)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Concept Drift — Rolling MAE Simulation
# ─────────────────────────────────────────────────────────────────────────────

# Simulate production performance over time
# Phase 1 (0-200): stable, mae ≈ baseline
# Phase 2 (200-300): gradual degradation (concept drift)
# Phase 3 (300-365): post-retrain recovery

BASELINE_MAE = 5.0
WINDOW       = 30

errors_timeline = []
timestamps = pd.date_range("2024-01-01", periods=T, freq="D")

for t in range(T):
    if t < 200:
        err = BASELINE_MAE + np.random.normal(0, 1.5)
    elif t < 300:
        # Gradual concept drift
        progress = (t - 200) / 100
        err = BASELINE_MAE * (1 + 0.8 * progress) + np.random.normal(0, 1.5)
    else:
        # Post-retrain: drop back to baseline + slight improvement
        err = BASELINE_MAE * 0.95 + np.random.normal(0, 1.5)
    errors_timeline.append(max(0.5, err))

errors = np.array(errors_timeline)
rolling_mae = pd.Series(errors).rolling(WINDOW, min_periods=5).mean().values


# ─────────────────────────────────────────────────────────────────────────────
# 5. CUSUM Monitor
# ─────────────────────────────────────────────────────────────────────────────

k = 0.5; h = 5.0
BASELINE_STD = BASELINE_MAE * 0.3

S_pos = []; S_neg = []
sp, sn = 0.0, 0.0
cusum_alerts = []

for t, err in enumerate(errors):
    z  = (err - BASELINE_MAE) / BASELINE_STD
    sp = max(0, sp + z - k)
    sn = max(0, sn - z - k)

    # Reset after retrain event
    if t == 300:
        sp, sn = 0.0, 0.0

    S_pos.append(sp); S_neg.append(sn)
    if sp > h or sn > h:
        cusum_alerts.append(t)

S_pos = np.array(S_pos); S_neg = np.array(S_neg)
print(f"\nCUSUM: {len(cusum_alerts)} alert steps detected")
print(f"  First alert at t={cusum_alerts[0] if cusum_alerts else 'none'}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Alert Evaluation
# ─────────────────────────────────────────────────────────────────────────────

monitoring_metrics = {
    "max_feature_psi":    float(psi_df["psi"].max()),
    "n_drifted_features": int(drift_df["drifted"].sum()),
    "rolling_mae":        float(rolling_mae[250]),    # sampled at peak drift
    "rolling_mae_ratio":  float(rolling_mae[250] / BASELINE_MAE),
    "cusum_alarm":        int(S_pos[250] > h),
}

print(f"\nMonitoring metrics at peak drift (t=250):")
for k, v in monitoring_metrics.items():
    print(f"  {k:25s}: {v:.4f}")

alerts_fired = []
if monitoring_metrics["max_feature_psi"] >= 0.25:
    alerts_fired.append(("CRITICAL", "PSI > 0.25", "retrain"))
elif monitoring_metrics["max_feature_psi"] >= 0.10:
    alerts_fired.append(("WARNING",  "PSI > 0.10", "monitor"))
if monitoring_metrics["rolling_mae_ratio"] >= 1.5:
    alerts_fired.append(("CRITICAL", "MAE > 1.5x baseline", "retrain"))
elif monitoring_metrics["rolling_mae_ratio"] >= 1.2:
    alerts_fired.append(("WARNING",  "MAE > 1.2x baseline", "notify"))
if monitoring_metrics["cusum_alarm"]:
    alerts_fired.append(("CRITICAL", "CUSUM alarm", "retrain"))

print(f"\nAlerts fired:")
for sev, msg, action in alerts_fired:
    print(f"  [{sev}] {msg} → action: {action}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualization Dashboard
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 12))
gs  = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

# Panel 1: PSI bar chart
ax1 = fig.add_subplot(gs[0, :2])
psi_colors = ["#F44336" if p >= 0.25 else "#FF9800" if p >= 0.1 else "#4CAF50"
              for p in psi_df["psi"]]
bars = ax1.bar(psi_df["feature"], psi_df["psi"], color=psi_colors)
ax1.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
ax1.axhline(0.25, color="red", linestyle="--", linewidth=1.2, label="Critical (0.25)")
ax1.axhline(0.10, color="orange", linestyle="--", linewidth=1.2, label="Warning (0.10)")
ax1.set_title("PSI Per Feature", fontsize=11)
ax1.set_ylabel("PSI"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3, axis="y")
ax1.tick_params(axis="x", rotation=20)

# Panel 2: Feature distribution — most drifted feature
ax2 = fig.add_subplot(gs[0, 2])
most_drifted = psi_df.sort_values("psi", ascending=False).iloc[0]["feature"]
ref_d = train_df[most_drifted].values
cur_d = serve_df[most_drifted].values
ax2.hist(ref_d, bins=25, alpha=0.6, color="#2196F3", label="Reference", density=True)
ax2.hist(cur_d, bins=25, alpha=0.6, color="#FF5722", label="Serving",   density=True)
ax2.set_title(f"Distribution: {most_drifted}", fontsize=11)
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

# Panel 3: Rolling MAE over time
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(timestamps, rolling_mae, color="#2196F3", linewidth=1.5, label="Rolling MAE (30d)")
ax3.axhline(BASELINE_MAE,       color="#4CAF50", linestyle="--", linewidth=1.5, label=f"Baseline MAE = {BASELINE_MAE}")
ax3.axhline(BASELINE_MAE * 1.5, color="#FF5722", linestyle=":",  linewidth=1.5, label="Alert threshold (1.5x)")
ax3.axhline(BASELINE_MAE * 1.2, color="#FF9800", linestyle=":",  linewidth=1.2, label="Warning threshold (1.2x)")
ax3.axvline(timestamps[200], color="purple", linestyle="-", linewidth=1.5, alpha=0.5, label="Drift onset (t=200)")
ax3.axvline(timestamps[300], color="green",  linestyle="-", linewidth=1.5, alpha=0.5, label="Model retrained (t=300)")
ax3.fill_between(timestamps, BASELINE_MAE * 1.5, rolling_mae,
                  where=(rolling_mae > BASELINE_MAE * 1.5), alpha=0.15, color="red")
ax3.set_title("Concept Drift Monitoring — Rolling MAE", fontsize=11)
ax3.set_ylabel("MAE"); ax3.legend(fontsize=8, loc="upper left"); ax3.grid(alpha=0.3)
ax3.tick_params(axis="x", rotation=30)

# Panel 4: CUSUM
ax4 = fig.add_subplot(gs[2, :2])
ax4.plot(timestamps, S_pos, color="#F44336", linewidth=1.2, label="CUSUM S+")
ax4.plot(timestamps, S_neg, color="#2196F3", linewidth=1.2, label="CUSUM S-")
ax4.axhline(h, color="black", linestyle="--", linewidth=1.2, label=f"Decision boundary h={h}")
ax4.axvline(timestamps[300], color="green", linestyle="-", linewidth=1.5, alpha=0.5, label="Retrain + reset")
if cusum_alerts:
    first_alert = timestamps[cusum_alerts[0]]
    ax4.axvline(first_alert, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label="First alert")
ax4.set_title("CUSUM Performance Monitor", fontsize=11)
ax4.set_ylabel("CUSUM Statistic"); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)
ax4.tick_params(axis="x", rotation=30)

# Panel 5: Alert summary
ax5 = fig.add_subplot(gs[2, 2])
ax5.axis("off")
summary_text = "MONITORING SUMMARY\n" + "─"*26 + "\n"
summary_text += f"Max PSI:        {monitoring_metrics['max_feature_psi']:.4f}\n"
summary_text += f"Drifted feats:  {monitoring_metrics['n_drifted_features']}/{len(feature_cols)}\n"
summary_text += f"Rolling MAE:    {monitoring_metrics['rolling_mae']:.4f}\n"
summary_text += f"MAE ratio:      {monitoring_metrics['rolling_mae_ratio']:.2f}x\n"
summary_text += f"CUSUM alarm:    {'YES' if monitoring_metrics['cusum_alarm'] else 'no'}\n"
summary_text += "\nALERTS FIRED:\n"
for sev, msg, action in alerts_fired:
    summary_text += f"  [{sev}] {msg}\n"
summary_text += f"\nACTION: {'RETRAIN' if any(a[2]=='retrain' for a in alerts_fired) else 'MONITOR'}"

ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
          fontsize=9, verticalalignment="top", fontfamily="monospace",
          bbox=dict(boxstyle="round", facecolor="#FFF9C4", alpha=0.8))

plt.suptitle("Production Drift Detection Dashboard", fontsize=14, fontweight="bold")
plt.savefig("drift_detection_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nDashboard saved: drift_detection_dashboard.png")
