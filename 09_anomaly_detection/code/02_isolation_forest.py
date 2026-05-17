"""
code/02_isolation_forest.py
=============================
Module 09 — Anomaly Detection
Practical: Isolation Forest on time series with feature engineering.

Demonstrates:
  - Lag + rolling + calendar feature matrix
  - Isolation Forest fit on normal data, score on test
  - Threshold calibration at target FPR
  - Comparison: plain value IF vs. feature-engineered IF
  - One-Class SVM and LOF comparison
  - Evaluation against injected anomalies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Series with Anomalies
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
N = 500
t = np.arange(N)

trend    = 0.04 * t
seasonal = 8 * np.sin(2 * np.pi * t / 7) + 4 * np.sin(2 * np.pi * t / 365)
noise    = np.random.normal(0, 2, N)
series   = 100 + trend + seasonal + noise

# Inject anomalies (test portion: last 150 steps)
TRAIN_N = 350
series_anomalous = series.copy()
anomaly_points = [370, 400, 430, 460, 490]
for idx in anomaly_points:
    series_anomalous[idx] += np.random.choice([-1, 1]) * np.random.uniform(20, 35)

# Collective anomaly: flatline at 450–455
series_anomalous[450:456] = series_anomalous[449]

dates = pd.date_range("2022-01-01", periods=N, freq="D")
s_full = pd.Series(series_anomalous, index=dates, name="value")

true_anomalies = np.zeros(N, dtype=bool)
true_anomalies[anomaly_points] = True
true_anomalies[450:456]        = True

print(f"Series: N={N}, train={TRAIN_N}, test={N-TRAIN_N}")
print(f"Injected anomalies: {true_anomalies.sum()} points")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_features(s, lags=(1, 2, 3, 7, 14), windows=(7, 30)):
    """Build lag + rolling feature matrix from a pd.Series."""
    feat = pd.DataFrame(index=s.index)
    feat["value"] = s.values
    for lag in lags:
        feat[f"lag_{lag}"] = s.shift(lag).values
    feat["diff_1"] = s.diff(1).values
    feat["diff_7"] = s.diff(7).values
    for w in windows:
        rol = s.rolling(w, min_periods=w//2)
        feat[f"rmean_{w}"] = rol.mean().values
        feat[f"rstd_{w}"]  = rol.std().values
        feat[f"rz_{w}"]    = ((s - rol.mean()) / (rol.std() + 1e-12)).values
        feat[f"rmin_{w}"]  = rol.min().values
        feat[f"rmax_{w}"]  = rol.max().values
    # Calendar
    feat["dow"]   = s.index.dayofweek if hasattr(s.index, 'dayofweek') else 0
    feat["month"] = s.index.month if hasattr(s.index, 'month') else 0
    return feat.dropna()

features_full  = build_features(s_full)
feat_train     = features_full.iloc[:TRAIN_N - int(features_full.index[0] == s_full.index[14])]
feat_test      = features_full.iloc[len(feat_train):]
y_test         = true_anomalies[feat_train.index.get_loc(feat_test.index[0]) if feat_test.index[0] in s_full.index else 0:]

# Align true labels to feature index
true_test = np.array([true_anomalies[np.where(dates == d)[0][0]] if d in dates else False
                       for d in feat_test.index])

scaler = StandardScaler()
X_train = scaler.fit_transform(feat_train.values)
X_test  = scaler.transform(feat_test.values)

print(f"\nFeature matrix: {X_train.shape[1]} features")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Isolation Forest — Plain Value vs. Feature-Engineered
# ─────────────────────────────────────────────────────────────────────────────

# (a) Plain: use only current value
X_plain_train = s_full.values[:TRAIN_N].reshape(-1, 1)
X_plain_test  = s_full.values[TRAIN_N:].reshape(-1, 1)
true_plain    = true_anomalies[TRAIN_N:]

iso_plain = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
iso_plain.fit(X_plain_train)
score_plain = -iso_plain.decision_function(X_plain_test)
label_plain = iso_plain.predict(X_plain_test) == -1

# (b) Feature-engineered
iso_feat = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
iso_feat.fit(X_train)
score_feat = -iso_feat.decision_function(X_test)
label_feat = iso_feat.predict(X_test) == -1


# ─────────────────────────────────────────────────────────────────────────────
# 4. One-Class SVM and LOF
# ─────────────────────────────────────────────────────────────────────────────

ocsvm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
ocsvm.fit(X_train)
label_svm = ocsvm.predict(X_test) == -1

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
lof.fit(X_train)
label_lof = lof.predict(X_test) == -1


# ─────────────────────────────────────────────────────────────────────────────
# 5. Threshold Calibration Demo (Target FPR)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(scores, target_fpr=0.02):
    return float(np.quantile(scores, 1 - target_fpr))

def apply_threshold(scores, threshold):
    return scores > threshold

train_scores = -iso_feat.decision_function(X_train)
thr_1pct     = calibrate(train_scores, 0.01)
thr_2pct     = calibrate(train_scores, 0.02)
thr_5pct     = calibrate(train_scores, 0.05)

print(f"\nThreshold calibration (from training set):")
print(f"  Target 1% FPR: threshold = {thr_1pct:.4f}")
print(f"  Target 2% FPR: threshold = {thr_2pct:.4f}")
print(f"  Target 5% FPR: threshold = {thr_5pct:.4f}")
label_calibrated_2pct = apply_threshold(score_feat, thr_2pct)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(preds, true, name):
    true = true[:len(preds)]
    tp = (preds & true).sum()
    fp = (preds & ~true).sum()
    fn = (~preds & true).sum()
    p  = tp / (tp + fp + 1e-12)
    r  = tp / (tp + fn + 1e-12)
    f1 = 2*p*r / (p + r + 1e-12)
    return {"Method": name, "TP": int(tp), "FP": int(fp), "FN": int(fn),
            "Prec": round(p,3), "Recall": round(r,3), "F1": round(f1,3),
            "Flagged": int(preds.sum())}

# Align plain test labels
true_plain_aligned = true_plain[:len(label_plain)]
true_feat_aligned  = true_test[:len(label_feat)]

results = pd.DataFrame([
    evaluate(label_plain,           true_plain_aligned, "IF (plain value)"),
    evaluate(label_feat,            true_feat_aligned,  "IF (feature-engineered)"),
    evaluate(label_calibrated_2pct, true_feat_aligned,  "IF (calibrated 2% FPR)"),
    evaluate(label_svm,             true_feat_aligned,  "One-Class SVM"),
    evaluate(label_lof,             true_feat_aligned,  "LOF"),
]).set_index("Method")

print("\nEvaluation Results:")
print(results.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualization
# ─────────────────────────────────────────────────────────────────────────────

test_dates = feat_test.index
fig, axes  = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# Plot series + detected anomalies for 3 methods
configs = [
    ("IF (plain value)",        label_plain[:len(test_dates)],      "#F44336"),
    ("IF (feature-engineered)", label_feat[:len(test_dates)],       "#4CAF50"),
    ("Calibrated 2% FPR",       label_calibrated_2pct[:len(test_dates)], "#2196F3"),
]
test_vals = s_full.loc[test_dates].values

for ax, (title, preds, color) in zip(axes, configs):
    ax.plot(test_dates, test_vals, color="steelblue", linewidth=1.5, label="Series")
    ax.scatter(test_dates[preds[:len(test_dates)]], test_vals[preds[:len(test_dates)]],
               color=color, s=60, zorder=5, label=f"Detected ({preds[:len(test_dates)].sum()})")

    # True anomaly markers
    true_mask = true_test[:len(test_dates)]
    ax.scatter(test_dates[true_mask], test_vals[true_mask],
               color="orange", s=80, marker="^", zorder=4, label="True anomaly")

    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

plt.suptitle("Isolation Forest Anomaly Detection — Method Comparison",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("isolation_forest_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: isolation_forest_comparison.png")
