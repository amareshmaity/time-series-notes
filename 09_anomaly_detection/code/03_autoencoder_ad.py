"""
code/03_autoencoder_ad.py
==========================
Module 09 — Anomaly Detection
Practical: LSTM autoencoder for time series anomaly detection.

Demonstrates:
  - Building sliding windows from a 1D series
  - Training LSTM autoencoder on normal data
  - Computing per-step reconstruction error as anomaly score
  - Threshold calibration (percentile, sigma, GMM)
  - Comparison against statistical baseline
  - Visualization of reconstruction and anomaly scores
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture

# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset — ECG-style periodic signal with collective anomalies
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(7)
N     = 600
t     = np.arange(N)
period = 24   # simulate 24-step cycles (e.g., hourly IoT signal)

# Normal: smooth periodic signal
signal = np.sin(2 * np.pi * t / period) + 0.5 * np.sin(4 * np.pi * t / period)
noise  = np.random.normal(0, 0.1, N)
series = signal + noise

# Inject collective anomalies (waveform shape changes)
TRAIN_N = 450
anomaly_mask = np.zeros(N, dtype=bool)

# 1. Flatline (sensor stuck)
series[480:492]    = 0.3
anomaly_mask[480:492] = True

# 2. Amplitude spike
series[510:518] *= 3.5
anomaly_mask[510:518] = True

# 3. Phase shift (cyclical pattern changes)
series[540:555] = np.sin(2 * np.pi * t[:15] / period * 0.5)
anomaly_mask[540:555] = True

print(f"Series: N={N}, period={period}")
print(f"Train: {TRAIN_N} | Test: {N-TRAIN_N}")
print(f"Anomalies injected: {anomaly_mask.sum()} steps")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LSTM Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    """LSTM autoencoder for univariate sequence reconstruction."""

    def __init__(self, hidden_size=32, n_layers=2, seq_len=24):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(1, hidden_size, n_layers, batch_first=True,
                                dropout=0.0)
        self.decoder = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True,
                                dropout=0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """x: (batch, seq_len, 1)"""
        _, (h, c)  = self.encoder(x)
        ctx        = h[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _     = self.decoder(ctx, (h, c))
        return self.fc(out)   # (batch, seq_len, 1)


def build_windows(arr, w, stride=1):
    """Sliding windows from 1D array: returns (N_windows, w, 1)."""
    wins = [arr[i:i+w] for i in range(0, len(arr) - w + 1, stride)]
    return np.array(wins, dtype=np.float32)[:, :, np.newaxis]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training
# ─────────────────────────────────────────────────────────────────────────────

WINDOW = period     # window = one full cycle
STRIDE = 1
EPOCHS = 40
BATCH  = 64
LR     = 1e-3
DEVICE = "cpu"

# Normalize
mu    = series[:TRAIN_N].mean()
sigma = series[:TRAIN_N].std() + 1e-8
norm  = (series - mu) / sigma

# Training windows (from normal data)
X_train = build_windows(norm[:TRAIN_N], WINDOW, STRIDE)
X_train_t = torch.tensor(X_train)
loader    = DataLoader(TensorDataset(X_train_t, X_train_t), batch_size=BATCH, shuffle=True)

model     = LSTMAutoencoder(hidden_size=32, n_layers=2, seq_len=WINDOW).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print(f"\nTraining LSTM autoencoder ({EPOCHS} epochs)...")
losses = []
for epoch in range(EPOCHS):
    model.train()
    total = 0
    for xb, _ in loader:
        xb    = xb.to(DEVICE)
        x_hat = model(xb)
        loss  = criterion(x_hat, xb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item()
    losses.append(total / len(loader))
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {losses[-1]:.6f}")

print(f"Final training loss: {losses[-1]:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Compute Reconstruction Error
# ─────────────────────────────────────────────────────────────────────────────

def compute_error(model, arr_norm, window, stride=1, device="cpu"):
    """Per-step max reconstruction error across overlapping windows."""
    wins   = build_windows(arr_norm, window, stride)
    X_t    = torch.tensor(wins).to(device)
    model.eval()
    with torch.no_grad():
        x_hat = model(X_t)
    errors = ((X_t - x_hat)**2).squeeze(-1).cpu().numpy()  # (N_wins, window)

    # Aggregate: for each timestep, take the MAX error across all windows containing it
    score = np.full(len(arr_norm), np.nan)
    for i, err_row in enumerate(errors):
        for j, err in enumerate(err_row):
            t = i + j
            if t < len(score):
                score[t] = err if np.isnan(score[t]) else max(score[t], err)

    return score


# Scores on training set (for threshold calibration)
train_scores = compute_error(model, norm[:TRAIN_N], WINDOW)
test_scores  = compute_error(model, norm[TRAIN_N:], WINDOW)

print(f"\nTrain error stats — mean:{np.nanmean(train_scores):.6f}, "
      f"p95:{np.nanpercentile(train_scores, 95):.6f}, "
      f"p99:{np.nanpercentile(train_scores, 99):.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Threshold Calibration
# ─────────────────────────────────────────────────────────────────────────────

valid_train = train_scores[~np.isnan(train_scores)]

# Method 1: Percentile
thr_99   = float(np.nanpercentile(valid_train, 99))
thr_995  = float(np.nanpercentile(valid_train, 99.5))

# Method 2: Sigma
mu_e, s_e = valid_train.mean(), valid_train.std()
thr_3sig  = mu_e + 3.0 * s_e
thr_4sig  = mu_e + 4.0 * s_e

# Method 3: GMM (bimodal error distribution detection)
gmm_data = valid_train.reshape(-1, 1)
try:
    gmm      = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(gmm_data)
    means_   = gmm.means_.flatten()
    stds_    = np.sqrt(gmm.covariances_.flatten())
    weights_ = gmm.weights_
    # Decision boundary between two components
    lower_comp, upper_comp = sorted(range(2), key=lambda i: means_[i])
    thr_gmm = float((means_[lower_comp] / stds_[lower_comp] + means_[upper_comp] / stds_[upper_comp])
                    / (1/stds_[lower_comp] + 1/stds_[upper_comp]))
    print(f"\nGMM component 0: mean={means_[lower_comp]:.5f}, std={stds_[lower_comp]:.5f}")
    print(f"GMM component 1: mean={means_[upper_comp]:.5f}, std={stds_[upper_comp]:.5f}")
    print(f"GMM decision boundary: {thr_gmm:.6f}")
except Exception as e:
    thr_gmm = thr_995
    print(f"GMM failed ({e}), using p99.5 threshold")

print(f"\nCalibrated thresholds:")
print(f"  p99   threshold: {thr_99:.6f}")
print(f"  p99.5 threshold: {thr_995:.6f}")
print(f"  3σ    threshold: {thr_3sig:.6f}")
print(f"  GMM   threshold: {thr_gmm:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

true_test = anomaly_mask[TRAIN_N:][:len(test_scores)]
valid_mask = ~np.isnan(test_scores)

def eval_thr(scores, true, thr, name):
    valid = ~np.isnan(scores)
    pred  = np.zeros(len(scores), dtype=bool)
    pred[valid] = scores[valid] > thr
    tp = (pred & true).sum()
    fp = (pred & ~true).sum()
    fn = (~pred & true).sum()
    p  = tp/(tp+fp+1e-12); r = tp/(tp+fn+1e-12)
    f1 = 2*p*r/(p+r+1e-12)
    return {"Method": name, "Thr": round(thr,5), "TP": int(tp), "FP": int(fp),
            "FN": int(fn), "Prec": round(p,3), "Recall": round(r,3), "F1": round(f1,3)}

rows = [
    eval_thr(test_scores, true_test, thr_99,  "LSTM-AE (p99)"),
    eval_thr(test_scores, true_test, thr_995, "LSTM-AE (p99.5)"),
    eval_thr(test_scores, true_test, thr_3sig,"LSTM-AE (3σ)"),
    eval_thr(test_scores, true_test, thr_gmm, "LSTM-AE (GMM)"),
]
result_df = pd.DataFrame(rows).set_index("Method")
print("\nEvaluation (test set):")
print(result_df.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualization
# ─────────────────────────────────────────────────────────────────────────────

test_series = series[TRAIN_N:]
fig, axes   = plt.subplots(3, 1, figsize=(15, 11))

# Top: Series with true anomalies
axes[0].plot(test_series, color="steelblue", linewidth=1.5, label="Test series")
axes[0].fill_between(range(len(test_series)), test_series.min(), test_series.max(),
                      where=true_test[:len(test_series)],
                      color="red", alpha=0.2, label="True anomaly region")
axes[0].set_title("Test Series with Injected Anomaly Regions", fontsize=11)
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

# Middle: Reconstruction error + thresholds
valid_idx = np.where(~np.isnan(test_scores))[0]
axes[1].plot(valid_idx, test_scores[valid_idx], color="darkorange", linewidth=1.5, label="Recon error")
for thr, label, color in [(thr_99, "p99", "green"), (thr_995, "p99.5", "blue"), (thr_gmm, "GMM", "red")]:
    axes[1].axhline(thr, color=color, linestyle="--", linewidth=1.3, label=f"Thr ({label})")
axes[1].set_title("LSTM Autoencoder Reconstruction Error (anomaly score)", fontsize=11)
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

# Bottom: Detections
pred_gmm = np.zeros(len(test_scores), dtype=bool)
pred_gmm[valid_mask] = test_scores[valid_mask] > thr_gmm

axes[2].plot(test_series, color="steelblue", linewidth=1.5, alpha=0.6, label="Series")
axes[2].scatter(np.where(pred_gmm)[0], test_series[np.where(pred_gmm)[0]],
                color="red", s=50, zorder=5, label=f"Detected ({pred_gmm.sum()})")
axes[2].scatter(np.where(true_test[:len(pred_gmm)])[0], test_series[np.where(true_test[:len(pred_gmm)])[0]],
                color="orange", s=30, marker="^", zorder=4, label="True anomaly")
axes[2].set_title(f"Detected Anomalies (GMM threshold, F1={result_df.loc['LSTM-AE (GMM)', 'F1']:.3f})", fontsize=11)
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

plt.suptitle("LSTM Autoencoder Anomaly Detection", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("autoencoder_anomaly_detection.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: autoencoder_anomaly_detection.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Training Loss Curve
# ─────────────────────────────────────────────────────────────────────────────

plt.figure(figsize=(8, 4))
plt.plot(losses, color="#2196F3", linewidth=2)
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.title("LSTM Autoencoder Training Loss Curve")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("autoencoder_training_loss.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved: autoencoder_training_loss.png")
