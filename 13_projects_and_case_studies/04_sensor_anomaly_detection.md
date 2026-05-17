# 04 — Sensor Anomaly Detection

> **Module**: 13 Projects & Case Studies | **Project**: 4 of 5
> **Domain**: IoT / Manufacturing | **Problem**: Multivariate anomaly detection + Root Cause Analysis
>
> Industrial sensors generate continuous multivariate streams where anomalies signal equipment failure, process deviation, or cyber-physical attacks. Early detection reduces downtime; precise root cause narrows the repair scope. This project builds a layered detection pipeline: statistical → ML → deep learning, with a structured RCA step.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Dataset & EDA](#2-dataset--eda)
3. [Detection Pipeline Design](#3-detection-pipeline-design)
4. [Statistical Methods (Layer 1)](#4-statistical-methods-layer-1)
5. [Isolation Forest (Layer 2)](#5-isolation-forest-layer-2)
6. [LSTM Autoencoder (Layer 3)](#6-lstm-autoencoder-layer-3)
7. [Root Cause Analysis](#7-root-cause-analysis)
8. [Alert Pipeline & Monitoring](#8-alert-pipeline--monitoring)
9. [Key Lessons](#9-key-lessons)

---

## 1. Problem Definition

```
Business goal:
  Detect anomalies in multivariate sensor streams from industrial machinery.
  Alert maintenance team within 5 minutes of anomaly onset.
  Identify the primary fault channel (root cause) for faster repair.

Dataset options:
  - NASA SMAP / MSL: Mars rover and soil moisture telemetry (public)
  - SMD (Server Machine Dataset): 28 servers, 38 channels each
  - SWAT (Secure Water Treatment): 51 sensors, labeled attacks

KPIs:
  F1-score ≥ 0.70 on labeled test set
  Precision ≥ 0.80 (false alarms are expensive in manufacturing)
  Time-to-detect (TTD): alert within 5 minutes of anomaly start
  False positive rate < 5% on nominal data

Anomaly types to detect:
  Point anomaly:      single spike in one channel
  Contextual:         value normal in isolation but abnormal in context
  Collective:         sequence of normal-looking values that pattern-match an anomaly
```

---

## 2. Dataset & EDA

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_sensor_data(
    n_sensors: int = 8,
    n_steps:   int = 3000,
    seed:      int = 42,
) -> dict:
    """
    Synthetic multivariate sensor stream with labeled anomalies.
    
    Anomaly types injected:
      Type A (t=600-700):   point spike on sensors 0-1
      Type B (t=1200-1400): mean shift on sensors 2-3 (contextual)
      Type C (t=2000-2100): oscillation on sensors 4-5 (collective)
    """
    np.random.seed(seed)
    t    = np.arange(n_steps)
    data = np.zeros((n_steps, n_sensors))

    for s in range(n_sensors):
        freq  = 0.02 + s * 0.005
        amp   = 1.0 + s * 0.1
        data[:, s] = amp * np.sin(2*np.pi*freq*t) + np.random.normal(0, 0.1, n_steps)

    # Cross-sensor correlation: sensor 1 lags sensor 0
    data[:, 1] += 0.5 * np.roll(data[:, 0], 3)

    # Inject anomalies
    labels = np.zeros(n_steps)

    # Type A: spike
    data[600:650, 0] += 8.0
    data[620:680, 1] += 6.0
    labels[600:680] = 1

    # Type B: mean shift
    data[1200:1400, 2] += 3.5
    data[1200:1400, 3] -= 2.0
    labels[1200:1400] = 1

    # Type C: oscillation
    data[2000:2100, 4] += 4.0 * np.sin(2*np.pi*0.1*t[2000:2100])
    data[2000:2100, 5] += 3.5 * np.sin(2*np.pi*0.1*t[2000:2100] + np.pi/3)
    labels[2000:2100] = 1

    timestamps = pd.date_range("2024-01-01", periods=n_steps, freq="1min")
    df = pd.DataFrame(data, columns=[f"S{i:02d}" for i in range(n_sensors)], index=timestamps)
    return {"data": df, "labels": labels, "n_anomaly": int(labels.sum())}


def sensor_eda(data_dict: dict) -> None:
    df     = data_dict["data"]
    labels = data_dict["labels"]
    n, D   = df.shape

    fig, axes = plt.subplots(3, 1, figsize=(16, 10))

    # Raw sensor streams
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, D))
    for i, col in enumerate(df.columns):
        ax.plot(df.index, df[col].values + i*3, color=colors[i], linewidth=0.8, alpha=0.8, label=col)
    # Highlight anomaly regions
    for start, end in _find_anomaly_windows(labels):
        ax.axvspan(df.index[start], df.index[end], alpha=0.15, color="red")
    ax.set_title("Multivariate Sensor Streams (offset for clarity)", fontsize=11)
    ax.legend(fontsize=7, ncol=4, loc="upper right"); ax.grid(alpha=0.3)

    # Correlation matrix
    ax = axes[1]
    corr = df.corr()
    im   = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(D)); ax.set_xticklabels(df.columns, rotation=30, ha="right")
    ax.set_yticks(range(D)); ax.set_yticklabels(df.columns)
    ax.set_title("Sensor Correlation Matrix", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.6)

    # Class balance
    ax = axes[2]
    ax.bar(["Normal", "Anomaly"], [(labels==0).sum(), (labels==1).sum()],
            color=["#4CAF50","#F44336"])
    ax.set_title(f"Label Distribution (anomaly rate = {labels.mean()*100:.1f}%)", fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout(); plt.show()


def _find_anomaly_windows(labels):
    windows = []
    in_anom = False
    for t, l in enumerate(labels):
        if l == 1 and not in_anom:
            start = t; in_anom = True
        elif l == 0 and in_anom:
            windows.append((start, t)); in_anom = False
    if in_anom:
        windows.append((start, len(labels)-1))
    return windows
```

---

## 3. Detection Pipeline Design

```
LAYERED DETECTION PIPELINE:

  Layer 1 — Statistical (fast, interpretable):
    STL decomposition → residual z-score per channel
    Rolling mean ± 3σ control chart
    → High recall, low precision; generates candidate alerts

  Layer 2 — ML Ensemble (balanced):
    Isolation Forest on multi-channel windows
    Local Outlier Factor (streaming)
    → Ensemble score reduces false positives vs. Layer 1

  Layer 3 — Deep Learning (complex patterns):
    LSTM Autoencoder → reconstruction error
    Best for collective and contextual anomalies

  Fusion:
    Weighted vote: (L1 + L2 + L3) / 3 → threshold at 0.5
    OR: Any layer ≥ 0.8 triggers high-priority alert.

  Root Cause Analysis (post-alert):
    Per-channel anomaly score → top-k channels identified
    SHAP on Isolation Forest → which features contributed most
```

---

## 4. Statistical Methods (Layer 1)

```python
from scipy import stats

def zscore_detector(
    df: pd.DataFrame,
    window: int = 60,
    threshold: float = 3.5,
) -> pd.DataFrame:
    """
    Rolling z-score per channel. Returns combined anomaly score.
    
    Score = max over channels of |z_score|.
    Flags when any channel exceeds threshold.
    """
    z_scores = pd.DataFrame(index=df.index)
    for col in df.columns:
        rolling_mean = df[col].rolling(window, min_periods=10).mean()
        rolling_std  = df[col].rolling(window, min_periods=10).std()
        z_scores[col] = np.abs((df[col] - rolling_mean) / (rolling_std + 1e-12))

    score  = z_scores.max(axis=1)
    alert  = (score > threshold).astype(int)
    return pd.DataFrame({"score": score, "alert": alert}, index=df.index)


def cusum_multivariate(
    df: pd.DataFrame,
    k: float = 0.5,
    h: float = 5.0,
    window_train: int = 200,
) -> pd.DataFrame:
    """
    Multivariate CUSUM: run on the L2-norm of standardized observations.
    
    1. Standardize using training window statistics.
    2. Compute ||z_t||₂ — Mahalanobis-like distance.
    3. Run CUSUM on the scalar sequence.
    """
    train_mean = df.iloc[:window_train].mean()
    train_std  = df.iloc[:window_train].std() + 1e-12
    z          = (df - train_mean) / train_std
    norm_z     = np.sqrt((z**2).sum(axis=1))   # L2 norm of standardized obs

    mu_0  = float(norm_z.iloc[:window_train].mean())
    sigma = float(norm_z.iloc[:window_train].std())
    s     = float(mu_0) + k * sigma   # CUSUM slack

    S_pos = np.zeros(len(norm_z))
    for i in range(1, len(norm_z)):
        S_pos[i] = max(0, S_pos[i-1] + (norm_z.iloc[i] - s) / sigma)

    score = pd.Series(S_pos, index=df.index) / h
    alert = (S_pos > h).astype(int)
    return pd.DataFrame({"score": score, "alert": pd.Series(alert, index=df.index)})


def evaluate_detector(scores: pd.Series, labels: np.ndarray,
                      threshold: float = None, name: str = "Detector") -> dict:
    """Evaluate anomaly detector: F1, precision, recall, AUC-ROC."""
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                  roc_auc_score, average_precision_score)
    y_score = scores.values
    y_true  = labels[:len(y_score)]

    if threshold is None:
        threshold = np.percentile(y_score, 95)   # top 5% = anomaly

    y_pred = (y_score >= threshold).astype(int)
    result = {
        "F1":      round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Prec":    round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":  round(recall_score(y_true, y_pred, zero_division=0), 4),
        "AUC-ROC": round(roc_auc_score(y_true, y_score) if len(np.unique(y_true))>1 else 0.5, 4),
        "AP":      round(average_precision_score(y_true, y_score) if len(np.unique(y_true))>1 else 0.5, 4),
    }
    print(f"{name}: F1={result['F1']:.4f}, Prec={result['Prec']:.4f}, "
          f"Recall={result['Recall']:.4f}, AUC={result['AUC-ROC']:.4f}")
    return result
```

---

## 5. Isolation Forest (Layer 2)

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def isolation_forest_detector(
    df: pd.DataFrame,
    train_fraction: float = 0.5,
    window_size: int = 10,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """
    Isolation Forest on sliding windows of multivariate sensor data.

    Parameters
    ----------
    window_size   : number of time steps per sample (captures local context)
    contamination : expected fraction of anomalies in training data
    """
    # Create window features: (n_windows, window_size × n_sensors)
    D    = df.shape[1]
    n    = len(df)
    wins = []
    for i in range(window_size, n):
        wins.append(df.iloc[i-window_size:i].values.flatten())
    X = np.array(wins)

    train_end = int(len(X) * train_fraction)
    X_train = X[:train_end]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_all_s = scaler.transform(X)

    iforest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iforest.fit(X_tr_s)

    # Anomaly score: more negative = more anomalous
    raw_score  = -iforest.score_samples(X_all_s)
    normalized = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-12)

    # Pad to align with original df
    pad   = np.full(window_size, 0.0)
    score = np.concatenate([pad, normalized])[:n]

    return pd.DataFrame({
        "score": pd.Series(score, index=df.index),
        "alert": pd.Series((score > np.percentile(score, 95)).astype(int), index=df.index),
        "model": iforest,
        "scaler": scaler,
    })
```

---

## 6. LSTM Autoencoder (Layer 3)

```python
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for multivariate TS anomaly detection.
    Anomaly score = reconstruction MSE per time step.
    """

    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 2,
                 latent: int = 16):
        super().__init__()
        self.n_features = n_features
        self.encoder    = nn.LSTM(n_features, hidden, n_layers, batch_first=True)
        self.enc_proj   = nn.Linear(hidden, latent)
        self.dec_input  = nn.Linear(latent, hidden)
        self.decoder    = nn.LSTM(hidden, hidden, n_layers, batch_first=True)
        self.out        = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, D) → reconstructed: (batch, T, D)."""
        _, (h_n, _) = self.encoder(x)
        z           = self.enc_proj(h_n[-1])          # (B, latent)
        h_dec       = self.dec_input(z).unsqueeze(0)  # (1, B, hidden)
        h_dec       = h_dec.repeat(self.decoder.num_layers, 1, 1)
        dec_in      = h_dec[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        out, _      = self.decoder(dec_in, (h_dec, torch.zeros_like(h_dec)))
        return self.out(out)   # (B, T, D)


def train_lstm_ae(
    df_train: pd.DataFrame,
    seq_len:  int = 30,
    hidden:   int = 64,
    epochs:   int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> LSTMAutoencoder:
    """Train LSTM autoencoder on nominal (normal) sensor data."""
    from torch.utils.data import DataLoader, TensorDataset

    # Normalize
    scaler    = StandardScaler()
    X_norm    = scaler.fit_transform(df_train.values)
    sequences = [X_norm[i:i+seq_len] for i in range(len(X_norm)-seq_len)]
    X_tensor  = torch.tensor(np.array(sequences), dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True)
    model  = LSTMAutoencoder(n_features=df_train.shape[1], hidden=hidden)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train(); total = 0.0
        for (X_b,) in loader:
            out  = model(X_b)
            loss = nn.functional.mse_loss(out, X_b)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        sched.step()
        if (epoch+1) % 10 == 0:
            print(f"  LSTM-AE epoch {epoch+1}/{epochs} — Loss: {total/len(loader):.6f}")

    return model, scaler


def lstm_ae_score(
    model: LSTMAutoencoder,
    df:    pd.DataFrame,
    scaler,
    seq_len: int = 30,
) -> pd.Series:
    """Compute per-timestep reconstruction error (anomaly score)."""
    model.eval()
    X_norm = scaler.transform(df.values)
    scores = np.full(len(df), 0.0)

    with torch.no_grad():
        for i in range(len(X_norm) - seq_len):
            x   = torch.tensor(X_norm[i:i+seq_len], dtype=torch.float32).unsqueeze(0)
            out = model(x).squeeze(0).numpy()
            err = float(np.mean((X_norm[i:i+seq_len] - out)**2))
            scores[i + seq_len - 1] = err

    return pd.Series(scores, index=df.index)
```

---

## 7. Root Cause Analysis

```python
def channel_anomaly_scores(
    df:        pd.DataFrame,
    ae_model,
    scaler,
    seq_len:   int = 30,
    window:    int = 20,
) -> pd.DataFrame:
    """
    Per-channel reconstruction error for root cause identification.
    Returns (T, n_sensors) DataFrame: which channels are most anomalous.
    """
    ae_model.eval()
    X_norm = scaler.transform(df.values)
    D      = df.shape[1]
    scores = np.zeros((len(df), D))

    with torch.no_grad():
        for i in range(len(X_norm) - seq_len):
            x   = torch.tensor(X_norm[i:i+seq_len], dtype=torch.float32).unsqueeze(0)
            out = ae_model(x).squeeze(0).numpy()
            err = (X_norm[i:i+seq_len] - out)**2
            scores[i + seq_len - 1] = err.mean(axis=0)

    return pd.DataFrame(scores, columns=df.columns, index=df.index)


def identify_root_cause(
    channel_scores: pd.DataFrame,
    alert_time:     pd.Timestamp,
    lookback:       int = 20,
    top_k:          int = 3,
) -> list:
    """
    Identify top-k sensors responsible for the anomaly at alert_time.
    Returns list of (sensor_name, contribution_score).
    """
    window = channel_scores.loc[:alert_time].iloc[-lookback:]
    mean_score = window.mean()
    top        = mean_score.sort_values(ascending=False)[:top_k]
    result     = [(name, round(float(score), 6)) for name, score in top.items()]
    print(f"\nRoot cause (top-{top_k} channels at {alert_time}):")
    for name, score in result:
        print(f"  {name}: {score:.6f}")
    return result
```

---

## 8. Alert Pipeline & Monitoring

```python
def ensemble_anomaly_score(
    layer1_score: pd.Series,
    layer2_score: pd.Series,
    layer3_score: pd.Series,
    weights:      tuple = (0.25, 0.35, 0.40),
) -> pd.Series:
    """
    Weighted ensemble of three detection layers.
    All scores normalized to [0,1] before combining.
    """
    def normalize(s):
        mn = s.min(); mx = s.max()
        return (s - mn) / (mx - mn + 1e-12)

    s1 = normalize(layer1_score.reindex(layer3_score.index, fill_value=0))
    s2 = normalize(layer2_score.reindex(layer3_score.index, fill_value=0))
    s3 = normalize(layer3_score)
    return weights[0]*s1 + weights[1]*s2 + weights[2]*s3


class SensorAlertPipeline:
    """Production alert pipeline with cooldown and severity levels."""

    def __init__(self, threshold_warn=0.6, threshold_crit=0.85, cooldown_steps=30):
        self.thresh_warn = threshold_warn
        self.thresh_crit = threshold_crit
        self.cooldown    = cooldown_steps
        self._last_alert = -cooldown_steps
        self._step       = 0

    def process(self, score: float, timestamp, channel_scores=None) -> dict:
        self._step += 1
        in_cooldown = (self._step - self._last_alert) < self.cooldown

        if score >= self.thresh_crit and not in_cooldown:
            self._last_alert = self._step
            return {"level": "CRITICAL", "score": score, "timestamp": timestamp,
                    "action": "page on-call"}
        elif score >= self.thresh_warn and not in_cooldown:
            self._last_alert = self._step
            return {"level": "WARNING", "score": score, "timestamp": timestamp,
                    "action": "notify team"}
        return {"level": "NORMAL", "score": score, "timestamp": timestamp, "action": None}
```

---

## 9. Key Lessons

```
LESSON 1: Layer anomaly detectors by cost-to-compute.
  Fast statistical → ML → deep learning.
  Only invoke expensive layers when Layer 1 triggers.

LESSON 2: Train only on NORMAL data for autoencoders.
  If anomaly labels are scarce, unsupervised training on nominal data
  ensures the model learns a tight reconstruction of "normal".
  Anomalies produce high reconstruction error.

LESSON 3: Threshold selection drives precision/recall tradeoff.
  Top-5% threshold → high recall, lower precision.
  Adjust based on cost: false alarm (maintenance cost) vs. missed anomaly (failure cost).

LESSON 4: Multivariate > univariate for collective anomalies.
  A channel-by-channel z-score misses correlated multi-channel deviations.
  LSTM-AE captures cross-sensor temporal patterns.

LESSON 5: Root cause matters as much as detection.
  An alert without a channel diagnosis doubles repair time.
  Per-channel reconstruction error enables instant RCA.

LESSON 6: Class imbalance is extreme (< 5% anomaly).
  Avoid accuracy as a metric — it will be 95%+ by predicting all-normal.
  Use F1, precision-recall AUC, and F0.5 (precision-weighted) for evaluation.
```

---

*← [03 — Retail](./03_retail_sales_forecasting.md) | [Module README](./README.md) | Next: [05 — Patient Monitoring](./05_patient_monitoring_system.md) →*
