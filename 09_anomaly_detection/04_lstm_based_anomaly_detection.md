# 04 — LSTM-Based Anomaly Detection

> **Module**: 09 Anomaly Detection | **File**: 4 of 6
>
> LSTM predictive models treat anomaly detection as a **one-class classification problem via prediction error**: train an LSTM to predict the next value given history; if it predicts poorly at time t, t is anomalous. This complements autoencoders by detecting **sequence prediction failures** rather than reconstruction failures.

---

## Table of Contents

1. [Prediction Error as Anomaly Signal](#1-prediction-error-as-anomaly-signal)
2. [LSTM Forecasting Model for AD](#2-lstm-forecasting-model-for-ad)
3. [LSTMAD — Multivariate Anomaly Detection](#3-lstmad--multivariate-anomaly-detection)
4. [Comparing Prediction-Based vs. Reconstruction-Based AD](#4-comparing-prediction-based-vs-reconstruction-based-ad)
5. [Smoothed Error and Temporal Aggregation](#5-smoothed-error-and-temporal-aggregation)
6. [Threshold Calibration with Gaussian Mixture](#6-threshold-calibration-with-gaussian-mixture)
7. [Production Pipeline](#7-production-pipeline)

---

## 1. Prediction Error as Anomaly Signal

### 1.1 Intuition

```
Normal behavior: LSTM learns the temporal patterns well
  → predicted value ŷₜ ≈ yₜ → low prediction error eₜ = |yₜ - ŷₜ|

Anomalous behavior: at t, the value deviates from expected pattern
  → predicted value ŷₜ ≠ yₜ → high prediction error eₜ

Decision rule:
  eₜ > threshold  →  t is anomalous

Advantages over autoencoder:
  ✅ Simpler — standard LSTM forecasting model
  ✅ Interpretable — error is in original units
  ✅ Works well for point and contextual anomalies
  ✅ No window reconstruction needed — per-point scoring
```

### 1.2 Multi-Step Error Aggregation

```
Single-step LSTM:
  ŷₜ = f(y_{t-W}, ..., y_{t-1})
  eₜ = |yₜ - ŷₜ|

Multi-step LSTM (predicts next H steps):
  [ŷₜ₊₁, ..., ŷₜ₊ₕ] = f(y_{t-W}, ..., y_t)
  anomaly score for t = weighted sum of errors at all horizons

Multi-step is more robust (exploits multiple predictions at each point).
```

---

## 2. LSTM Forecasting Model for AD

### 2.1 Architecture and Training

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

class LSTMForecaster(nn.Module):
    """
    LSTM forecasting model for anomaly detection via prediction error.

    Trained on normal data to predict next h steps.
    At inference: large prediction error → anomaly.
    """

    def __init__(
        self,
        input_size:  int = 1,
        hidden_size: int = 64,
        n_layers:    int = 2,
        output_steps: int = 1,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.output_steps = output_steps

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        returns: (batch, output_steps)
        """
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # use last hidden state


def build_forecast_windows(
    series: np.ndarray,
    context_len: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (context, target) pairs for LSTM training.

    Returns
    -------
    X: (N, context_len, 1) — input sequences
    y: (N, horizon)        — target values
    """
    X, y = [], []
    for i in range(len(series) - context_len - horizon + 1):
        X.append(series[i:i + context_len])
        y.append(series[i + context_len:i + context_len + horizon])
    X = np.array(X, dtype=np.float32)[:, :, np.newaxis]  # (N, L, 1)
    y = np.array(y, dtype=np.float32)                     # (N, H)
    return X, y


def train_lstm_forecaster(
    train_series: np.ndarray,
    context_len: int = 50,
    horizon: int = 1,
    hidden_size: int = 64,
    n_layers: int = 2,
    n_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple:
    """
    Train LSTM forecaster on normal data.

    Returns
    -------
    (model, mu, sigma, context_len, horizon)
    """
    mu    = train_series.mean()
    sigma = train_series.std() + 1e-8
    norm  = (train_series - mu) / sigma

    X, y    = build_forecast_windows(norm, context_len, horizon)
    X_t     = torch.tensor(X)
    y_t     = torch.tensor(y)
    loader  = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model     = LSTMForecaster(1, hidden_size, n_layers, horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model.train()
    for epoch in range(n_epochs):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            y_hat  = model(xb)
            loss   = criterion(y_hat, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} — Loss: {total/len(loader):.6f}")

    return model, mu, sigma, context_len, horizon


def score_lstm_forecaster(
    model: LSTMForecaster,
    test_series: np.ndarray,
    mu: float,
    sigma: float,
    context_len: int,
    horizon: int = 1,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute per-step prediction error on test series.

    Returns
    -------
    error_score: (len(test_series),) — NaN for first context_len steps
    """
    norm = (test_series - mu) / sigma
    X, y = build_forecast_windows(norm, context_len, horizon)

    model.eval()
    with torch.no_grad():
        y_hat = model(torch.tensor(X).to(device)).cpu().numpy()

    # Map back to original scale and compute MAE
    y_true_orig = y * sigma + mu
    y_hat_orig  = y_hat * sigma + mu
    errors      = np.abs(y_true_orig - y_hat_orig).mean(axis=1)  # (N,) avg over horizon

    # Align to the time step where prediction is made (last context step)
    score = np.full(len(test_series), np.nan)
    for i, err in enumerate(errors):
        target_idx = i + context_len  # first predicted step
        if target_idx < len(test_series):
            score[target_idx] = err

    return score
```

---

## 3. LSTMAD — Multivariate Anomaly Detection

### 3.1 The LSTMAD Algorithm

LSTMAD (Malhotra et al., 2015) extends single-variate prediction error to **multivariate** time series:

```
1. Train LSTM to predict next h steps of ALL D features
2. At each time t, compute error vector:
     eₜ = [|y₁ₜ - ŷ₁ₜ|, |y₂ₜ - ŷ₂ₜ|, ..., |yDₜ - ŷDₜ|]  ∈ ℝᴰ

3. Model the error distribution on NORMAL data:
     eₜ ~ N(μₑ, Σₑ)   (fit Gaussian to training errors)

4. Anomaly score at test time:
     aₜ = (eₜ - μₑ)ᵀ Σₑ⁻¹ (eₜ - μₑ)   ← Mahalanobis distance

5. Alert: aₜ > χ²_{D, α}  (chi-squared threshold at confidence α)

Advantage: accounts for correlations between features in error space
```

### 3.2 Implementation

```python
import numpy as np
from scipy.stats import chi2

class LSTMADMultivariate:
    """
    LSTMAD: LSTM-based multivariate anomaly detection.
    Trains LSTM forecaster, models error as multivariate Gaussian,
    uses Mahalanobis distance as anomaly score.
    """

    def __init__(
        self,
        context_len: int = 30,
        horizon: int = 1,
        hidden_size: int = 64,
        n_layers: int = 2,
        n_epochs: int = 30,
        confidence: float = 0.99,
        device: str = "cpu",
    ):
        self.context_len = context_len
        self.horizon     = horizon
        self.confidence  = confidence
        self.device      = device
        self._model_args = dict(
            hidden_size=hidden_size, n_layers=n_layers, n_epochs=n_epochs
        )

    def fit(self, df_train: pd.DataFrame) -> "LSTMADMultivariate":
        """
        Fit one LSTM per feature (or use a shared multi-output LSTM).
        Model the training error distribution.
        """
        self.D       = df_train.shape[1]
        self.cols    = df_train.columns.tolist()
        self.models_ = {}
        self.params_ = {}

        all_errors = []
        for col in self.cols:
            series = df_train[col].values
            model, mu, sigma, _, _ = train_lstm_forecaster(
                series, self.context_len, self.horizon, **self._model_args,
                device=self.device,
            )
            self.models_[col] = model
            self.params_[col] = (mu, sigma)

            errors = score_lstm_forecaster(
                model, series, mu, sigma, self.context_len,
                self.horizon, self.device,
            )
            all_errors.append(errors[~np.isnan(errors)])

        # Fit multivariate Gaussian to training errors
        min_len = min(len(e) for e in all_errors)
        E       = np.column_stack([e[:min_len] for e in all_errors])  # (T, D)
        self._mu_e    = E.mean(axis=0)
        self._cov_e   = np.cov(E.T) + 1e-6 * np.eye(self.D)
        self._cov_inv = np.linalg.pinv(self._cov_e)
        self._threshold = chi2.ppf(self.confidence, df=self.D)
        return self

    def predict(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """Score each time step in test set."""
        error_cols = []
        for col in self.cols:
            mu, sigma = self.params_[col]
            errors = score_lstm_forecaster(
                self.models_[col], df_test[col].values,
                mu, sigma, self.context_len, self.horizon, self.device,
            )
            error_cols.append(errors)

        E      = np.column_stack(error_cols)  # (T, D)
        scores = np.full(len(df_test), np.nan)

        for t in range(self.context_len, len(df_test)):
            e  = E[t] - self._mu_e
            md = float(e @ self._cov_inv @ e)   # Mahalanobis distance
            scores[t] = md

        return pd.DataFrame({
            "mahalanobis_score": scores,
            "anomaly":           scores > self._threshold,
        }, index=df_test.index)
```

---

## 4. Comparing Prediction-Based vs. Reconstruction-Based AD

| Dimension                | LSTM Prediction Error     | LSTM Autoencoder              |
|--------------------------|---------------------------|-------------------------------|
| **Signal**               | Can't predict next value   | Can't reconstruct window      |
| **Anomaly types**         | Point, contextual          | All (incl. collective)        |
| **Labeling required**     | No                        | No                            |
| **Training complexity**   | Medium                    | Medium-High                   |
| **Score interpretability**| High (original units)     | Medium (reconstruction MSE)   |
| **Collective detection**  | Weak                      | Strong                        |
| **Flatline detection**    | Excellent (predicts motion)| Poor (flatline reconstructs well)|
| **Practical preference**  | Spikes, sensor outliers    | Waveform shape changes        |

### 4.1 Ensemble Recommendation

```python
def combined_anomaly_score(
    pred_scores: np.ndarray,
    recon_scores: np.ndarray,
    pred_weight: float = 0.5,
) -> np.ndarray:
    """
    Weighted combination of prediction-error and reconstruction-error scores.
    Both must be pre-normalized to [0, 1] range before combining.
    """
    from scipy.stats import rankdata

    # Rank-normalize each score to [0, 1]
    n = len(pred_scores)
    valid = ~(np.isnan(pred_scores) | np.isnan(recon_scores))

    combined = np.full(n, np.nan)
    combined[valid] = (
        pred_weight       * rankdata(pred_scores[valid]) / valid.sum() +
        (1 - pred_weight) * rankdata(recon_scores[valid]) / valid.sum()
    )
    return combined
```

---

## 5. Smoothed Error and Temporal Aggregation

### 5.1 Smoothing Motivation

Single-step prediction error is noisy. Real anomalies tend to span multiple consecutive steps:

```python
def smooth_anomaly_scores(
    scores: np.ndarray,
    window: int = 5,
    method: str = "ewm",
) -> np.ndarray:
    """
    Smooth anomaly scores to reduce false positives from single-step noise.

    Parameters
    ----------
    window : smoothing window / EWM span
    method : 'rolling_mean', 'rolling_max', or 'ewm' (exponential weighted mean)
    """
    s = pd.Series(scores)

    if method == "rolling_mean":
        return s.rolling(window, min_periods=1, center=True).mean().values
    elif method == "rolling_max":
        return s.rolling(window, min_periods=1, center=True).max().values
    elif method == "ewm":
        return s.ewm(span=window, adjust=False).mean().values
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
```

### 5.2 Anomaly Region Grouping

```python
def group_anomaly_regions(
    anomaly_flags: np.ndarray,
    min_gap: int = 3,
) -> list[tuple[int, int]]:
    """
    Group consecutive anomalous time steps into regions.
    Merges anomaly segments separated by fewer than min_gap normal steps.

    Returns
    -------
    list of (start_idx, end_idx) tuples
    """
    flags  = np.asarray(anomaly_flags, dtype=bool)
    n      = len(flags)
    in_region = False
    start     = 0
    regions   = []

    # Fill small gaps
    for t in range(n):
        if flags[t]:
            if not in_region:
                start     = t
                in_region = True
        else:
            if in_region:
                # Check if gap to next anomaly is small enough to merge
                next_anomaly = np.where(flags[t:])[0]
                if len(next_anomaly) > 0 and next_anomaly[0] < min_gap:
                    continue  # fill the gap
                regions.append((start, t - 1))
                in_region = False

    if in_region:
        regions.append((start, n - 1))

    return regions
```

---

## 6. Threshold Calibration with Gaussian Mixture

### 6.1 Bimodal Error Distribution

Prediction errors often have a bimodal distribution:
- Mode 1: small errors from normal time steps
- Mode 2: large errors from anomalous time steps

Fitting a **Gaussian Mixture Model (GMM)** separates these modes automatically:

```python
from sklearn.mixture import GaussianMixture

def gmm_threshold(
    train_errors: np.ndarray,
    n_components: int = 2,
    anomaly_component: str = "higher_mean",
) -> float:
    """
    Fit GMM to training errors; return threshold between normal/anomaly components.

    Parameters
    ----------
    n_components      : number of Gaussian components (2 = normal + anomaly)
    anomaly_component : 'higher_mean' assumes anomaly component has larger mean

    Returns
    -------
    threshold : float — intersection of two Gaussians (decision boundary)
    """
    errors = train_errors[~np.isnan(train_errors)].reshape(-1, 1)

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(errors)

    means     = gmm.means_.flatten()
    stds      = np.sqrt(gmm.covariances_.flatten())
    weights   = gmm.weights_

    # Sort components by mean
    order     = np.argsort(means)
    means     = means[order]
    stds      = stds[order]
    weights   = weights[order]

    # Decision boundary: where log-likelihoods of two components are equal
    # Approximate as: midpoint weighted by inverse std
    threshold = (means[0] / stds[0] + means[-1] / stds[-1]) / (1/stds[0] + 1/stds[-1])

    print(f"GMM components:")
    for i in range(n_components):
        print(f"  Component {i}: mean={means[i]:.4f}, std={stds[i]:.4f}, weight={weights[i]:.3f}")
    print(f"Decision threshold: {threshold:.4f}")

    return float(threshold)
```

---

## 7. Production Pipeline

```python
class LSTMADPipeline:
    """
    Production LSTM prediction-error anomaly detection pipeline.
    Supports single-variate series; extend to multivariate via LSTMADMultivariate.
    """

    def __init__(
        self,
        context_len: int = 50,
        horizon: int = 1,
        hidden_size: int = 64,
        n_layers: int = 2,
        n_epochs: int = 30,
        target_fpr: float = 0.01,
        smooth_window: int = 5,
        device: str = "cpu",
    ):
        self.context_len  = context_len
        self.horizon      = horizon
        self.n_epochs     = n_epochs
        self.target_fpr   = target_fpr
        self.smooth_window = smooth_window
        self.device       = device
        self._model_args  = dict(
            hidden_size=hidden_size, n_layers=n_layers,
            n_epochs=n_epochs, device=device,
        )

    def fit(self, series: np.ndarray) -> "LSTMADPipeline":
        """Fit on normal training data and calibrate threshold."""
        self._model, self._mu, self._sigma, self._ctx, self._h = \
            train_lstm_forecaster(series, self.context_len, self.horizon,
                                   **self._model_args)

        train_scores = score_lstm_forecaster(
            self._model, series, self._mu, self._sigma,
            self._ctx, self._h, self.device,
        )
        valid        = train_scores[~np.isnan(train_scores)]
        self._threshold = float(np.quantile(valid, 1 - self.target_fpr))
        print(f"Threshold: {self._threshold:.4f} (FPR target: {self.target_fpr:.1%})")
        return self

    def predict(self, series: np.ndarray, smooth: bool = True) -> pd.DataFrame:
        """Score and classify test series."""
        scores = score_lstm_forecaster(
            self._model, series, self._mu, self._sigma,
            self._ctx, self._h, self.device,
        )
        if smooth:
            scores = smooth_anomaly_scores(scores, self.smooth_window, "ewm")

        anomaly = scores > self._threshold
        regions = group_anomaly_regions(anomaly)

        return pd.DataFrame({
            "value":   series,
            "score":   scores,
            "anomaly": anomaly,
        }), regions
```

---

*← [03 — Autoencoder Detection](./03_autoencoder_anomaly_detection.md) | [Module README](./README.md) | Next: [05 — Online Detection](./05_online_anomaly_detection.md) →*
