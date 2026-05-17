# 03 — Autoencoder Anomaly Detection

> **Module**: 09 Anomaly Detection | **File**: 3 of 6
>
> Autoencoders learn to compress and reconstruct normal data. When presented with anomalies, they fail to reconstruct them well — producing high reconstruction error that serves as the anomaly score. This note covers dense autoencoders, LSTM autoencoders, Variational Autoencoders (VAE), and threshold selection strategies.

---

## Table of Contents

1. [The Reconstruction Error Principle](#1-the-reconstruction-error-principle)
2. [Dense Autoencoder for Time Series Windows](#2-dense-autoencoder-for-time-series-windows)
3. [LSTM Autoencoder](#3-lstm-autoencoder)
4. [Variational Autoencoder (VAE)](#4-variational-autoencoder-vae)
5. [Threshold Selection](#5-threshold-selection)
6. [Multivariate Anomaly Detection](#6-multivariate-anomaly-detection)
7. [Production Pipeline](#7-production-pipeline)

---

## 1. The Reconstruction Error Principle

### 1.1 Core Idea

```
Training Phase (on normal data only):
  Input x → Encoder → z (latent code) → Decoder → x̂ (reconstruction)
  Loss = reconstruction_error(x, x̂) → minimized for NORMAL patterns

Inference Phase:
  Normal  data: x → [AE] → x̂ ≈ x    → low reconstruction error  → not anomaly
  Anomalous data: x → [AE] → x̂ ≠ x  → high reconstruction error → anomaly

Key: The autoencoder "forgets" how to reconstruct anomalies because
     it never saw them during training.
```

### 1.2 Reconstruction Error Metrics

```
For a window xₜ = [xₜ₋W, ..., xₜ]:

Point-wise MSE:   E(t) = (1/W) Σ (xₜ₋ᵢ - x̂ₜ₋ᵢ)²
Point-wise MAE:   E(t) = (1/W) Σ |xₜ₋ᵢ - x̂ₜ₋ᵢ|
Max error:        E(t) = max_i |xₜ₋ᵢ - x̂ₜ₋ᵢ|     (sensitive to spikes)

MSE/MAE → flag systematic reconstruction failure (collective anomaly)
Max error → flag any single-point deviation within window
```

### 1.3 Window Construction

```
Sliding window approach:
  Input:  [x_{t-W+1}, x_{t-W+2}, ..., x_t]    shape: (W,)
  Target: same as input (self-supervised)
  Stride: s (overlap = W - s; smaller s = more overlapping windows)

Output: reconstruction error for each window
        → assign to central or final time step
```

---

## 2. Dense Autoencoder for Time Series Windows

### 2.1 Architecture

```
Input: W-dimensional window
  ↓
Dense(W → W/2, ReLU)
  ↓
Dense(W/2 → W/4, ReLU)   ← bottleneck (latent code)
  ↓
Dense(W/4 → W/2, ReLU)
  ↓
Dense(W/2 → W, Linear)   ← reconstruction
  ↓
Output: reconstructed window

Training loss: MSE(input, output)
```

### 2.2 Implementation (PyTorch)

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

class DenseAutoencoder(nn.Module):
    """Dense autoencoder for time series window reconstruction."""

    def __init__(self, window_size: int, bottleneck: int = None):
        super().__init__()
        if bottleneck is None:
            bottleneck = max(2, window_size // 4)

        self.encoder = nn.Sequential(
            nn.Linear(window_size, window_size // 2),
            nn.ReLU(),
            nn.Linear(window_size // 2, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, window_size // 2),
            nn.ReLU(),
            nn.Linear(window_size // 2, window_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def build_windows(series: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """
    Convert a 1D series into a 2D matrix of sliding windows.

    Returns (N_windows, window_size)
    """
    windows = []
    for i in range(0, len(series) - window_size + 1, stride):
        windows.append(series[i:i + window_size])
    return np.array(windows, dtype=np.float32)


def train_autoencoder(
    train_series: np.ndarray,
    window_size: int = 30,
    bottleneck: int = None,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[DenseAutoencoder, np.ndarray]:
    """
    Train a dense autoencoder on normal time series data.

    Returns
    -------
    (trained_model, train_reconstruction_errors)
    """
    # Build windows from training series
    windows = build_windows(train_series, window_size)

    # Normalize to [0, 1] per feature (across training data)
    mu    = windows.mean(axis=0, keepdims=True)
    sigma = windows.std(axis=0, keepdims=True) + 1e-8

    windows_norm = (windows - mu) / sigma

    X_tensor = torch.tensor(windows_norm)
    loader   = DataLoader(TensorDataset(X_tensor, X_tensor),
                          batch_size=batch_size, shuffle=True)

    model     = DenseAutoencoder(window_size, bottleneck).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            x_hat   = model(x_batch)
            loss    = criterion(x_hat, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} — Loss: {total_loss/len(loader):.6f}")

    # Compute per-window reconstruction error on training set
    model.eval()
    with torch.no_grad():
        x_hat  = model(torch.tensor(windows_norm))
        errors = ((torch.tensor(windows_norm) - x_hat)**2).mean(dim=1).numpy()

    return model, errors, mu, sigma, window_size


def score_autoencoder(
    model: DenseAutoencoder,
    test_series: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    window_size: int,
    stride: int = 1,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute per-window reconstruction error on test series.
    Returns an error score for each window (aligned to window end index).
    """
    windows      = build_windows(test_series, window_size, stride)
    windows_norm = (windows - mu) / sigma

    model.eval()
    with torch.no_grad():
        x_hat  = model(torch.tensor(windows_norm).to(device))
        errors = ((torch.tensor(windows_norm).to(device) - x_hat)**2).mean(dim=1).cpu().numpy()

    # Align errors to end of each window
    end_indices = np.arange(window_size - 1, len(test_series), stride)[:len(errors)]
    error_series = np.full(len(test_series), np.nan)
    for i, idx in enumerate(end_indices):
        if idx < len(test_series):
            error_series[idx] = errors[i]

    return error_series
```

---

## 3. LSTM Autoencoder

### 3.1 Why LSTM?

Dense autoencoders treat windows as flat vectors — they lose temporal order. LSTM autoencoders process windows **sequentially**, capturing temporal dependencies:

```
LSTM Encoder:
  [x₁, x₂, ..., xW] → LSTM → hidden state hW = context vector

LSTM Decoder:
  hW → LSTM → [x̂₁, x̂₂, ..., x̂W] (reconstructed)

Advantage: naturally captures temporal structure
           → better at detecting collective anomalies (pattern breaks)
```

### 3.2 Implementation

```python
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """
    LSTM sequence-to-sequence autoencoder for time series.

    Encodes the sequence into a fixed-size context vector,
    then decodes back to the original sequence length.
    """

    def __init__(
        self,
        input_size:  int = 1,
        hidden_size: int = 64,
        n_layers:    int = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,   # decoder input = repeated context
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        returns: (batch, seq_len, input_size) — reconstructed sequence
        """
        batch_size, seq_len, _ = x.shape

        # Encode: use last hidden state as context
        _, (h_n, c_n) = self.encoder(x)

        # Decode: repeat context vector for each step
        context = h_n[-1].unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq, hidden)
        out, _  = self.decoder(context, (h_n, c_n))
        return self.output_layer(out)


def train_lstm_autoencoder(
    train_series: np.ndarray,
    window_size: int = 50,
    hidden_size: int = 64,
    n_layers: int = 2,
    n_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple:
    """Train LSTM autoencoder on normal windows."""
    # Normalize
    mu, sigma = train_series.mean(), train_series.std() + 1e-8
    norm      = (train_series - mu) / sigma

    windows = build_windows(norm, window_size)
    X       = torch.tensor(windows[:, :, np.newaxis], dtype=torch.float32)  # (N, W, 1)
    loader  = DataLoader(TensorDataset(X, X), batch_size=batch_size, shuffle=True)

    model     = LSTMAutoencoder(input_size=1, hidden_size=hidden_size, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        model.train()
        total = 0.0
        for xb, _ in loader:
            xb    = xb.to(device)
            x_hat = model(xb)
            loss  = criterion(x_hat, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} — Loss: {total/len(loader):.6f}")

    return model, mu, sigma, window_size


def score_lstm_autoencoder(
    model: LSTMAutoencoder,
    test_series: np.ndarray,
    mu: float,
    sigma: float,
    window_size: int,
    device: str = "cpu",
) -> np.ndarray:
    """Score test series: per-step max reconstruction error."""
    norm    = (test_series - mu) / sigma
    windows = build_windows(norm, window_size)
    X       = torch.tensor(windows[:, :, np.newaxis], dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        x_hat = model(X)

    errors = ((X - x_hat).squeeze(-1)**2).cpu().numpy()  # (N_windows, W)

    # For each time step, take the max error across all windows it belongs to
    score = np.full(len(test_series), np.nan)
    for i, window_errors in enumerate(errors):
        for j, err in enumerate(window_errors):
            t = i + j
            if t < len(score):
                score[t] = err if np.isnan(score[t]) else max(score[t], err)

    return score
```

---

## 4. Variational Autoencoder (VAE)

### 4.1 VAE vs. Standard AE

```
Standard AE:
  Encoder → z (deterministic point in latent space)
  → Reconstruction error only

VAE:
  Encoder → (μ, σ) (parameters of latent Gaussian)
  → Sample z ~ N(μ, σ²)
  → Decoder reconstructs from z

VAE anomaly score = ELBO loss = Reconstruction Loss + KL Divergence
  KL(q(z|x) || N(0,1))  — measures how far from normal prior

Advantage: uncertainty-aware anomaly scoring
           High KL term → unusual encoding → anomaly
```

### 4.2 Implementation

```python
class VAE(nn.Module):
    """
    Variational Autoencoder for time series windows.
    Uses ELBO (reconstruction + KL) as anomaly score.
    """

    def __init__(self, window_size: int, latent_dim: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        h = max(16, window_size // 2)

        # Encoder outputs mean and log-variance of q(z|x)
        self.encoder_shared = nn.Sequential(
            nn.Linear(window_size, h), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(h, latent_dim)
        self.fc_logvar = nn.Linear(h, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h), nn.ReLU(),
            nn.Linear(h, window_size),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h      = self.encoder_shared(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + ε·σ, ε ~ N(0,1)."""
        if self.training:
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(0.5 * logvar)
        return mu   # use mean at inference for deterministic scoring

    def forward(self, x: torch.Tensor) -> tuple:
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_hat      = self.decoder(z)
        return x_hat, mu, logvar

    def elbo_loss(self, x: torch.Tensor, x_hat: torch.Tensor,
                  mu: torch.Tensor, logvar: torch.Tensor,
                  beta: float = 1.0) -> torch.Tensor:
        """
        ELBO = -E[log p(x|z)] + β·KL[q(z|x) || p(z)]

        beta > 1: stronger disentanglement (β-VAE)
        """
        recon = nn.functional.mse_loss(x_hat, x, reduction="sum")
        kl    = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return recon + beta * kl

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Score = per-sample reconstruction MSE + KL divergence."""
        self.eval()
        with torch.no_grad():
            x_hat, mu, logvar = self(x)
            recon = ((x - x_hat)**2).mean(dim=1)
            kl    = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(dim=1)
        return recon + kl
```

---

## 5. Threshold Selection

### 5.1 Strategies

```python
import numpy as np
from scipy.stats import norm

def select_threshold(
    train_errors: np.ndarray,
    method: str = "percentile",
    target_fpr: float = 0.01,
    n_sigma: float = 3.0,
) -> float:
    """
    Select anomaly threshold from training-set reconstruction errors.

    Methods
    -------
    percentile  : threshold = (1 - target_fpr) percentile of train errors
    sigma       : threshold = mean + n_sigma * std (Gaussian assumption)
    extreme     : threshold = mean + n_sigma * std of log(errors) (heavy tail)

    Returns
    -------
    threshold : float — flag test errors above this as anomalies
    """
    train_errors = train_errors[~np.isnan(train_errors)]

    if method == "percentile":
        return float(np.quantile(train_errors, 1 - target_fpr))

    elif method == "sigma":
        mu, sigma = train_errors.mean(), train_errors.std()
        return float(mu + n_sigma * sigma)

    elif method == "extreme":
        # Model log(errors) as normal — handles heavy tail better
        log_e = np.log(train_errors + 1e-12)
        mu, sigma = log_e.mean(), log_e.std()
        return float(np.exp(mu + n_sigma * sigma))

    raise ValueError(f"Unknown method: {method}")
```

---

## 6. Multivariate Anomaly Detection

### 6.1 Multivariate Window Feature Matrix

```python
def build_multivariate_windows(
    df: pd.DataFrame,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """
    Build sliding windows from multivariate time series.

    Parameters
    ----------
    df          : (T, D) DataFrame — T timesteps, D features
    window_size : W
    stride      : step between windows

    Returns
    -------
    (N_windows, W, D) numpy array
    """
    D       = df.shape[1]
    values  = df.values.astype(np.float32)
    windows = []
    for i in range(0, len(values) - window_size + 1, stride):
        windows.append(values[i:i + window_size])
    return np.array(windows)   # (N, W, D)


class MultivariateAE(nn.Module):
    """LSTM autoencoder for multivariate time series (D features)."""

    def __init__(self, n_features: int, hidden_size: int = 64, n_layers: int = 2):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, n_layers,
                                batch_first=True, dropout=0.1)
        self.decoder = nn.LSTM(hidden_size, hidden_size, n_layers,
                                batch_first=True, dropout=0.1)
        self.output  = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.encoder(x)
        ctx = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(ctx, (h, c))
        return self.output(out)
```

---

## 7. Production Pipeline

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class AutoencoderAnomalyPipeline:
    """
    Production pipeline: fit LSTM-AE on normal data, score new windows,
    calibrate threshold, expose predict() for deployment.
    """

    def __init__(
        self,
        window_size: int = 50,
        hidden_size: int = 64,
        n_layers: int = 2,
        target_fpr: float = 0.01,
        n_epochs: int = 30,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.window_size = window_size
        self.target_fpr  = target_fpr
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.device      = device
        self._model_args = dict(hidden_size=hidden_size, n_layers=n_layers)
        self._fitted     = False

    def fit(self, series: np.ndarray) -> "AutoencoderAnomalyPipeline":
        self._mu    = series.mean()
        self._sigma = series.std() + 1e-8
        norm        = (series - self._mu) / self._sigma

        self._model, self._mu, self._sigma, _ = train_lstm_autoencoder(
            series, self.window_size, **self._model_args,
            n_epochs=self.n_epochs, batch_size=self.batch_size,
            device=self.device,
        )

        # Compute train errors for threshold calibration
        train_scores     = score_lstm_autoencoder(
            self._model, series, self._mu, self._sigma,
            self.window_size, self.device,
        )
        valid_scores     = train_scores[~np.isnan(train_scores)]
        self._threshold  = float(np.quantile(valid_scores, 1 - self.target_fpr))
        self._fitted     = True
        print(f"Threshold calibrated: {self._threshold:.6f} (FPR target: {self.target_fpr:.1%})")
        return self

    def predict(self, series: np.ndarray) -> pd.DataFrame:
        assert self._fitted, "Call .fit() first."
        scores = score_lstm_autoencoder(
            self._model, series, self._mu, self._sigma,
            self.window_size, self.device,
        )
        return pd.DataFrame({
            "value":    series,
            "score":    scores,
            "anomaly":  scores > self._threshold,
        })
```

---

*← [02 — Isolation Forest](./02_isolation_forest_for_ts.md) | [Module README](./README.md) | Next: [04 — LSTM-Based Detection](./04_lstm_based_anomaly_detection.md) →*
