# 04 — Deep Learning Classification

> **Module**: 10 Classification & Clustering | **File**: 4 of 5
>
> Deep learning classifiers learn hierarchical representations directly from raw time series. FCN, ResNet, and InceptionTime are the dominant architectures for time series classification, consistently achieving state-of-the-art accuracy on the UCR benchmark while being fully end-to-end trainable.

---

## Table of Contents

1. [Why Deep Learning for TSC](#1-why-deep-learning-for-tsc)
2. [Fully Convolutional Network (FCN)](#2-fully-convolutional-network-fcn)
3. [ResNet for Time Series](#3-resnet-for-time-series)
4. [InceptionTime](#4-inceptiontime)
5. [LSTM Classifier](#5-lstm-classifier)
6. [Training Best Practices](#6-training-best-practices)
7. [Transfer Learning and Pre-Training](#7-transfer-learning-and-pre-training)
8. [Production Pipeline](#8-production-pipeline)

---

## 1. Why Deep Learning for TSC

### 1.1 Advantages Over Traditional Methods

```
Traditional (DTW, ROCKET, tsfresh):
  ✅ Fast to train and run
  ✅ Works with very small datasets
  ✅ No GPU needed
  ❌ Fixed feature extractors (not learned)
  ❌ Limited to univariate or simple multivariate

Deep learning:
  ✅ Learns task-specific representations end-to-end
  ✅ Excellent for multivariate, multi-scale patterns
  ✅ Transfer learning from pre-trained models
  ✅ State-of-the-art on large datasets
  ❌ Needs more data (typically N > 500)
  ❌ Hyperparameter tuning required
  ❌ GPU training recommended for large T or N
```

### 1.2 Architecture Comparison

| Model          | Depth | Key Feature              | Best For                        |
|----------------|-------|--------------------------|----------------------------------|
| FCN            | 3     | Global average pooling   | General TSC, baseline            |
| ResNet-TS      | 11    | Residual connections     | Long series, complex patterns    |
| InceptionTime  | 6     | Multi-scale convolutions | Best overall, ensemble benefit   |
| LSTM           | 2-4   | Sequential hidden state  | Long-range dependencies          |
| Transformer-TS | 4-12  | Self-attention           | Very long series, multivariate   |

---

## 2. Fully Convolutional Network (FCN)

### 2.1 Architecture

```
Input: (batch, T, 1)  [univariate] or (batch, T, D) [multivariate]

Layer 1: Conv1D(128, kernel=8) → BN → ReLU
Layer 2: Conv1D(256, kernel=5) → BN → ReLU
Layer 3: Conv1D(128, kernel=3) → BN → ReLU

Global Average Pooling: (batch, T, 128) → (batch, 128)
  → Reduces any-length output to fixed 128-dim vector
  → No FC layers → no need for fixed T at test time

Classifier: Dense(n_classes) → Softmax

Key design choice: Global Average Pooling instead of Flatten
  → Works on variable-length series
  → Strong regularization effect (much less overfitting than Flatten)
```

### 2.2 Implementation (PyTorch)

```python
import torch
import torch.nn as nn
import numpy as np

class FCN(nn.Module):
    """
    Fully Convolutional Network for time series classification.
    Wang et al. (2017) — baseline deep learning TSC model.

    Input:  (batch, T, n_channels) — batch of time series
    Output: (batch, n_classes) — class logits
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 2):
        super().__init__()

        self.layers = nn.Sequential(
            # Block 1: wide kernel to capture long-range patterns
            nn.Conv1d(n_channels, 128, kernel_size=8, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Block 2: medium kernel
            nn.Conv1d(128, 256, kernel_size=5, padding="same"),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # Block 3: narrow kernel for fine details
            nn.Conv1d(256, 128, kernel_size=3, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, T, n_channels)
        """
        # Conv1d expects (batch, channels, length)
        x = x.permute(0, 2, 1)        # (batch, n_channels, T)
        x = self.layers(x)             # (batch, 128, T)
        x = x.mean(dim=2)             # Global Average Pooling → (batch, 128)
        return self.classifier(x)      # (batch, n_classes)
```

---

## 3. ResNet for Time Series

### 3.1 Architecture

```
ResNet-TS (Wang et al., 2017) uses residual connections to allow
gradient flow through deep networks:

Input → [Res Block 1] → [Res Block 2] → [Res Block 3] → GAP → Dense

Each Residual Block:
  Input x
  ├── Main path: Conv(64, k=8) → BN → ReLU → Conv(64, k=5) → BN → ReLU → Conv(64, k=3) → BN
  └── Skip path: Conv(64, k=1) → BN   [matches channels if needed]
  Output: ReLU(main + skip)

Depth: 3 residual blocks × 3 conv layers = 9 conv + 1 output = 11 layers
```

### 3.2 Implementation

```python
class ResBlock1D(nn.Module):
    """Residual block for 1D time series."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 8, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 5, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding="same"),
            nn.BatchNorm1d(out_channels),
        )

        # Projection shortcut if channel dimensions differ
        self.skip = (
            nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, padding="same"),
                           nn.BatchNorm1d(out_channels))
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.main(x) + self.skip(x))


class ResNetTS(nn.Module):
    """
    ResNet for Time Series Classification.
    Wang et al. (2017). Standard 3-block architecture.

    Input:  (batch, T, n_channels)
    Output: (batch, n_classes)
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 2):
        super().__init__()

        self.resblocks = nn.Sequential(
            ResBlock1D(n_channels, 64),
            ResBlock1D(64, 128),
            ResBlock1D(128, 128),
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)       # (batch, n_channels, T)
        x = self.resblocks(x)         # (batch, 128, T)
        x = x.mean(dim=2)            # GAP → (batch, 128)
        return self.classifier(x)
```

---

## 4. InceptionTime

### 4.1 Architecture

**InceptionTime** (Fawaz et al., 2020) is an ensemble of 5 Inception-style networks. It is the current state-of-the-art deep learning TSC model:

```
Each Inception Module:
  Input x
  ├── Path 1: Bottleneck Conv1D(32, k=1) → Conv1D(32, k=39)
  ├── Path 2: Bottleneck Conv1D(32, k=1) → Conv1D(32, k=19)
  ├── Path 3: Bottleneck Conv1D(32, k=1) → Conv1D(32, k=9)
  └── Path 4: MaxPool1D(3) → Conv1D(32, k=1)
  → Concatenate → (batch, 128, T) → BN → ReLU

Full network: 3 Inception Modules + Residual shortcut every 3 modules
             + GAP → Dense → Softmax

Key idea: Multi-scale convolution at different kernel sizes
          (9, 19, 39) = captures patterns at short, medium, long scales simultaneously

Ensemble: 5 independently trained networks → majority vote
          → +2–5% accuracy improvement over single model
```

### 4.2 Implementation

```python
class InceptionModule(nn.Module):
    """
    Inception module for time series — multi-scale temporal feature extraction.
    """

    def __init__(self, in_channels: int, n_filters: int = 32, bottleneck: int = 32):
        super().__init__()

        # Bottleneck to reduce channels before large kernels
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, 1, padding="same")

        # Three parallel convolutions at different scales
        self.conv_large  = nn.Conv1d(bottleneck, n_filters, kernel_size=39, padding="same")
        self.conv_medium = nn.Conv1d(bottleneck, n_filters, kernel_size=19, padding="same")
        self.conv_small  = nn.Conv1d(bottleneck, n_filters, kernel_size=9,  padding="same")

        # Max pooling path — residual feature
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_mp = nn.Conv1d(in_channels, n_filters, kernel_size=1, padding="same")

        self.bn   = nn.BatchNorm1d(n_filters * 4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        btn  = self.bottleneck(x)
        out  = torch.cat([
            self.conv_large(btn),
            self.conv_medium(btn),
            self.conv_small(btn),
            self.conv_mp(self.maxpool(x)),
        ], dim=1)
        return self.relu(self.bn(out))


class InceptionTime(nn.Module):
    """
    InceptionTime: 3 Inception modules with residual shortcut every 3 blocks.
    Single network (use ensemble of 5 for best accuracy).

    Input:  (batch, T, n_channels)
    Output: (batch, n_classes)
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 2, n_filters: int = 32):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        n_out           = n_filters * 4   # 128

        # 3 inception modules
        self.inc1 = InceptionModule(n_channels, n_filters)
        self.inc2 = InceptionModule(n_out, n_filters)
        self.inc3 = InceptionModule(n_out, n_filters)

        # Residual shortcut
        self.shortcut = nn.Sequential(
            nn.Conv1d(n_channels, n_out, 1, padding="same"),
            nn.BatchNorm1d(n_out),
        )
        self.relu     = nn.ReLU()

        # Classifier
        self.classifier = nn.Linear(n_out, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = x.permute(0, 2, 1)   # (batch, n_channels, T)
        res = self.shortcut(x)

        out = self.inc1(x)
        out = self.inc2(out)
        out = self.inc3(out)

        out = self.relu(out + res)   # residual addition
        out = out.mean(dim=2)        # GAP
        return self.classifier(out)


class InceptionTimeEnsemble(nn.Module):
    """Ensemble of 5 InceptionTime networks — majority vote."""

    def __init__(self, n_channels, n_classes, n_members=5):
        super().__init__()
        self.members = nn.ModuleList([
            InceptionTime(n_channels, n_classes) for _ in range(n_members)
        ])

    def forward(self, x):
        logits = torch.stack([m(x) for m in self.members], dim=0)
        return logits.mean(dim=0)   # average logits (soft voting)
```

---

## 5. LSTM Classifier

### 5.1 Architecture

```
LSTM captures long-range temporal dependencies through its gated recurrent structure:

Input → LSTM(hidden=128, layers=2, dropout=0.2) → [last hidden state] → Dense → Softmax

Variants:
  1. Last hidden state:    use h_T (classic)
  2. Max/Mean pooling:     pool over all h₁, ..., hT (better for long series)
  3. Attention pooling:    weighted sum of h₁, ..., hT (most expressive)
```

### 5.2 Implementation

```python
class LSTMClassifier(nn.Module):
    """
    LSTM-based time series classifier.

    Input:  (batch, T, n_channels)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        n_channels:  int = 1,
        n_classes:   int = 2,
        hidden_size: int = 128,
        n_layers:    int = 2,
        dropout:     float = 0.2,
        pooling:     str = "last",   # 'last', 'mean', 'attention'
    ):
        super().__init__()
        self.pooling = pooling

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )
        lstm_out = hidden_size * 2   # bidirectional

        if pooling == "attention":
            self.attn = nn.Linear(lstm_out, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, n_channels)"""
        out, (h_n, _) = self.lstm(x)   # out: (batch, T, 2*hidden)

        if self.pooling == "last":
            # Concatenate last forward and backward states
            pooled = torch.cat([h_n[-2], h_n[-1]], dim=1)
        elif self.pooling == "mean":
            pooled = out.mean(dim=1)
        elif self.pooling == "attention":
            weights = torch.softmax(self.attn(out), dim=1)   # (batch, T, 1)
            pooled  = (out * weights).sum(dim=1)               # (batch, 2*hidden)

        return self.classifier(pooled)
```

---

## 6. Training Best Practices

### 6.1 Standard Training Loop

```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def train_ts_classifier(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int = 500,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 30,
    device: str = "cpu",
) -> dict:
    """
    Standard training loop for deep learning TSC models.

    Uses:
    - Cosine annealing LR schedule (OneCycleLR)
    - Label smoothing cross-entropy loss (robust to label noise)
    - Early stopping with patience
    - Best model checkpoint saving

    X_train: (n_train, T, n_channels) float32
    y_train: (n_train,) int labels
    """
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_vl = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_vl = torch.tensor(y_val,   dtype=torch.long).to(device)

    loader    = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(loader), epochs=n_epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0
    history      = {"train_loss": [], "val_acc": []}

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = criterion(logits, yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_vl)
            val_acc    = (val_logits.argmax(1) == y_vl).float().mean().item()
        model.train()

        history["train_loss"].append(total_loss / len(loader))
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (best val acc: {best_val_acc:.4f})")
            break

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} — Loss: {history['train_loss'][-1]:.4f}, "
                  f"Val Acc: {val_acc:.4f} (best: {best_val_acc:.4f})")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return history
```

### 6.2 Key Hyperparameters

```
LR Schedule:   OneCycleLR (cosine) with max_lr = 1e-3
Batch size:    32–64 (smaller = better generalization for small N)
Weight decay:  1e-4 (L2 regularization)
Dropout:       0.1–0.3 (after global pooling or in LSTM)
Label smoothing: 0.1 (helps with overconfident softmax)
Early stopping: patience = 30–50 epochs
Data augmentation: jitter σ = 0.05, scaling ∈ [0.8, 1.2]

Epochs:
  Small dataset (N < 500): 500–1000 epochs with early stopping
  Large dataset (N > 5000): 100–300 epochs
```

---

## 7. Transfer Learning and Pre-Training

### 7.1 Pre-Training on UCR Archive

```
Strategy 1: Self-supervised pre-training
  1. Train encoder on large TS dataset (e.g., ElectricDevices) with:
     - Contrastive learning (SimCLR for TS)
     - Masked autoencoding (TS-BERT)
     - Temporal contrastive learning
  2. Fine-tune encoder + new head on target dataset

Strategy 2: Multi-task pre-training
  1. Train on multiple UCR datasets simultaneously
  2. Fine-tune on target task
  → Works best when source and target domains are similar

Strategy 3: ROCKET features as pre-trained features
  1. Fit ROCKET on source domain data
  2. Use ROCKET features for target domain classification
  → No gradient training needed on target
```

### 7.2 Contrastive Augmentation for TS

```python
def contrastive_augment(x: torch.Tensor, sigma: float = 0.05) -> tuple:
    """
    Generate two views of a time series for contrastive learning.
    Each view = different random augmentation of the same sample.
    """
    def jitter(s): return s + torch.randn_like(s) * sigma
    def scale(s):  return s * torch.empty(s.size(0), 1, 1).uniform_(0.8, 1.2).to(s.device)
    def crop_resize(s, crop_r=0.9):
        T       = s.size(2)
        n_keep  = int(T * crop_r)
        start   = torch.randint(0, T - n_keep + 1, (1,)).item()
        cropped = s[:, :, start:start + n_keep]
        return torch.nn.functional.interpolate(cropped, T, mode="linear", align_corners=False)

    view1 = jitter(scale(x))
    view2 = crop_resize(jitter(x))
    return view1, view2
```

---

## 8. Production Pipeline

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class DeepTSClassifier:
    """
    Production-ready deep learning time series classifier.
    Defaults to InceptionTime; supports FCN, ResNet, LSTM.
    """

    MODELS = {
        "inception": InceptionTime,
        "fcn":       FCN,
        "resnet":    ResNetTS,
        "lstm":      LSTMClassifier,
    }

    def __init__(
        self,
        model_name:  str = "inception",
        n_channels:  int = 1,
        n_classes:   int = 2,
        n_epochs:    int = 300,
        batch_size:  int = 32,
        lr:          float = 1e-3,
        patience:    int = 30,
        device:      str = "cpu",
    ):
        self.model_name = model_name
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.patience   = patience
        self.device     = device
        self._fitted    = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train the classifier.

        X_train: (n_samples, T) or (n_samples, T, n_channels)
        y_train: integer class labels
        """
        if X_train.ndim == 2:
            X_train = X_train[:, :, np.newaxis]

        # Z-normalize per series
        self._mu    = X_train.mean(axis=1, keepdims=True)
        self._sigma = X_train.std(axis=1, keepdims=True) + 1e-8
        X_train_n   = (X_train - self._mu) / self._sigma

        if X_val is None:
            split   = int(0.85 * len(X_train_n))
            X_val   = X_train_n[split:]
            y_val   = y_train[split:]
            X_train_n = X_train_n[:split]
            y_train   = y_train[:split]
        elif X_val.ndim == 2:
            X_val = X_val[:, :, np.newaxis]
            X_val = (X_val - X_val.mean(axis=1, keepdims=True)) / (X_val.std(axis=1, keepdims=True) + 1e-8)

        # Build model
        self.model_ = self.MODELS[self.model_name](
            n_channels=self.n_channels, n_classes=self.n_classes
        )

        from sklearn.preprocessing import LabelEncoder
        self._le = LabelEncoder()
        y_train_enc = self._le.fit_transform(y_train)
        y_val_enc   = self._le.transform(y_val)

        self._history = train_ts_classifier(
            self.model_, X_train_n.astype(np.float32), y_train_enc,
            X_val.astype(np.float32), y_val_enc,
            n_epochs=self.n_epochs, batch_size=self.batch_size,
            lr=self.lr, patience=self.patience, device=self.device,
        )
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted
        if X.ndim == 2: X = X[:, :, np.newaxis]
        X_n   = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        X_t   = torch.tensor(X_n, dtype=torch.float32).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_t)
        preds = logits.argmax(1).cpu().numpy()
        return self._le.inverse_transform(preds)
```

---

*← [03 — Feature-Based](./03_feature_based_classification.md) | [Module README](./README.md) | Next: [05 — TS Clustering](./05_ts_clustering_methods.md) →*
