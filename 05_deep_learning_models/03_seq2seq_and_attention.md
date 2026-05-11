# 03 — Seq2Seq and Attention Mechanisms

> **Module**: 05 Deep Learning Models | **File**: 3 of 6
>
> The encoder-decoder (Seq2Seq) architecture is the foundation of modern multi-step forecasting. Attention mechanisms enable the decoder to selectively focus on relevant parts of the input — the precursor to Transformer models.

---

## Table of Contents

1. [Why Seq2Seq for Multi-Step Forecasting](#1-why-seq2seq-for-multi-step-forecasting)
2. [Encoder-Decoder Architecture](#2-encoder-decoder-architecture)
3. [Bahdanau Attention](#3-bahdanau-attention)
4. [Teacher Forcing](#4-teacher-forcing)
5. [Implementation](#5-implementation)

---

## 1. Why Seq2Seq for Multi-Step Forecasting

### 1.1 The Problem with Direct LSTM

A direct LSTM maps `lookback → horizon` in one shot — it uses the **last hidden state** to represent the entire input context. This is a bottleneck:

```
Input: [y1, y2, ..., y60]  →  [h_60]  →  [ŷ61, ..., ŷ72]
                               ↑
                          Information bottleneck:
                          ALL 60 steps compressed into one vector
```

For long horizons (H > 10), the decoder has no direct access to which input steps were most relevant for each output step.

### 1.2 Seq2Seq Solution

```
Encoder: Processes full input sequence → produces context vectors {h_1, ..., h_L}
Decoder: At each output step t, attends to ALL encoder states → selects relevant context
```

---

## 2. Encoder-Decoder Architecture

### 2.1 Encoder

The encoder is a standard LSTM that reads the input sequence:

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, x):
        # x: (batch, lookback, input_size)
        outputs, (h_n, c_n) = self.lstm(x)
        # outputs: (batch, lookback, hidden_size) — all hidden states
        # h_n, c_n: (num_layers, batch, hidden_size) — final states
        return outputs, (h_n, c_n)
```

### 2.2 Decoder (Without Attention)

```python
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm   = nn.LSTM(output_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x_t, h_prev, c_prev):
        # x_t: (batch, 1, output_size) — input to decoder at step t
        out, (h_n, c_n) = self.lstm(x_t, (h_prev, c_prev))
        pred = self.fc_out(out.squeeze(1))   # (batch, output_size)
        return pred, h_n, c_n
```

---

## 3. Bahdanau Attention

### 3.1 Attention Score Computation

At decoder step `t`, compute how relevant each encoder state `h_enc[s]` is:

```
Score:    e(t, s) = v · tanh(W_enc · h_enc[s] + W_dec · h_dec[t-1] + b)
Weight:   α(t, s) = softmax(e(t, s)) over all s
Context:  c(t)    = Σ_s α(t, s) · h_enc[s]

Where:
  α(t, s) ∈ [0,1], Σ_s α(t,s) = 1  ← attention distribution
  c(t) = weighted sum of encoder states (context vector)
```

### 3.2 Attention Implementation

```python
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_enc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_dec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v     = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, enc_outputs, dec_hidden):
        """
        enc_outputs: (batch, lookback, hidden_size)
        dec_hidden:  (batch, hidden_size) — last decoder hidden state
        Returns:
          context: (batch, hidden_size)
          weights: (batch, lookback) — attention weights
        """
        dec_hidden = dec_hidden.unsqueeze(1)                 # (batch, 1, hidden_size)
        energy     = torch.tanh(
            self.W_enc(enc_outputs) + self.W_dec(dec_hidden) # broadcast
        )                                                    # (batch, lookback, hidden_size)
        scores  = self.v(energy).squeeze(-1)                 # (batch, lookback)
        weights = torch.softmax(scores, dim=1)               # (batch, lookback)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)  # (batch, hidden)
        return context, weights


class AttentionDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers):
        super().__init__()
        self.attention = BahdanauAttention(hidden_size)
        # Decoder input = [previous output, context vector]
        self.lstm   = nn.LSTM(output_size + hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size + hidden_size, output_size)

    def forward(self, x_t, h_prev, c_prev, enc_outputs):
        # x_t: (batch, 1, output_size)
        dec_hidden = h_prev[-1]   # last layer hidden state: (batch, hidden_size)
        context, attn_weights = self.attention(enc_outputs, dec_hidden)

        lstm_input = torch.cat([x_t, context.unsqueeze(1)], dim=-1)
        out, (h_n, c_n) = self.lstm(lstm_input, (h_prev, c_prev))

        pred = self.fc_out(torch.cat([out.squeeze(1), context], dim=-1))
        return pred, h_n, c_n, attn_weights
```

---

## 4. Teacher Forcing

### 4.1 What is Teacher Forcing

During training, instead of feeding the decoder's own prediction as the next input, we feed the **ground truth** previous value:

```
Without teacher forcing (free running):
  ŷ₁ = decoder(x_0=0, h_enc)
  ŷ₂ = decoder(x_1=ŷ₁, h)   ← uses predicted value
  (errors compound during training → unstable)

With teacher forcing:
  ŷ₁ = decoder(x_0=0, h_enc)
  ŷ₂ = decoder(x_1=y_true_1, h)   ← uses ground truth
  (stable training, but mismatch with inference)
```

### 4.2 Scheduled Sampling

```python
def decode_with_scheduled_sampling(
    decoder, enc_outputs, h, c,
    y_true, teacher_force_ratio=0.5,
):
    """Gradually reduce teacher forcing during training."""
    H = y_true.shape[1]      # horizon
    batch = y_true.shape[0]
    outputs = []

    # First decoder input = zeros (or last encoder input)
    x_t = torch.zeros(batch, 1, 1).to(y_true.device)

    for t in range(H):
        pred, h, c, _ = decoder(x_t, h, c, enc_outputs)
        outputs.append(pred)

        use_teacher = torch.rand(1).item() < teacher_force_ratio
        if use_teacher:
            x_t = y_true[:, t:t+1, :]    # ground truth
        else:
            x_t = pred.unsqueeze(1)       # model's own prediction

    return torch.stack(outputs, dim=1)   # (batch, H, output_size)
```

---

## 5. Implementation

### 5.1 Full Seq2Seq Model

```python
class Seq2SeqForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_horizon,
                 dropout=0.2, teacher_force_ratio=0.5):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = AttentionDecoder(1, hidden_size, num_layers)
        self.horizon  = output_horizon
        self.tf_ratio = teacher_force_ratio

    def forward(self, x, y_true=None):
        enc_outputs, (h, c) = self.encoder(x)

        batch = x.shape[0]
        preds = []
        x_t = torch.zeros(batch, 1, 1).to(x.device)

        for t in range(self.horizon):
            pred, h, c, _ = self.decoder(x_t, h, c, enc_outputs)
            preds.append(pred)

            if y_true is not None and torch.rand(1).item() < self.tf_ratio:
                x_t = y_true[:, t:t+1, :]
            else:
                x_t = pred.detach().unsqueeze(1)

        return torch.stack(preds, dim=1).squeeze(-1)   # (batch, horizon)
```

### 5.2 Training

```python
model = Seq2SeqForecaster(
    input_size=1, hidden_size=128, num_layers=2,
    output_horizon=12, dropout=0.2, teacher_force_ratio=0.5
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.HuberLoss(delta=1.0)   # robust to outliers

for epoch in range(100):
    model.train()
    for X_batch, y_batch in train_dl:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        # Pass y_true for teacher forcing during training
        y_pred = model(X_batch, y_true=y_batch.unsqueeze(-1))
        loss = criterion(y_pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

---

*← [02 — TCN](./02_temporal_convolutional_networks.md) | [Module README](./README.md) | Next: [04 — N-BEATS](./04_nbeats_and_nhits.md) →*
