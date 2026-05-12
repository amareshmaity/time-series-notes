# 07 — Fine-Tuning Time Series Foundation Models

> **Module**: 06 Transformers & Foundation Models | **File**: 7 of 7
>
> Fine-tuning adapts a pre-trained foundation model to a specific domain using a small amount of task-specific data. This note covers fine-tuning Chronos, LoRA for TS transformers, and when fine-tuning beats zero-shot.

---

## Table of Contents

1. [Fine-Tuning vs. Zero-Shot vs. From Scratch](#1-fine-tuning-vs-zero-shot-vs-from-scratch)
2. [Fine-Tuning Chronos (Full)](#2-fine-tuning-chronos-full)
3. [LoRA for Time Series Transformers](#3-lora-for-time-series-transformers)
4. [Fine-Tuning TimeGPT via API](#4-fine-tuning-timegpt-via-api)
5. [Domain Adaptation Best Practices](#5-domain-adaptation-best-practices)
6. [When Fine-Tuning Hurts](#6-when-fine-tuning-hurts)

---

## 1. Fine-Tuning vs. Zero-Shot vs. From Scratch

| Approach | Data Needed | Compute | Accuracy | When to Use |
|----------|------------|---------|----------|-------------|
| **Zero-shot** | None | None | Good baseline | Cold start, rapid prototype |
| **Fine-tuning** | 100–10K series | Low–Medium | Best | Domain shift, specialty data |
| **From scratch** | Millions of series | Very High | Comparable | Research, proprietary corpus |

### Decision Rule

```
Zero-shot skill > 0.10    → stay zero-shot
Zero-shot skill 0–0.10    → try fine-tuning (small investment, big gain)
Zero-shot skill < 0       → fine-tune or train classical model
Data < 100 obs per series → zero-shot only (fine-tuning risks overfitting)
```

---

## 2. Fine-Tuning Chronos (Full)

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from chronos import ChronosPipeline

# ── Dataset wrapper ──────────────────────────────────────────────────────────

class ChronosFineTuneDataset(Dataset):
    """
    Sliding-window dataset for fine-tuning Chronos.
    Each sample: (context_window, forecast_window) pair.
    """
    def __init__(self, series_list: list, context_length: int,
                 prediction_length: int):
        self.context_length    = context_length
        self.prediction_length = prediction_length
        self.samples = []

        for series in series_list:
            s = np.array(series, dtype=np.float32)
            n = len(s)
            # Extract all valid (context, forecast) pairs
            for start in range(0, n - context_length - prediction_length + 1, 1):
                ctx  = s[start : start + context_length]
                tgt  = s[start + context_length : start + context_length + prediction_length]
                self.samples.append((ctx, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctx, tgt = self.samples[idx]
        return torch.tensor(ctx), torch.tensor(tgt)


# ── Fine-tuning loop ─────────────────────────────────────────────────────────

def fine_tune_chronos(
    series_list:       list,
    model_size:        str   = "small",
    context_length:    int   = 512,
    prediction_length: int   = 24,
    n_steps:           int   = 500,
    lr:                float = 1e-4,
    batch_size:        int   = 32,
    device:            str   = "cpu",
    save_path:         str   = "chronos_finetuned",
):
    """
    Fine-tune Chronos on custom time series data.

    Parameters:
        series_list:    list of 1D numpy arrays (your domain data)
        model_size:     'tiny', 'mini', 'small', 'base'
        n_steps:        gradient update steps (200–2000 typical)
        lr:             learning rate (1e-5 to 1e-3)
    """
    # Load base model
    pipeline = ChronosPipeline.from_pretrained(
        f"amazon/chronos-t5-{model_size}",
        device_map=device,
        torch_dtype=torch.float32,
    )
    model = pipeline.model.to(device)

    # Build dataset
    dataset = ChronosFineTuneDataset(series_list, context_length, prediction_length)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Fine-tune dataset: {len(dataset)} samples")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr * 0.1
    )

    model.train()
    step = 0
    loader_iter = iter(loader)

    while step < n_steps:
        try:
            ctx_batch, tgt_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            ctx_batch, tgt_batch = next(loader_iter)

        ctx_batch = ctx_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        optimizer.zero_grad()

        # Chronos loss: cross-entropy on quantized token predictions
        # The model's forward pass handles tokenization internally
        loss = model(context=ctx_batch, future=tgt_batch).loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % 100 == 0:
            print(f"  Step {step:4d}/{n_steps} | loss={loss.item():.5f} "
                  f"| lr={scheduler.get_last_lr()[0]:.2e}")

    # Save fine-tuned pipeline
    pipeline.model = model
    pipeline.save_pretrained(save_path)
    print(f"\n✅ Fine-tuned model saved to: {save_path}")
    return pipeline
```

---

## 3. LoRA for Time Series Transformers

LoRA (Low-Rank Adaptation) fine-tunes **only a tiny fraction of parameters** — reducing GPU memory and preventing catastrophic forgetting of pre-trained knowledge.

### 3.1 LoRA Theory

```
Standard fine-tuning: update ALL weights W  (millions of parameters)
LoRA: freeze W, add low-rank adapter:
    W' = W + α · (A · B)

Where:
  W: (d_out, d_in) — original frozen weight
  A: (d_out, r)    — trainable (random init)
  B: (r, d_in)     — trainable (zero init → W' = W at start)
  r: rank (typically 4–64, << d_out, d_in)
  α: scaling factor (α/r)

Trainable params:  r*(d_in + d_out)   vs.   d_in*d_out  (full)
Typical reduction: 10–1000× fewer parameters
```

### 3.2 LoRA Implementation

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA adaptation.
    Freezes original weights; only A and B are trained.
    """
    def __init__(
        self,
        original_linear: nn.Linear,
        rank:  int   = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        d_out, d_in = original_linear.weight.shape

        # Freeze original weights
        self.weight = original_linear.weight
        self.bias   = original_linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # LoRA adapters (trainable)
        self.lora_A   = nn.Parameter(torch.empty(d_out, rank))
        self.lora_B   = nn.Parameter(torch.zeros(rank, d_in))  # zero init
        self.scaling  = alpha / rank
        self.dropout  = nn.Dropout(dropout)

        # Kaiming init for A (same as linear layer default)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original path (frozen)
        base = nn.functional.linear(x, self.weight, self.bias)
        # LoRA path (trained)
        lora = self.dropout(x) @ self.lora_B.T @ self.lora_A.T
        return base + self.scaling * lora


def apply_lora_to_transformer(
    model: nn.Module,
    rank:  int   = 8,
    alpha: float = 16.0,
    target_modules: list = None,
) -> nn.Module:
    """
    Replace attention projection layers with LoRA versions.
    Freezes all original weights; only LoRA adapters are trained.

    target_modules: list of layer name substrings to target
                    Default: ['q_proj', 'v_proj'] (query and value)
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "query", "value",
                          "W_q", "W_v"]   # common names

    # Freeze all params first
    for param in model.parameters():
        param.requires_grad_(False)

    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this layer should get LoRA
            if any(t in name for t in target_modules):
                # Navigate to parent and replace
                parts  = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, parts[-1], lora_layer)
                replaced += 1

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied to {replaced} layers | "
          f"Trainable: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.2f}%)")
    return model


# ── Usage example with PatchTST ─────────────────────────────────────────────

# from 03_patchtst_timesnet.md import PatchTST
# model = PatchTST(n_vars=7, lookback=336, horizon=96)

# Apply LoRA — only 2% of params become trainable
# model_lora = apply_lora_to_transformer(model, rank=8, alpha=16,
#                                         target_modules=["W_q", "W_v"])
```

### 3.3 LoRA Fine-Tuning Loop

```python
def lora_fine_tune(
    model:       nn.Module,
    train_dl:    DataLoader,
    val_dl:      DataLoader,
    n_epochs:    int   = 20,
    lr:          float = 1e-3,
    device:      str   = "cpu",
):
    """Fine-tune a LoRA-adapted model. Only LoRA adapters are updated."""
    model = model.to(device)

    # Only optimize LoRA parameters (requires_grad=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Optimizing {sum(p.numel() for p in trainable):,} LoRA parameters")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )
    best_val, best_state = float("inf"), None

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            tr_loss += loss.item()
        scheduler.step()

        # Validate
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for X, y in val_dl:
                vl_loss += criterion(model(X.to(device)), y.to(device)).item()

        tr = tr_loss / len(train_dl)
        vl = vl_loss / len(val_dl)

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()
                          if "lora" in k}   # save ONLY LoRA weights

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | train={tr:.5f} | val={vl:.5f}")

    print(f"\n✅ LoRA fine-tuning done. Best val={best_val:.5f}")
    # Restore best LoRA weights only
    model.load_state_dict(best_state, strict=False)
    return model
```

---

## 4. Fine-Tuning TimeGPT via API

TimeGPT supports fine-tuning via the Nixtla API with a single parameter:

```python
from nixtla import NixtlaClient
import os

client = NixtlaClient(api_key=os.environ["NIXTLA_API_KEY"])

# Fine-tune during forecast call
forecast_ft = client.forecast(
    df=train_df,
    h=30,
    freq="D",
    finetune_steps=100,       # 100 gradient steps on your data
    finetune_loss="default",  # 'mae', 'mse', or 'default' (quantile)
    finetune_depth=1,         # how many layers to unfreeze (1=top only)
    level=[90],
)

# Cross-validate with fine-tuning
cv_ft = client.cross_validation(
    df=df,
    h=30,
    freq="D",
    n_windows=3,
    finetune_steps=50,
)
print(cv_ft.head())
```

**Key parameters:**

| Parameter | Values | Effect |
|-----------|--------|--------|
| `finetune_steps` | 50–2000 | More steps = more specialization (risk: overfitting) |
| `finetune_loss` | `"mae"`, `"mse"`, `"default"` | Loss function for adaptation |
| `finetune_depth` | 1–5 | Layers to unfreeze (1=safest, 5=aggressive) |

---

## 5. Domain Adaptation Best Practices

```
Rule 1: Start zero-shot, measure skill
  → Fine-tune only if zero-shot skill < 0.05

Rule 2: Use short fine-tuning (50–200 steps)
  → Over-tuning destroys generalization (catastrophic forgetting)
  → Monitor validation loss; stop at minimum

Rule 3: Reserve held-out test data BEFORE fine-tuning
  → Never tune on test data (information leakage)

Rule 4: Prefer LoRA over full fine-tuning
  → Fewer trainable parameters → less overfitting risk
  → Can always restore original weights

Rule 5: Domain data quality > quantity
  → 50 high-quality in-domain series > 500 noisy ones
  → Remove obvious outliers and corrupted windows before training

Rule 6: Evaluate with walk-forward CV, not single test split
  → At least 3 windows to get stable RMSE estimate
```

---

## 6. When Fine-Tuning Hurts

Fine-tuning can make things **worse** in these situations:

| Situation | Problem | Fix |
|-----------|---------|-----|
| Very short series (< 100 obs) | Overfitting to noise | Stay zero-shot |
| Fine-tuning on test-like data | Data leakage | Strict train/val/test split |
| Too many steps | Catastrophic forgetting of pre-trained patterns | Early stopping |
| Wrong loss function | Optimizes wrong metric | Match fine-tune loss to business metric |
| Distribution mismatch (fine-tune ≠ deploy) | Adapts to wrong domain | Audit data before fine-tuning |

```python
def fine_tune_safety_check(series_list: list, horizon: int) -> dict:
    """
    Safety checks before committing to fine-tuning.
    Returns a dict of checks with pass/fail status.
    """
    checks = {}

    # Check 1: Sufficient data per series
    lengths = [len(s) for s in series_list]
    min_len = min(lengths)
    checks["min_series_length"] = {
        "value":  min_len,
        "pass":   min_len >= 4 * horizon,
        "advice": f"Need >= 4×horizon ({4*horizon}) obs; shortest has {min_len}",
    }

    # Check 2: Sufficient number of series
    n_series = len(series_list)
    checks["n_series"] = {
        "value":  n_series,
        "pass":   n_series >= 5,
        "advice": "Recommend >= 5 series for reliable fine-tuning",
    }

    # Check 3: Not too much variation in scales (may cause instability)
    import numpy as np
    means = [abs(np.mean(s)) + 1e-8 for s in series_list]
    scale_ratio = max(means) / min(means)
    checks["scale_uniformity"] = {
        "value":  scale_ratio,
        "pass":   scale_ratio < 1000,
        "advice": f"Scale ratio {scale_ratio:.1f}× — normalize if > 1000×",
    }

    for name, result in checks.items():
        status = "✅" if result["pass"] else "⚠️ "
        print(f"{status} {name}: {result['advice']}")

    all_pass = all(r["pass"] for r in checks.values())
    print(f"\n{'✅ Fine-tuning is safe to proceed.' if all_pass else '⚠️  Review warnings before fine-tuning.'}")
    return checks
```

---

## Summary

| Topic | Key Takeaway |
|-------|-------------|
| **Full fine-tuning** | Best accuracy; requires GPU; risk of forgetting pre-trained patterns |
| **LoRA** | 10–100× fewer trainable params; safe for small data; recommended default |
| **TimeGPT API fine-tuning** | Easiest; `finetune_steps=100` is a good default start |
| **When to fine-tune** | Zero-shot skill < 0.05 AND you have ≥ 5 series with ≥ 4×H observations |
| **Safety** | Always evaluate with walk-forward CV; never tune on test data |

---

*← [06 — Zero-Shot Forecasting](./06_zero_shot_forecasting.md) | [Module README](./README.md) | Next Module: [07 — Forecasting Strategies](../07_forecasting_strategies/README.md) →*
