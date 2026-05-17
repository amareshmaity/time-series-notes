# 05 — Diffusion Models for Time Series

> **Module**: 12 Multivariate & Advanced Topics | **File**: 5 of 6
>
> Diffusion models have redefined generative AI in images and audio. For time series, they enable principled probabilistic forecasting, imputation of missing values, and generation of synthetic data — all framed as score-matching on the temporal data manifold. This note covers score-based diffusion theory, TimeGrad, CSDI, and practical training patterns.

---

## Table of Contents

1. [Diffusion Model Intuition](#1-diffusion-model-intuition)
2. [Denoising Diffusion Probabilistic Models (DDPM)](#2-denoising-diffusion-probabilistic-models-ddpm)
3. [TimeGrad — Autoregressive Diffusion Forecasting](#3-timegrad--autoregressive-diffusion-forecasting)
4. [CSDI — Conditional Score-Based Diffusion for Imputation](#4-csdi--conditional-score-based-diffusion-for-imputation)
5. [Score-Based Models (Score Matching)](#5-score-based-models-score-matching)
6. [Training and Sampling](#6-training-and-sampling)
7. [Implementation Patterns](#7-implementation-patterns)

---

## 1. Diffusion Model Intuition

### 1.1 The Core Idea

```
Diffusion models learn to reverse a noise process:

FORWARD PROCESS (fixed, not learned):
  x₀     → x₁ → x₂ → ... → xₜ → ... → xₙ
  (clean)    (noisy)           (pure Gaussian noise)
  
  xₜ = √(ᾱₜ) x₀ + √(1-ᾱₜ) ε,  ε ~ N(0,I)
  
  where ᾱₜ = Π_{s=1}^{t} (1-βₛ),  βₛ ∈ (0,1) (noise schedule)

REVERSE PROCESS (learned):
  xₙ → ... → xₜ₋₁ → ... → x₁ → x₀
  (noise)                      (clean signal)
  
  p_θ(xₜ₋₁ | xₜ) = N(xₜ₋₁; μ_θ(xₜ, t), Σ_θ(xₜ, t))
  
  A neural network ε_θ(xₜ, t) learns to predict the noise ε.

Training objective (simplified):
  L = E_{t, x₀, ε} [||ε - ε_θ(xₜ, t)||²]
  
  → Predict the noise that was added → subtract to get x₀.

Why it works:
  The model learns the score function ∇ log p(x) of the data distribution.
  Starting from Gaussian noise and following the learned score → generates samples.
```

### 1.2 Why Diffusion for Time Series?

```
ADVANTAGES over GANs and VAEs:
  ✅ Training stability (no adversarial mode collapse)
  ✅ Superior sample quality (outperforms TimeGAN on most benchmarks)
  ✅ Natural uncertainty quantification (sample multiple times → ensemble)
  ✅ Principled imputation (condition on observed values)
  ✅ Flexible conditioning (condition on history, covariates, graph structure)

CHALLENGES:
  ❌ Slow sampling (T reverse steps, typically 100-1000)
  ❌ More compute-intensive than GAN training
  ❌ Tuning the noise schedule is important and non-trivial
  ❌ Sequential structure not natural — needs careful architecture design
```

---

## 2. Denoising Diffusion Probabilistic Models (DDPM)

### 2.1 Noise Schedule

```
Linear noise schedule (Ho et al., 2020):
  βₜ linearly increases from β₁ = 1e-4 to βₜ = 0.02

Cosine noise schedule (Nichol & Dhariwal, 2021):
  ᾱₜ = cos²(π/2 · (t/T + s) / (1+s))  (more gradual noise)
  Better for images; also better for TS (preserves structure longer)

For time series: use fewer steps (T=50-200) vs. images (T=1000)
  Fewer steps → faster sampling → practical for real-time inference
```

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_noise_schedule(T: int, s: float = 0.008) -> dict:
    """
    Cosine noise schedule for diffusion models.

    Parameters
    ----------
    T : number of diffusion steps
    s : small offset for numerical stability

    Returns
    -------
    dict with betas, alphas, alpha_bars, sqrt versions
    """
    t     = np.linspace(0, T, T + 1)
    f_t   = np.cos((t/T + s) / (1+s) * np.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]

    betas       = 1 - alpha_bar[1:] / alpha_bar[:-1]
    betas       = np.clip(betas, 1e-6, 0.999)
    alphas      = 1 - betas
    alpha_bar   = np.cumprod(alphas)

    return {
        "betas":          torch.tensor(betas, dtype=torch.float32),
        "alphas":         torch.tensor(alphas, dtype=torch.float32),
        "alpha_bar":      torch.tensor(alpha_bar, dtype=torch.float32),
        "sqrt_alpha_bar": torch.tensor(np.sqrt(alpha_bar), dtype=torch.float32),
        "sqrt_one_minus_alpha_bar": torch.tensor(np.sqrt(1 - alpha_bar), dtype=torch.float32),
    }


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: dict) -> tuple:
    """
    Forward diffusion: sample xₜ given x₀ and timestep t.
    xₜ = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε

    Parameters
    ----------
    x0       : (batch, T_seq) or (batch, T_seq, D) clean signal
    t        : (batch,) integer timestep indices
    schedule : noise schedule dict from cosine_noise_schedule()

    Returns
    -------
    xt    : noised signal at timestep t
    noise : the noise that was added (training target)
    """
    noise = torch.randn_like(x0)
    s_ab  = schedule["sqrt_alpha_bar"][t].view(-1, *([1]*(x0.ndim-1)))
    s_oab = schedule["sqrt_one_minus_alpha_bar"][t].view(-1, *([1]*(x0.ndim-1)))
    xt    = s_ab * x0 + s_oab * noise
    return xt, noise
```

---

## 3. TimeGrad — Autoregressive Diffusion Forecasting

### 3.1 Architecture

```
TimeGrad (Rasul et al., 2021):
  Combines autoregressive RNN with DDPM for probabilistic forecasting.

  1. RNN encoder processes historical context {x₁,...,x_c}
     → Hidden state h_c encodes temporal context
     
  2. For each forecast step t = c+1,...,c+H:
     a. DDPM generates y_t | h_{t-1}
        (diffusion conditioned on RNN hidden state)
     b. RNN updates: h_t = RNN(y_t, h_{t-1})

  3. Sampling:
     Start from Gaussian noise y_T
     Reverse T steps conditioned on h_{t-1}
     → Sample y_t from p_θ(y_t | h_{t-1})

Training:
  Standard DDPM loss: predict noise ε given (y_τ, τ, h_{t-1})
  Backprop through both denoising network AND RNN.
```

```python
class TimeGradDenoisingNet(nn.Module):
    """
    Denoising network for TimeGrad.
    Predicts noise ε given noisy signal y_τ, diffusion step τ, and context h.
    """

    def __init__(
        self,
        context_dim: int,    # RNN hidden state dimension
        hidden_dim:  int = 64,
        n_layers:    int = 3,
        T:           int = 100,
    ):
        super().__init__()
        self.T       = T
        # Timestep embedding
        self.step_emb = nn.Embedding(T, hidden_dim)
        # Denoising MLP (input: noisy signal + context + step embedding)
        layers = []
        in_dim = 1 + context_dim + hidden_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else 1
            layers += [nn.Linear(in_dim, out_dim)]
            if i < n_layers - 1:
                layers += [nn.SiLU()]
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        y_tau: torch.Tensor,   # (batch, 1) — noisy signal
        tau:   torch.Tensor,   # (batch,) — diffusion step
        h:     torch.Tensor,   # (batch, context_dim) — RNN context
    ) -> torch.Tensor:
        """Predict noise ε given (y_tau, tau, h). Returns (batch, 1)."""
        step_e = self.step_emb(tau)          # (batch, hidden_dim)
        x_in   = torch.cat([y_tau, h, step_e], dim=-1)
        return self.net(x_in)


class TimeGradRNN(nn.Module):
    """
    Simple GRU-based RNN context encoder for TimeGrad.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.rnn    = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.hidden = nn.Parameter(torch.zeros(n_layers, 1, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, T_context, 1)
        Returns last hidden state: (batch, hidden_dim)
        """
        h0     = self.hidden.expand(-1, x.size(0), -1).contiguous()
        _, h_n = self.rnn(x, h0)
        return h_n[-1]   # (batch, hidden_dim)
```

---

## 4. CSDI — Conditional Score-Based Diffusion for Imputation

### 4.1 Overview

```
CSDI (Tashiro et al., 2021):
  Application: TS imputation — fill in missing values given observed values.

  Conditioning:
    Target values (observed): xₒ  → conditioned on
    Missing values: xₘ           → to be generated

  Architecture:
    Transformer-based denoising network.
    Input: noisy target sequence y_τ concatenated with observed xₒ.
    Self-attention over: (1) time dimension, (2) feature dimension.
    
  Loss:
    L = E [||εₘ - ε_θ(y_τ, xₒ, τ)||²]
    Only penalize noise prediction on MISSING positions.

  Sampling:
    Condition on observed xₒ at every reverse step.
    → Fill in missing values consistently with observed context.

Extensions:
  CSDI-probabilistic forecasting:
    Mark future values as "missing" → CSDI forecasts them.
  
  CSDI-spatiotemporal:
    Add spatial conditioning (sensor graph) for traffic imputation.
```

```python
class CSDITransformerBlock(nn.Module):
    """
    One transformer block for CSDI denoising network.
    Applies attention over both time and feature dimensions.
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        # Temporal attention
        self.time_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # Feature attention (across D variables)
        self.feat_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff        = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, T, D, d_model)
        Applies attention across time (dim=1) and features (dim=2).
        """
        B, T, D, C = x.shape

        # Temporal attention: for each feature, attend over time
        xt = x.reshape(B*D, T, C)
        xt = self.norm1(xt + self.time_attn(xt, xt, xt)[0])

        # Feature attention: for each time step, attend over features
        xd = xt.reshape(B, D, T, C).permute(0,2,1,3).reshape(B*T, D, C)
        xd = self.norm2(xd + self.feat_attn(xd, xd, xd)[0])
        xd = xd.reshape(B, T, D, C)

        xd = self.norm3(xd + self.ff(xd))
        return xd
```

---

## 5. Score-Based Models (Score Matching)

### 5.1 Score Function and NCSN

```
Score function:
  s(x) = ∇_x log p(x)
  
  Points from low-probability regions toward high-probability regions.
  If we know s(x), we can generate samples via Langevin dynamics:
    x_{t+1} = x_t + η s(x_t) + √(2η) ε

Score matching (Hyvärinen, 2005):
  Train s_θ(x) ≈ ∇_x log p(x) using:
    L = E [||s_θ(x) - ∇_x log p(x)||²]

Noisy Score Matching (Song & Ermon, 2019):
  Train s_θ(x, σ) across multiple noise levels σ₁ > σ₂ > ... > σₙ
  → Captures score at multiple scales
  → Annealed Langevin dynamics for generation

Connection to DDPM:
  DDPM's noise prediction = scaled score function:
    ε_θ(x_t, t) = -σₜ · s_θ(x_t, t)
```

---

## 6. Training and Sampling

### 6.1 DDPM Training Loop

```python
def ddpm_training_step(
    model:    nn.Module,
    x0:       torch.Tensor,
    schedule: dict,
    T:        int,
) -> torch.Tensor:
    """
    One DDPM training step for time series denoising model.

    1. Sample random diffusion step τ ~ Uniform(1, T)
    2. Sample noise ε ~ N(0, I)
    3. Compute xτ = √ᾱτ x₀ + √(1-ᾱτ) ε
    4. Predict ε̂ = model(xτ, τ)
    5. Loss = ||ε - ε̂||²

    Parameters
    ----------
    model    : denoising neural network
    x0       : (batch, T_seq) or (batch, T_seq, D) — clean signal
    schedule : noise schedule dict
    T        : total diffusion steps
    """
    B = x0.shape[0]
    # 1. Sample random timesteps
    tau   = torch.randint(0, T, (B,), device=x0.device)
    # 2-3. Add noise
    xt, noise = q_sample(x0, tau, schedule)
    # 4. Predict noise
    noise_pred = model(xt, tau)
    # 5. Loss
    return F.mse_loss(noise_pred, noise)


def ddpm_sample(
    model:    nn.Module,
    schedule: dict,
    shape:    tuple,
    T:        int,
    device:   str = "cpu",
    condition = None,
) -> torch.Tensor:
    """
    DDPM reverse sampling: generate clean signal from Gaussian noise.
    
    Parameters
    ----------
    model    : trained denoising network
    schedule : noise schedule dict
    shape    : shape of output tensor (batch, ...)
    T        : total diffusion steps
    condition: optional conditioning tensor (for conditional generation)
    
    Returns
    -------
    x0 : generated signal of `shape`
    """
    model.eval()
    betas     = schedule["betas"].to(device)
    alphas    = schedule["alphas"].to(device)
    alpha_bar = schedule["alpha_bar"].to(device)

    x = torch.randn(*shape, device=device)

    with torch.no_grad():
        for t in reversed(range(T)):
            tau  = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps  = model(x, tau) if condition is None else model(x, tau, condition)

            # Compute posterior mean
            a_t  = alphas[t]
            ab_t = alpha_bar[t]
            coef = (1 - a_t) / (1 - ab_t).sqrt()
            mu   = (1 / a_t.sqrt()) * (x - coef * eps)

            if t > 0:
                noise = torch.randn_like(x)
                sigma = betas[t].sqrt()
                x     = mu + sigma * noise
            else:
                x = mu

    return x
```

### 6.2 DDIM — Accelerated Sampling

```python
def ddim_sample(
    model:     nn.Module,
    schedule:  dict,
    shape:     tuple,
    T_sample:  int = 50,    # inference steps (much less than training T)
    T_train:   int = 1000,  # training steps
    eta:       float = 0.0, # 0 = deterministic DDIM
    device:    str = "cpu",
) -> torch.Tensor:
    """
    DDIM (Song et al., 2021) — deterministic fast sampling.
    
    Uses T_sample steps instead of T_train → 5-20x speedup.
    eta = 0: fully deterministic (good for inference)
    eta = 1: stochastic (matches DDPM)
    """
    model.eval()
    alpha_bar = schedule["alpha_bar"].to(device)

    # Select T_sample evenly-spaced timesteps
    step_indices = np.linspace(0, T_train - 1, T_sample, dtype=int)[::-1]

    x = torch.randn(*shape, device=device)

    with torch.no_grad():
        for i, t in enumerate(step_indices):
            tau   = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps   = model(x, tau)
            ab_t  = alpha_bar[t]
            x0_pred = (x - (1-ab_t).sqrt() * eps) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-3, 3)

            if i < len(step_indices) - 1:
                t_prev = step_indices[i + 1]
                ab_prev = alpha_bar[t_prev]
                sigma = eta * ((1-ab_prev)/(1-ab_t)).sqrt() * ((1-ab_t/ab_prev)).sqrt()
                dir_xt = (1 - ab_prev - sigma**2).sqrt() * eps
                x = ab_prev.sqrt() * x0_pred + dir_xt
                if eta > 0:
                    x += sigma * torch.randn_like(x)
            else:
                x = x0_pred

    return x
```

---

## 7. Implementation Patterns

### 7.1 Minimal DDPM for Time Series

```python
class TSDenoiser(nn.Module):
    """
    Minimal denoising network for univariate TS diffusion model.
    Architecture: Timestep embedding + 1D Residual CNN.
    """

    def __init__(self, seq_len: int, d_model: int = 64, T: int = 200):
        super().__init__()
        self.T         = T
        self.step_emb  = nn.Sequential(
            nn.Embedding(T, d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )
        self.input_proj = nn.Linear(1, d_model)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, 3, padding=1),
                nn.GroupNorm(8, d_model),
                nn.SiLU(),
                nn.Conv1d(d_model, d_model, 3, padding=1),
                nn.GroupNorm(8, d_model),
            ) for _ in range(4)
        ])
        self.output = nn.Linear(d_model, 1)

    def forward(self, x_tau: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        x_tau : (batch, T_seq) — noisy signal
        tau   : (batch,) — diffusion step

        Returns predicted noise: (batch, T_seq)
        """
        # Step embedding added to each position
        step_e = self.step_emb(tau).unsqueeze(1)   # (batch, 1, d_model)
        h      = self.input_proj(x_tau.unsqueeze(-1)) + step_e  # (B, T_seq, d_model)
        h      = h.permute(0, 2, 1)   # (B, d_model, T_seq)
        for block in self.res_blocks:
            h = h + block(h)
        h = h.permute(0, 2, 1)   # (B, T_seq, d_model)
        return self.output(h).squeeze(-1)   # (B, T_seq)


def train_ts_diffusion_model():
    """Train a minimal DDPM on synthetic TS data."""
    import torch.optim as optim

    T_steps  = 200
    seq_len  = 64
    n_epochs = 100
    B        = 32

    schedule = cosine_noise_schedule(T_steps)
    model    = TSDenoiser(seq_len=seq_len, d_model=64, T=T_steps)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Synthetic training data: sin waves with phase/amplitude variation
    np.random.seed(42)
    t = np.linspace(0, 4*np.pi, seq_len)
    losses = []

    for epoch in range(n_epochs):
        # Sample batch of synthetic series
        phases   = np.random.uniform(-np.pi, np.pi, B)
        amps     = np.random.uniform(0.5, 1.5, B)
        X_batch  = torch.tensor(
            np.array([amps[i] * np.sin(t + phases[i]) for i in range(B)]),
            dtype=torch.float32
        )

        loss = ddpm_training_step(model, X_batch, schedule, T_steps)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        losses.append(loss.item())
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} — Loss: {loss.item():.6f}")

    # Generate samples
    print("\nGenerating samples...")
    samples = ddpm_sample(model, schedule, shape=(4, seq_len), T=T_steps)
    print(f"Generated: {samples.shape}, "
          f"range: [{samples.min():.2f}, {samples.max():.2f}]")
    return model, losses, samples


if __name__ == "__main__":
    model, losses, samples = train_ts_diffusion_model()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(losses); axes[0].set_title("Training Loss"); axes[0].grid(alpha=0.3)
    t = np.linspace(0, 4*np.pi, 64)
    for s in samples:
        axes[1].plot(s.numpy(), alpha=0.7)
    axes[1].set_title("Generated Samples (4 samples)"); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("diffusion_ts_demo.png", dpi=150); plt.show()
```

---

*← [04 — GNNs](./04_graph_neural_networks_for_ts.md) | [Module README](./README.md) | Next: [06 — Synthetic TS](./06_synthetic_ts_generation.md) →*
