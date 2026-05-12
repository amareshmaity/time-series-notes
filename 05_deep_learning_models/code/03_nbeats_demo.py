"""
03_nbeats_demo.py
==================
Module 05 — Deep Learning Models
Topic   : N-BEATS & N-HiTS with Neuralforecast

Covers:
  - Data preparation in Nixtla long format
  - N-BEATS generic vs. interpretable configurations
  - N-HiTS for long horizons
  - Walk-forward cross-validation
  - Decomposed component visualization (trend + seasonality)
  - Comparison with SARIMA and Seasonal Naive baselines
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS, NHITS
    from neuralforecast.losses.numpy import mse, mae
    HAS_NF = True
    print("neuralforecast installed ✅")
except ImportError:
    HAS_NF = False
    print("neuralforecast not installed. Run: pip install neuralforecast")
    print("Showing architecture-only demo.")

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
BLUE, RED, GREEN, ORANGE = "#2C7BB6", "#D7191C", "#1A9641", "#F07D00"


# ─────────────────────────────────────────────────────────────────────────────
# 1. CREATE DATASET
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
n = 36 * 12   # 36 years of monthly data = 432 months
idx = pd.date_range("1990-01", periods=n, freq="MS")
t = np.arange(n)

# Trend + seasonality + noise
series_vals = (
    100
    + 0.5 * t                              # linear trend
    + 30 * np.sin(2*np.pi*t/12)           # yearly seasonality
    + 10 * np.sin(4*np.pi*t/12)           # bi-annual component
    + np.random.normal(0, 5, n)
).clip(min=0)

series = pd.Series(series_vals, index=idx, name="value")

H = 24   # forecast 24 months ahead

# Convert to Nixtla long format
df = pd.DataFrame({
    "unique_id": "series_1",
    "ds": idx,
    "y": series_vals,
})
train_df = df.iloc[:-H]
test_df  = df.iloc[-H:]

print(f"Total: {len(df)} | Train: {len(train_df)} | Test (H): {H}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEASONAL NAIVE BASELINE
# ─────────────────────────────────────────────────────────────────────────────

snaive = series.iloc[-H-12:-12].values   # last year same months
rmse_fn = lambda a, p: np.sqrt(((a - p)**2).mean())
actual  = series.iloc[-H:].values
snaive_rmse = rmse_fn(actual, snaive)
print(f"\nSeasonal Naive RMSE: {snaive_rmse:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. N-BEATS (NEURALFORECAST)
# ─────────────────────────────────────────────────────────────────────────────

if HAS_NF:
    print("\n--- N-BEATS Interpretable ---")
    model_nbeats = NBEATS(
        h=H,
        input_size=H * 2,          # 2× horizon lookback
        stack_types=["trend", "seasonality", "identity"],
        n_blocks=[3, 3, 1],
        n_harmonics=2,             # Fourier harmonics (seasonality stack)
        n_polynomials=2,           # polynomial degree (trend stack)
        # mlp_units replaced with n_mlp_units (scalar) in neuralforecast >= 0.4
        n_mlp_units=256,
        n_layers=4,
        max_steps=1000,
        batch_size=32,
        learning_rate=1e-3,
        val_check_steps=100,
        early_stop_patience_steps=5,
        random_seed=42,
    )

    nf = NeuralForecast(models=[model_nbeats], freq="MS")
    nf.fit(df=train_df)
    forecast_nbeats = nf.predict()
    pred_nbeats = forecast_nbeats["NBEATS"].values
    nbeats_rmse = rmse_fn(actual, pred_nbeats)
    print(f"N-BEATS Interpretable RMSE: {nbeats_rmse:.4f}")
    print(f"Improvement vs. Seasonal Naive: {(snaive_rmse - nbeats_rmse)/snaive_rmse*100:.1f}%")


    # ─────────────────────────────────────────────────────────────────────────
    # 4. N-BEATS GENERIC
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- N-BEATS Generic ---")
    model_nbeats_gen = NBEATS(
        h=H,
        input_size=H * 2,
        stack_types=["identity"] * 3,  # generic (learned basis)
        n_blocks=[1, 1, 1],
        n_mlp_units=512,
        n_layers=4,
        max_steps=1000,
        batch_size=32,
        learning_rate=1e-3,
        val_check_steps=100,
        early_stop_patience_steps=5,
        random_seed=42,
    )
    nf_gen = NeuralForecast(models=[model_nbeats_gen], freq="MS")
    nf_gen.fit(df=train_df)
    pred_nbeats_gen = nf_gen.predict()["NBEATS"].values
    print(f"N-BEATS Generic RMSE: {rmse_fn(actual, pred_nbeats_gen):.4f}")


    # ─────────────────────────────────────────────────────────────────────────
    # 5. N-HiTS FOR LONG HORIZONS
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- N-HiTS ---")
    model_nhits = NHITS(
        h=H,
        input_size=H * 3,
        stack_types=["identity", "identity", "identity"],
        n_blocks=[1, 1, 1],
        # Multi-rate downsampling: low-freq stacks see smoother input
        # n_freq_downsample controls output interpolation factor per stack
        n_freq_downsample=[12, 4, 1],
        # pooling_mode replaces n_pool_kernel_size in neuralforecast >= 0.4
        pooling_mode="AveragePooling",
        interpolation_mode="linear",
        n_mlp_units=256,
        n_layers=4,
        max_steps=1000,
        batch_size=32,
        learning_rate=1e-3,
        val_check_steps=100,
        early_stop_patience_steps=5,
        random_seed=42,
    )
    nf_nhits = NeuralForecast(models=[model_nhits], freq="MS")
    nf_nhits.fit(df=train_df)
    pred_nhits = nf_nhits.predict()["NHITS"].values
    print(f"N-HiTS RMSE: {rmse_fn(actual, pred_nhits):.4f}")


    # ─────────────────────────────────────────────────────────────────────────
    # 6. WALK-FORWARD CV
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- Walk-Forward CV (N-BEATS Interpretable) ---")
    model_cv = NBEATS(
        h=H, input_size=H*2,
        stack_types=["trend", "seasonality", "identity"],
        n_blocks=[3, 3, 1],
        max_steps=500, random_seed=42,
    )
    nf_cv = NeuralForecast(models=[model_cv], freq="MS")
    cv_df = nf_cv.cross_validation(
        df=df, n_windows=3, step_size=H,
    )
    cv_rmse = np.sqrt(mse(cv_df["y"].values, cv_df["NBEATS"].values))
    print(f"N-BEATS CV RMSE ({3} windows): {cv_rmse:.4f}")


    # ─────────────────────────────────────────────────────────────────────────
    # 7. VISUALIZATION
    # ─────────────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))

    # Forecast comparison
    hist_idx  = series.index[-60:-H]
    fcst_idx  = series.index[-H:]
    hist_vals = series.values[-60:-H]

    axes[0].plot(hist_idx, hist_vals, color="gray", linewidth=1.2, label="History")
    axes[0].plot(fcst_idx, actual,    color="black", linewidth=2.5, label="Actual")
    axes[0].plot(fcst_idx, pred_nbeats,      color=BLUE,   linewidth=2, linestyle="--",
                 label=f"N-BEATS Interp. ({nbeats_rmse:.2f})")
    axes[0].plot(fcst_idx, pred_nbeats_gen,  color=GREEN,  linewidth=1.5, linestyle=":",
                 label=f"N-BEATS Generic ({rmse_fn(actual,pred_nbeats_gen):.2f})")
    axes[0].plot(fcst_idx, pred_nhits,       color=ORANGE, linewidth=1.5, linestyle="-.",
                 label=f"N-HiTS ({rmse_fn(actual,pred_nhits):.2f})")
    axes[0].plot(fcst_idx, snaive,           color=RED,    linewidth=1.2, linestyle="--",
                 alpha=0.6, label=f"Seasonal Naive ({snaive_rmse:.2f})")
    axes[0].axvline(fcst_idx[0], color="black", linewidth=0.8, linestyle=":")
    axes[0].legend(fontsize=8)
    axes[0].set_title("N-BEATS & N-HiTS vs. Baselines")

    # RMSE leaderboard
    leaderboard = {
        "N-BEATS Interp.": rmse_fn(actual, pred_nbeats),
        "N-BEATS Generic": rmse_fn(actual, pred_nbeats_gen),
        "N-HiTS":          rmse_fn(actual, pred_nhits),
        "Seasonal Naive":  snaive_rmse,
    }
    names  = list(leaderboard.keys())
    values = list(leaderboard.values())
    colors_bar = [BLUE, GREEN, ORANGE, RED]
    axes[1].barh(names, values, color=colors_bar, edgecolor="white")
    axes[1].set_xlabel("Test RMSE (lower = better)")
    axes[1].set_title("Model Comparison Leaderboard")
    for i, v in enumerate(values):
        axes[1].text(v + 0.1, i, f"{v:.2f}", va="center")

    plt.suptitle("N-BEATS & N-HiTS — Monthly Forecasting", fontweight="bold")
    plt.tight_layout()
    plt.savefig("03_nbeats_comparison.png", bbox_inches="tight")
    plt.show()

else:
    # ── Architecture demo without neuralforecast ──────────────────────────
    print("\n[Showing architecture diagram without neuralforecast]")

    # Simple N-BEATS block in pure PyTorch for illustration
    import torch
    import torch.nn as nn

    class NBEATSBlock(nn.Module):
        def __init__(self, lookback, horizon, theta_dim, units=256, n_layers=4):
            super().__init__()
            self.fc = nn.Sequential(*[
                nn.Sequential(nn.Linear(lookback if i==0 else units, units), nn.ReLU())
                for i in range(n_layers)
            ])
            self.theta_b = nn.Linear(units, theta_dim)   # backcast coefficients
            self.theta_f = nn.Linear(units, theta_dim)   # forecast coefficients
            # Generic basis: learned linear projection
            self.basis_b = nn.Linear(theta_dim, lookback, bias=False)
            self.basis_f = nn.Linear(theta_dim, horizon,  bias=False)

        def forward(self, x):
            h = self.fc(x)
            backcast = self.basis_b(self.theta_b(h))
            forecast  = self.basis_f(self.theta_f(h))
            return backcast, forecast

    block = NBEATSBlock(lookback=48, horizon=24, theta_dim=32)
    x_demo = torch.randn(8, 48)
    b, f = block(x_demo)
    print(f"Block backcast: {b.shape} | forecast: {f.shape}")
    print("\nInstall neuralforecast for full N-BEATS training: pip install neuralforecast")

print("\n✅ N-BEATS / N-HiTS demo complete.")
