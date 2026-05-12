"""
01_informer_demo.py
===================
Module 06 — Transformers & Foundation Models
Topic   : Informer for Long-Sequence Forecasting

Covers:
  - Long-sequence dataset preparation (Nixtla long format)
  - Informer model configuration via neuralforecast
  - Walk-forward cross-validation
  - Comparison: Informer vs. N-HiTS vs. Seasonal Naive
  - Forecast visualization with confidence intervals
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import Informer, NHITS
    from neuralforecast.losses.numpy import mse, mae
    HAS_NF = True
    print("neuralforecast installed ✅")
except ImportError:
    HAS_NF = False
    print("neuralforecast not installed. Run: pip install neuralforecast")

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
BLUE, RED, GREEN, ORANGE = "#2C7BB6", "#D7191C", "#1A9641", "#F07D00"


# ─────────────────────────────────────────────────────────────────────────────
# 1. LONG-SEQUENCE DATASET (3 years, hourly)
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
n    = 24 * 365 * 3   # 3 years of hourly data = 26,280 steps
idx  = pd.date_range("2021-01-01", periods=n, freq="H")
t    = np.arange(n)

series = (
    100
    + 0.001 * t                              # very slow trend
    + 30 * np.sin(2 * np.pi * t / 8766)     # yearly seasonality
    + 15 * np.sin(2 * np.pi * t / 168)      # weekly seasonality (168h)
    + 8  * np.sin(2 * np.pi * t / 24)       # daily seasonality
    + np.random.normal(0, 3, n)
).clip(min=0)

H = 168   # forecast 1 week ahead (168 hours) — LONG HORIZON

df = pd.DataFrame({"unique_id": "electricity", "ds": idx, "y": series})
train_df = df.iloc[:-H]
test_df  = df.iloc[-H:]
actual   = test_df["y"].values

print(f"Total:  {len(df):,} rows")
print(f"Train:  {len(train_df):,} | Test: {H} steps ({H/24:.0f} days)")

rmse_fn = lambda a, p: np.sqrt(((np.array(a) - np.array(p))**2).mean())


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEASONAL NAIVE BASELINE
# ─────────────────────────────────────────────────────────────────────────────

# For hourly data with weekly period: repeat last week
snaive = train_df["y"].values[-168:]   # last 168 hours
snaive_rmse = rmse_fn(actual, snaive)
print(f"\nSeasonal Naive RMSE: {snaive_rmse:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. INFORMER
# ─────────────────────────────────────────────────────────────────────────────

if HAS_NF:
    print("\n--- Training Informer ---")
    model_informer = Informer(
        h=H,
        input_size=H * 2,    # 2-week lookback

        # ProbSparse attention
        factor=5,            # K = factor × log(T) queries sampled
        hidden_size=64,
        n_head=4,
        e_layers=2,          # encoder stacks (distilling after each)
        d_layers=1,          # decoder layers
        d_ff=256,
        dropout=0.1,

        # Training
        max_steps=500,
        batch_size=32,
        learning_rate=1e-3,
        val_check_steps=50,
        early_stop_patience_steps=5,
        random_seed=42,
    )

    nf_informer = NeuralForecast(models=[model_informer], freq="H")
    nf_informer.fit(df=train_df)
    fc_informer = nf_informer.predict()["Informer"].values
    informer_rmse = rmse_fn(actual, fc_informer)
    print(f"Informer RMSE: {informer_rmse:.4f}")
    print(f"vs. Naive: {(snaive_rmse - informer_rmse)/snaive_rmse*100:.1f}% improvement")


    # ─────────────────────────────────────────────────────────────────────────
    # 4. N-HiTS (LONG HORIZON SPECIALIST — for comparison)
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- Training N-HiTS ---")
    model_nhits = NHITS(
        h=H,
        input_size=H * 3,
        n_freq_downsample=[24, 4, 1],   # hourly, 6h, daily aggregation
        pooling_mode="AveragePooling",
        interpolation_mode="linear",
        n_mlp_units=256,
        n_layers=4,
        max_steps=500,
        batch_size=32,
        learning_rate=1e-3,
        val_check_steps=50,
        early_stop_patience_steps=5,
        random_seed=42,
    )
    nf_nhits = NeuralForecast(models=[model_nhits], freq="H")
    nf_nhits.fit(df=train_df)
    fc_nhits = nf_nhits.predict()["NHITS"].values
    nhits_rmse = rmse_fn(actual, fc_nhits)
    print(f"N-HiTS RMSE: {nhits_rmse:.4f}")


    # ─────────────────────────────────────────────────────────────────────────
    # 5. WALK-FORWARD CV (Informer)
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- Walk-Forward CV ---")
    model_cv = Informer(
        h=H, input_size=H * 2,
        factor=5, hidden_size=64, n_head=4,
        e_layers=2, d_layers=1, d_ff=256,
        max_steps=300, random_seed=42,
    )
    nf_cv = NeuralForecast(models=[model_cv], freq="H")
    cv_df = nf_cv.cross_validation(df=df, n_windows=2, step_size=H)
    cv_rmse = np.sqrt(mse(cv_df["y"].values, cv_df["Informer"].values))
    print(f"Informer CV RMSE (2 windows): {cv_rmse:.4f}")


    # ─────────────────────────────────────────────────────────────────────────
    # 6. VISUALIZATION
    # ─────────────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Time plot — last 7 days history + 7 days forecast
    n_hist_show = 168   # 7 days
    hist_vals   = train_df["y"].values[-n_hist_show:]
    hist_idx    = np.arange(n_hist_show)
    fore_idx    = np.arange(n_hist_show, n_hist_show + H)

    axes[0].plot(hist_idx, hist_vals,
                 color="gray", linewidth=1.2, label="History")
    axes[0].plot(fore_idx, actual,
                 color="black", linewidth=2.5, label="Actual")
    axes[0].plot(fore_idx, fc_informer,
                 color=BLUE, linewidth=2, linestyle="--",
                 label=f"Informer ({informer_rmse:.2f})")
    axes[0].plot(fore_idx, fc_nhits,
                 color=GREEN, linewidth=1.5, linestyle=":",
                 label=f"N-HiTS ({nhits_rmse:.2f})")
    axes[0].plot(fore_idx, snaive,
                 color=RED, linewidth=1.2, linestyle="--",
                 alpha=0.6, label=f"Seasonal Naive ({snaive_rmse:.2f})")
    axes[0].axvline(n_hist_show - 0.5, color="black", linewidth=0.8, linestyle=":")
    axes[0].set_title("7-Day Ahead Forecast: Informer vs. Baselines")
    axes[0].legend(fontsize=9)

    # RMSE leaderboard
    models = ["Informer", "N-HiTS", "Seasonal Naive"]
    rmses  = [informer_rmse, nhits_rmse, snaive_rmse]
    colors = [BLUE, GREEN, RED]
    axes[1].barh(models, rmses, color=colors, edgecolor="white")
    axes[1].set_xlabel("RMSE (lower = better)")
    axes[1].set_title("Model Comparison — Long-Horizon Forecasting (H=168h)")
    for i, v in enumerate(rmses):
        axes[1].text(v + 0.05, i, f"{v:.3f}", va="center")

    plt.suptitle("Informer — Long-Sequence Time Series Forecasting",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("01_informer_comparison.png", bbox_inches="tight")
    plt.show()

else:
    # ── Architecture sketch without neuralforecast ───────────────────────────
    print("\n[Informer architecture — install neuralforecast for full demo]")
    print("""
Informer Architecture:
─────────────────────
Input (B, T, d_model)
  ↓
Encoder × e_layers:
  ┌─ ProbSparse Self-Attention [O(T log T)]
  │   Select top-K queries by sparsity score
  │   Full attention only for top-K queries
  └─ Distilling: MaxPool → halve sequence length
  
Decoder × d_layers:
  ┌─ Causal Self-Attention on label_len + forecast tokens
  └─ Cross-Attention to encoder output

Generative output: predict full H in one shot (not autoregressive)
    """)

print("\n✅ Informer demo complete.")
