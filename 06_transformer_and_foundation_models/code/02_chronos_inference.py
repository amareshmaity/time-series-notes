"""
02_chronos_inference.py
========================
Module 06 — Transformers & Foundation Models
Topic   : Amazon Chronos — Zero-Shot Probabilistic Forecasting

Covers:
  - Chronos zero-shot forecast (no training required)
  - Probabilistic output: median + prediction intervals
  - Batch inference across multiple series
  - Skill evaluation vs. SARIMA and Seasonal Naive
  - Visualization of forecast with uncertainty bands
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

try:
    from chronos import ChronosPipeline
    HAS_CHRONOS = True
    print("chronos-forecasting installed ✅")
except ImportError:
    HAS_CHRONOS = False
    print("chronos-forecasting not installed.")
    print("Run: pip install chronos-forecasting")

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
BLUE, RED, GREEN = "#2C7BB6", "#D7191C", "#1A9641"


# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATASET (multi-pattern)
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
n   = 365 * 3   # 3 years of daily data
idx = pd.date_range("2021-01-01", periods=n, freq="D")
t   = np.arange(n)

series = (
    100
    + 0.08 * t
    + 30 * np.sin(2 * np.pi * t / 365.25)   # annual seasonality
    + 12 * np.sin(2 * np.pi * t / 7)         # weekly seasonality
    + np.random.normal(0, 5, n)
).clip(min=0)

H      = 30    # 30-day ahead forecast
split  = n - H
train  = series[:split]
actual = series[split:]

rmse_fn = lambda a, p: np.sqrt(((np.array(a) - np.array(p))**2).mean())
mae_fn  = lambda a, p: np.abs(np.array(a) - np.array(p)).mean()

print(f"Series length: {n} | Train: {split} | Horizon: {H}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEASONAL NAIVE BASELINE
# ─────────────────────────────────────────────────────────────────────────────

period = 7   # weekly
snaive = np.tile(train[-period:], H // period + 1)[:H]
print(f"\nSeasonal Naive  | RMSE: {rmse_fn(actual, snaive):.4f} | MAE: {mae_fn(actual, snaive):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SARIMA BASELINE
# ─────────────────────────────────────────────────────────────────────────────

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    fit_sarima = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    fc_sarima  = fit_sarima.forecast(H)
    sarima_rmse = rmse_fn(actual, fc_sarima)
    print(f"SARIMA(1,1,1)(1,0,1,7) | RMSE: {sarima_rmse:.4f} | MAE: {mae_fn(actual, fc_sarima):.4f}")
    HAS_SARIMA = True
except Exception as e:
    print(f"SARIMA failed: {e}")
    HAS_SARIMA = False
    fc_sarima = snaive


# ─────────────────────────────────────────────────────────────────────────────
# 4. CHRONOS ZERO-SHOT FORECAST
# ─────────────────────────────────────────────────────────────────────────────

if HAS_CHRONOS:
    print("\n--- Loading Chronos (small) ---")
    # Downloads ~200MB model from HuggingFace on first run
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    print("Model loaded ✅")

    context = torch.tensor(train, dtype=torch.float32)

    print("\n--- Running zero-shot forecast ---")
    samples = pipeline.predict(
        context=context,
        prediction_length=H,
        num_samples=200,       # 200 trajectory samples for tight quantiles
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    # samples: (1, 200, H)
    s = samples[0].numpy()   # (200, H)

    fc_median = np.median(s, axis=0)
    fc_q10    = np.quantile(s, 0.10, axis=0)
    fc_q90    = np.quantile(s, 0.90, axis=0)

    chronos_rmse = rmse_fn(actual, fc_median)
    chronos_mae  = mae_fn(actual, fc_median)
    print(f"Chronos-small (zero-shot) | RMSE: {chronos_rmse:.4f} | MAE: {chronos_mae:.4f}")

    # Coverage: fraction of actuals inside 80% PI
    coverage = np.mean((actual >= fc_q10) & (actual <= fc_q90))
    print(f"80% PI Coverage: {coverage:.2%} (ideal ≈ 80%)")

    # Skill vs. naive
    naive_rmse  = rmse_fn(actual, snaive)
    skill = (naive_rmse - chronos_rmse) / naive_rmse
    print(f"Zero-shot skill vs. Seasonal Naive: {skill:+.3f}")


    # ─────────────────────────────────────────────────────────────────────────
    # 5. BATCH INFERENCE (MULTIPLE SERIES)
    # ─────────────────────────────────────────────────────────────────────────

    print("\n--- Batch inference (3 series) ---")
    series2 = (50 + 0.05*t + 15*np.sin(2*np.pi*t/365.25) + np.random.normal(0, 3, n)).clip(0)
    series3 = (200 + 0.15*t + 60*np.sin(2*np.pi*t/365.25) + np.random.normal(0, 10, n)).clip(0)

    batch_contexts = [
        torch.tensor(series[:split],  dtype=torch.float32),
        torch.tensor(series2[:split], dtype=torch.float32),
        torch.tensor(series3[:split], dtype=torch.float32),
    ]

    batch_samples = pipeline.predict(
        context=batch_contexts,
        prediction_length=H,
        num_samples=100,
    )
    print(f"Batch forecast shape: {batch_samples.shape}")  # (3, 100, H)
    for i in range(3):
        med = np.median(batch_samples[i].numpy(), axis=0)
        print(f"  Series {i+1} median forecast (first 5 steps): {med[:5].round(2)}")


    # ─────────────────────────────────────────────────────────────────────────
    # 6. VISUALIZATION
    # ─────────────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))

    # Forecast plot
    n_hist_show = 60
    hist_idx    = np.arange(-n_hist_show, 0)
    fore_idx    = np.arange(0, H)

    axes[0].plot(hist_idx, train[-n_hist_show:], color="black", linewidth=2, label="History")
    axes[0].plot(fore_idx, actual,   color="black", linewidth=2.5, linestyle="-", label="Actual")
    axes[0].plot(fore_idx, fc_median, color=BLUE, linewidth=2, linestyle="--",
                 label=f"Chronos Median ({chronos_rmse:.2f})")
    axes[0].fill_between(fore_idx, fc_q10, fc_q90,
                         color=BLUE, alpha=0.2, label=f"80% PI (coverage={coverage:.0%})")
    if HAS_SARIMA:
        axes[0].plot(fore_idx, fc_sarima, color=GREEN, linewidth=1.5, linestyle=":",
                     label=f"SARIMA ({sarima_rmse:.2f})")
    axes[0].plot(fore_idx, snaive, color=RED, linewidth=1.2, linestyle="--",
                 alpha=0.6, label=f"Seasonal Naive ({naive_rmse:.2f})")
    axes[0].axvline(-0.5, color="black", linewidth=0.8, linestyle=":")
    axes[0].legend(fontsize=9)
    axes[0].set_title("Chronos Zero-Shot Forecast with Prediction Intervals")

    # Sample paths from Chronos
    for i in range(20):
        axes[0].plot(fore_idx, s[i], color=BLUE, alpha=0.05, linewidth=0.8)

    # RMSE comparison
    model_names = ["Chronos\n(zero-shot)", "SARIMA" if HAS_SARIMA else "ETS",
                   "Seasonal Naive"]
    rmse_vals   = [chronos_rmse, sarima_rmse if HAS_SARIMA else naive_rmse, naive_rmse]
    bar_colors  = [BLUE, GREEN, RED]
    axes[1].barh(model_names, rmse_vals, color=bar_colors, edgecolor="white")
    axes[1].set_xlabel("RMSE (lower = better)")
    axes[1].set_title("Zero-Shot vs. Trained Baselines")
    for i, v in enumerate(rmse_vals):
        axes[1].text(v + 0.1, i, f"{v:.3f}", va="center")

    plt.suptitle("Amazon Chronos — Zero-Shot Probabilistic Forecasting",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("02_chronos_inference.png", bbox_inches="tight")
    plt.show()

else:
    print("\n[Chronos not installed — showing zero-shot concept summary]")
    print("""
Chronos Zero-Shot Pipeline:
────────────────────────────
1. Pre-trained on 100 billion TS data points (M4, M5, FRED, electricity, web...)
2. Tokenizes: real values → integer bins (quantization, B=4096 bins)
3. Uses T5 encoder-decoder backbone (language model for TS)
4. At inference:
     input: [y₁, ..., y_T] (your series — no training needed)
     output: 100+ sampled trajectories [ŷ_{T+1}, ..., ŷ_{T+H}]
5. Extract quantiles from samples → probabilistic forecast

Install: pip install chronos-forecasting
    """)

print("\n✅ Chronos inference demo complete.")
