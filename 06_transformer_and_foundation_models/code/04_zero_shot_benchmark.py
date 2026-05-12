"""
04_zero_shot_benchmark.py
==========================
Module 06 — Transformers & Foundation Models
Topic   : Zero-Shot Forecasting Benchmark

Covers:
  - Rigorous walk-forward evaluation of zero-shot models
  - Chronos vs. SARIMA vs. ETS vs. Seasonal Naive
  - Skill score calculation
  - Probabilistic calibration (prediction interval coverage)
  - Decision framework for zero-shot vs. trained models
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
BLUE, RED, GREEN, ORANGE, PURPLE = (
    "#2C7BB6", "#D7191C", "#1A9641", "#F07D00", "#7B2D8B"
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATASETS (3 DIFFERENT PATTERN TYPES)
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)

def make_series(pattern: str, n: int = 730) -> np.ndarray:
    """Create synthetic series with different complexity levels."""
    t = np.arange(n, dtype=np.float32)
    if pattern == "simple":
        # Clear weekly + annual seasonality
        return (100 + 0.1*t
                + 25*np.sin(2*np.pi*t/365.25)
                + 10*np.sin(2*np.pi*t/7)
                + np.random.normal(0, 4, n))
    elif pattern == "complex":
        # Multiple irregularly-spaced seasonalities + trend change
        series = (200 + np.where(t < n//2, 0.2*t, 0.05*t + 0.15*n//2)
                  + 40*np.sin(2*np.pi*t/365.25)
                  + 15*np.sin(2*np.pi*t/7)
                  + 8*np.sin(2*np.pi*t/30.4)
                  + np.random.normal(0, 10, n))
        return series.clip(min=0)
    elif pattern == "noisy":
        # High noise, weak signal
        return (50 + 0.05*t
                + 5*np.sin(2*np.pi*t/365.25)
                + np.random.normal(0, 20, n))
    raise ValueError(f"Unknown pattern: {pattern}")

DATASETS = {
    "simple_seasonal": make_series("simple"),
    "complex_multi":   make_series("complex"),
    "high_noise":      make_series("noisy"),
}
H = 30   # forecast horizon for all datasets


# ─────────────────────────────────────────────────────────────────────────────
# 2. EVALUATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def rmse(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted))**2))

def mae(actual, predicted):
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))

def skill_vs_naive(actual, predicted, naive):
    """Skill score: +1=perfect, 0=same as naive, <0=worse than naive."""
    return 1 - rmse(actual, predicted) / (rmse(actual, naive) + 1e-8)

def pi_coverage(actual, lower, upper):
    """Fraction of actuals inside prediction interval (should equal nominal level)."""
    return np.mean((np.array(actual) >= np.array(lower)) &
                   (np.array(actual) <= np.array(upper)))


# ─────────────────────────────────────────────────────────────────────────────
# 3. BASELINE MODELS
# ─────────────────────────────────────────────────────────────────────────────

def seasonal_naive_forecast(train: np.ndarray, horizon: int, period: int = 7):
    return np.tile(train[-period:], horizon // period + 1)[:horizon]

def ets_forecast(train: np.ndarray, horizon: int, period: int = 7):
    try:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        model = ETSModel(
            train, error="add", trend="add", seasonal="add",
            seasonal_periods=period, initialization_method="estimated",
        )
        fit = model.fit(disp=False)
        return fit.forecast(horizon)
    except Exception as e:
        print(f"  ETS failed: {e}")
        return seasonal_naive_forecast(train, horizon, period)

def sarima_forecast(train: np.ndarray, horizon: int, period: int = 7):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        fit = SARIMAX(
            train, order=(1, 1, 1), seasonal_order=(1, 0, 1, period),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        return fit.forecast(horizon)
    except Exception as e:
        print(f"  SARIMA failed: {e}")
        return seasonal_naive_forecast(train, horizon, period)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CHRONOS ZERO-SHOT
# ─────────────────────────────────────────────────────────────────────────────

def chronos_forecast(pipeline, train: np.ndarray, horizon: int,
                     n_samples: int = 100):
    """Returns (median, q10, q90)."""
    ctx     = torch.tensor(train, dtype=torch.float32)
    samples = pipeline.predict(ctx, prediction_length=horizon,
                               num_samples=n_samples)
    s = samples[0].numpy()
    return np.median(s, axis=0), np.quantile(s, 0.1, axis=0), np.quantile(s, 0.9, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. WALK-FORWARD BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    rmse_per_window:  list = field(default_factory=list)
    skill_per_window: list = field(default_factory=list)
    coverage:         Optional[float] = None  # PI coverage (probabilistic models)

def walk_forward_benchmark(
    series: np.ndarray,
    H:      int,
    n_windows: int = 3,
    period:    int = 7,
    pipeline=None,
) -> pd.DataFrame:
    """
    Walk-forward evaluation comparing zero-shot Chronos vs. baselines.
    
    Each window: train on everything before test_start, test on [test_start, test_start+H]
    Windows are spaced H steps apart, starting from the end.
    """
    n = len(series)
    min_train = max(4 * H, 4 * period)

    test_starts = sorted([
        n - H - i * H
        for i in range(n_windows)
        if (n - H - i * H) >= min_train
    ])

    records = []

    for window_idx, ts in enumerate(test_starts):
        train  = series[:ts]
        actual = series[ts:ts + H]

        naive = seasonal_naive_forecast(train, H, period)

        # Evaluate each model
        models_fc = {
            "Seasonal Naive": naive,
            "ETS":            ets_forecast(train, H, period),
            "SARIMA":         sarima_forecast(train, H, period),
        }

        if pipeline is not None:
            chronos_med, q10, q90 = chronos_forecast(pipeline, train, H)
            models_fc["Chronos (zero-shot)"] = chronos_med
            coverage = pi_coverage(actual, q10, q90)
        else:
            coverage = None

        for model_name, fc in models_fc.items():
            records.append({
                "window":  window_idx + 1,
                "model":   model_name,
                "rmse":    rmse(actual, fc),
                "mae":     mae(actual, fc),
                "skill":   skill_vs_naive(actual, fc, naive),
            })

        if coverage is not None:
            # Add coverage for Chronos separately
            for rec in records[-4:]:
                if rec["model"] == "Chronos (zero-shot)":
                    rec["pi_coverage_80pct"] = coverage

        print(f"  Window {window_idx+1} (test[{ts}:{ts+H}]) done")

    df = pd.DataFrame(records)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. RUN BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

try:
    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    print("Chronos loaded ✅")
except ImportError:
    pipeline = None
    print("Chronos not installed — benchmarking without it.")
    print("Run: pip install chronos-forecasting")

all_results = {}
for ds_name, series in DATASETS.items():
    print(f"\n{'='*55}")
    print(f"Dataset: {ds_name}  (n={len(series)})")
    print('='*55)
    df_res = walk_forward_benchmark(series, H, n_windows=3, period=7,
                                    pipeline=pipeline)
    all_results[ds_name] = df_res

    # Summary table per dataset
    summary = df_res.groupby("model")[["rmse", "skill"]].mean()
    summary = summary.sort_values("rmse")
    print(f"\n  Mean RMSE & Skill over {3} windows:")
    print(summary.to_string(float_format="{:.4f}".format))


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(DATASETS), figsize=(14, 6), sharey=False)

model_colors = {
    "Seasonal Naive":    RED,
    "ETS":               ORANGE,
    "SARIMA":            GREEN,
    "Chronos (zero-shot)": BLUE,
}

for ax, (ds_name, df_res) in zip(axes, all_results.items()):
    summary = df_res.groupby("model")["rmse"].mean().sort_values()
    models  = summary.index.tolist()
    values  = summary.values
    colors  = [model_colors.get(m, PURPLE) for m in models]

    bars = ax.bar(range(len(models)), values, color=colors, edgecolor="white")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=8)
    ax.set_title(ds_name.replace("_", "\n"), fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean RMSE" if ax is axes[0] else "")

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                f"{v:.2f}", ha="center", fontsize=8)

plt.suptitle("Zero-Shot Benchmark: Chronos vs. Classical Models\n(Walk-Forward, 3 Windows)",
             fontweight="bold")
plt.tight_layout()
plt.savefig("04_zero_shot_benchmark.png", bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 8. DECISION FRAMEWORK OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ZERO-SHOT DECISION FRAMEWORK")
print("="*60)
for ds_name, df_res in all_results.items():
    if "Chronos (zero-shot)" in df_res["model"].values:
        skill = df_res[df_res["model"] == "Chronos (zero-shot)"]["skill"].mean()
        if skill > 0.10:
            verdict = "✅ Deploy zero-shot (skill > 10%)"
        elif skill > 0:
            verdict = "⚠️  Marginal (0–10%) — consider fine-tuning"
        else:
            verdict = "❌ Zero-shot underperforms — train classical model"
        print(f"  {ds_name:<25} skill={skill:+.3f}  →  {verdict}")
    else:
        print(f"  {ds_name:<25} (Chronos not available)")

print("\n✅ Zero-shot benchmark complete.")
