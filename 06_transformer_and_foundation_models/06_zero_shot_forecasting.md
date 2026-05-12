# 06 — Zero-Shot Forecasting

> **Module**: 06 Transformers & Foundation Models | **File**: 6 of 7
>
> Zero-shot forecasting is the ability to produce accurate forecasts on **new, unseen time series without any task-specific training**. This note covers when it works, when it fails, how to evaluate it fairly, and the full decision framework for choosing between zero-shot and trained models.

---

## Table of Contents

1. [What is Zero-Shot Forecasting?](#1-what-is-zero-shot-forecasting)
2. [Few-Shot Forecasting](#2-few-shot-forecasting)
3. [When Zero-Shot Works (and When It Fails)](#3-when-zero-shot-works-and-when-it-fails)
4. [Fair Evaluation Protocol](#4-fair-evaluation-protocol)
5. [Zero-Shot vs. Few-Shot vs. Fine-Tuned Comparison](#5-zero-shot-vs-few-shot-vs-fine-tuned-comparison)
6. [Production Decision Framework](#6-production-decision-framework)
7. [Practical Zero-Shot Pipeline](#7-practical-zero-shot-pipeline)

---

## 1. What is Zero-Shot Forecasting?

```
Traditional ML pipeline:
  For each new dataset:
    collect → engineer features → train → tune → deploy
    (weeks of work)

Zero-shot pipeline:
  For each new dataset:
    collect → call pre-trained model → deploy
    (hours of work)

Analogy:
  Zero-shot forecasting is to time series
  what GPT-4 is to text generation:
  One pre-trained model handles any input.
```

### 1.1 How Zero-Shot is Possible

Foundation models learn universal temporal patterns from massive training corpora:

```
What the model learns during pre-training:
  ✅ Trend: "Values that grow linearly → they'll keep growing"
  ✅ Seasonality: "Weekly patterns repeat → next week same as last"
  ✅ Volatility: "Spiky series → wide prediction intervals"
  ✅ Level shift: "Sudden jump → new level likely persists"
  ✅ Regime change: "Old trend reversed → adapt"
  
What it CANNOT learn zero-shot:
  ❌ Domain-specific causality ("Hurricane → demand spike next week")
  ❌ Your specific promotion calendar
  ❌ Company-specific seasonal patterns not in training data
```

### 1.2 Zero-Shot vs. In-Context Learning

```
Zero-shot:
  Input: [y₁, ..., y_L]  (context window only)
  Output: [ŷ_{L+1}, ..., ŷ_{L+H}]
  
In-context learning (few-shot without gradient updates):
  Input: [example_series_1 | example_series_2 | target_series]
  → The model uses the examples to calibrate its predictions
  → No gradient updates; purely based on attention patterns
  
Fine-tuning (technically not zero-shot anymore):
  Update model weights on task-specific data
  → Requires compute; but better for specialized domains
```

---

## 2. Few-Shot Forecasting

Few-shot forecasting provides a small number of in-context examples to guide the model:

### 2.1 In-Context Examples Strategy

```python
import torch
import numpy as np
from chronos import ChronosPipeline

def few_shot_with_examples(
    pipeline: ChronosPipeline,
    target_series: np.ndarray,
    example_series_list: list,
    prediction_length: int,
    num_samples: int = 100,
):
    """
    Few-shot forecasting: prepend example series before the target.
    
    Concatenates example series (as additional context) to guide the model.
    This works because the transformer attends to ALL context — including
    the examples — when generating the forecast.
    
    Parameters:
        target_series:       the series to forecast (1D numpy array)
        example_series_list: list of 1D numpy arrays (similar series)
        prediction_length:   forecast horizon
    
    NOTE: This simple concatenation is an approximation of true in-context
          learning. True ICL would require interleaved example+label pairs.
    """
    # Concatenate examples before target (bounded by model context length)
    # Use a separator-like pattern (e.g., repeat last value) between series
    separator = np.full(3, target_series.mean())   # 3-step flat separator

    context_parts = []
    for ex in example_series_list:
        context_parts.append(ex)
        context_parts.append(separator)
    context_parts.append(target_series)

    combined = np.concatenate(context_parts)

    # Trim to model max context (chronos-t5-small: 512; large: 2048)
    max_ctx = 512
    if len(combined) > max_ctx:
        combined = combined[-max_ctx:]

    context_tensor = torch.tensor(combined, dtype=torch.float32)
    forecasts = pipeline.predict(
        context=context_tensor,
        prediction_length=prediction_length,
        num_samples=num_samples,
    )
    return forecasts[0].numpy()  # (num_samples, H)


# Usage example
np.random.seed(42)
n = 300
t = np.arange(n)
target = 100 + 0.1*t + 20*np.sin(2*np.pi*t/52) + np.random.normal(0, 5, n)

# Two related series as examples (same seasonal pattern, different scale)
example1 = 200 + 0.2*t + 40*np.sin(2*np.pi*t/52) + np.random.normal(0, 8, n)
example2 = 50  + 0.05*t + 10*np.sin(2*np.pi*t/52) + np.random.normal(0, 3, n)

# Few-shot forecast
# pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", ...)
# samples = few_shot_with_examples(pipeline, target, [example1, example2], H=24)
# median = np.median(samples, axis=0)
```

---

## 3. When Zero-Shot Works (and When It Fails)

### 3.1 Strong Zero-Shot Scenarios

| Scenario | Why It Works |
|----------|-------------|
| **Standard seasonal patterns** | Weekly/yearly cycles are ubiquitous in training data |
| **Linear trends** | Growth trends generalize well |
| **Cold start** (new product) | Only option when no history available |
| **Many short series** | Zero-shot avoids overfitting small datasets |
| **Rapid prototyping** | Instant baseline without engineering effort |
| **Highly irregular data** | Pre-trained models often handle noise better than overfitted ML |

### 3.2 Where Zero-Shot Fails

```
Failure modes:

1. Domain shift — Data pattern not in training corpus
   Example: "Specialized industrial sensor with unique failure signature"
   Sign: Zero-shot RMSE > seasonal naive

2. Covariate dependency — Forecast depends on known future events
   Example: "Sales spike when price drops — zero-shot doesn't know price"
   Fix: Fine-tune with covariates, or use TFT

3. Very long horizons — Pre-trained context window too short
   Example: H = 720 steps — exceeds chronos context length
   Fix: Use PatchTST or N-HiTS instead

4. Non-standard frequency — Training data lacked this frequency
   Example: 5-minute IoT sensor data
   Sign: Poor accuracy across all horizon lengths

5. Extreme value scales — Series with unusual magnitudes
   (Chronos normalizes per-instance, but very heavy tails can cause issues)
```

### 3.3 Diagnostic Test

```python
import numpy as np

def zero_shot_diagnostic(
    actual: np.ndarray,
    zero_shot_fc: np.ndarray,
    snaive_fc: np.ndarray,
) -> dict:
    """
    Quick diagnostic to decide if zero-shot is good enough.
    
    Rule of thumb:
      skill > 0.05  → Zero-shot adds meaningful value
      skill < 0     → Zero-shot is WORSE than naive (use classical instead)
    """
    rmse = lambda a, p: np.sqrt(((np.array(a) - np.array(p))**2).mean())

    rmse_naive    = rmse(actual, snaive_fc)
    rmse_zero     = rmse(actual, zero_shot_fc)

    # Forecast skill relative to seasonal naive (MASE-like)
    skill = (rmse_naive - rmse_zero) / (rmse_naive + 1e-8)

    print(f"Seasonal Naive RMSE: {rmse_naive:.4f}")
    print(f"Zero-Shot RMSE:      {rmse_zero:.4f}")
    print(f"Zero-Shot Skill:     {skill:+.3f}")

    if skill > 0.10:
        verdict = "✅ Zero-shot is strong (>10% improvement over naive)"
    elif skill > 0:
        verdict = "⚠️  Zero-shot is marginal (0–10% improvement)"
    else:
        verdict = "❌ Zero-shot underperforms naive — train a classical model"

    print(f"\nVerdict: {verdict}")
    return {"skill": skill, "rmse_zero": rmse_zero, "rmse_naive": rmse_naive}
```

---

## 4. Fair Evaluation Protocol

A common mistake is evaluating zero-shot models unfairly. Here's the correct protocol:

### 4.1 Correct Protocol

```python
import numpy as np
import pandas as pd

def fair_zero_shot_evaluation(
    series:              np.ndarray,
    horizon:             int = 30,
    n_windows:           int = 3,
    step_size:           int = None,
    pipeline=None,       # Chronos or any zero-shot model
    n_samples:           int = 100,
) -> pd.DataFrame:
    """
    Walk-forward evaluation of zero-shot model with multiple windows.
    
    IMPORTANT: Zero-shot model sees NO TEST data — each window is evaluated
    on an unseen slice. This mirrors real deployment conditions.
    
    Parameters:
        series:   full time series (numpy array)
        horizon:  forecast steps to evaluate per window
        n_windows: number of evaluation windows
        step_size: how far to advance origin per window (default: horizon)
    
    Returns:
        DataFrame with RMSE per window and model
    """
    if step_size is None:
        step_size = horizon

    n = len(series)
    # Earliest possible test start: ensure training has at least 2× horizon history
    min_train = 2 * horizon
    test_starts = []

    # Walk backward from end to determine test origins
    for i in range(n_windows):
        test_end   = n - i * step_size
        test_start = test_end - horizon
        if test_start < min_train:
            break
        test_starts.append(test_start)

    test_starts = sorted(test_starts)  # chronological order

    records = []
    rmse_fn = lambda a, p: np.sqrt(((np.array(a) - np.array(p))**2).mean())

    for window_idx, test_start in enumerate(test_starts):
        train = series[:test_start]
        actual = series[test_start:test_start + horizon]

        # Seasonal naive baseline
        period = 7
        snaive = np.tile(train[-period:], horizon // period + 1)[:horizon]
        records.append({"window": window_idx + 1, "model": "Seasonal Naive",
                        "rmse": rmse_fn(actual, snaive)})

        # Zero-shot model
        if pipeline is not None:
            import torch
            ctx = torch.tensor(train, dtype=torch.float32)
            samples = pipeline.predict(ctx, prediction_length=horizon,
                                       num_samples=n_samples)
            fc = np.median(samples[0].numpy(), axis=0)
            records.append({"window": window_idx + 1, "model": "Zero-Shot (Chronos)",
                            "rmse": rmse_fn(actual, fc)})

        print(f"Window {window_idx+1}: test[{test_start}:{test_start+horizon}] done")

    df = pd.DataFrame(records)
    pivot = df.pivot(index="window", columns="model", values="rmse")
    print("\nWalk-Forward RMSE by Window:")
    print(pivot.to_string(float_format="{:.4f}".format))
    print(f"\nMean RMSE:")
    print(pivot.mean().to_string(float_format="{:.4f}".format))
    return pivot
```

### 4.2 Common Evaluation Mistakes

| Mistake | Problem | Correct Approach |
|---------|---------|-----------------|
| Single test window | High variance estimate | Use ≥ 3 walk-forward windows |
| Optimize on test set | Information leakage | Chronos is parameter-free — no tuning needed |
| Compare to MAE-tuned SARIMA vs. Chronos RMSE | Unfair metric | Use same metric for all models |
| Include test period in context | Data leakage | Context must end strictly before test start |
| Count fine-tuning as "zero-shot" | Misleading | Fine-tuned models evaluated separately |

---

## 5. Zero-Shot vs. Few-Shot vs. Fine-Tuned Comparison

```python
import numpy as np
import pandas as pd

def model_selection_benchmark(series: np.ndarray, H: int = 30) -> pd.DataFrame:
    """
    Comprehensive benchmark: zero-shot → few-shot → fine-tuned → classical
    
    Use this to decide which approach is worth the investment.
    """
    split  = len(series) - H
    train  = series[:split]
    actual = series[split:]
    period = 7   # adjust for your frequency

    rmse = lambda a, p: np.sqrt(((np.array(a) - np.array(p))**2).mean())
    results = {}

    # ── 1. Seasonal Naive (free) ─────────────────────────────────────────────
    snaive = np.tile(train[-period:], H // period + 1)[:H]
    results["Seasonal Naive"] = {"rmse": rmse(actual, snaive), "cost": "free",
                                  "setup_time": "1 min"}

    # ── 2. ETS (statsmodels) ─────────────────────────────────────────────────
    try:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        fit = ETSModel(train, error="add", trend="add", seasonal="add",
                       seasonal_periods=period).fit(disp=False)
        fc_ets = fit.forecast(H)
        results["ETS"] = {"rmse": rmse(actual, fc_ets), "cost": "free",
                           "setup_time": "2 min"}
    except Exception:
        pass

    # ── 3. Zero-Shot Chronos ─────────────────────────────────────────────────
    # (Uncomment when chronos-forecasting is installed)
    # import torch
    # from chronos import ChronosPipeline
    # pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", ...)
    # samples = pipe.predict(torch.tensor(train, dtype=torch.float32),
    #                        prediction_length=H, num_samples=100)
    # fc_ch = np.median(samples[0].numpy(), axis=0)
    # results["Chronos-small (zero-shot)"] = {
    #     "rmse": rmse(actual, fc_ch), "cost": "free (open weights)",
    #     "setup_time": "15 min (download)"
    # }

    # ── 4. Fine-tuned Chronos ─────────────────────────────────────────────────
    # After fine-tuning on similar series
    # results["Chronos-small (fine-tuned)"] = {
    #     "rmse": rmse(actual, fc_ft), "cost": "GPU compute",
    #     "setup_time": "2–4 hours"
    # }

    # ── Summary ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(results).T
    df["rmse"] = df["rmse"].astype(float)
    df = df.sort_values("rmse")
    print("\n" + "="*65)
    print("Model Selection Benchmark")
    print("="*65)
    print(df.to_string())
    print("="*65)
    return df
```

---

## 6. Production Decision Framework

```
                    ┌─────────────────────────────────────────────┐
                    │  Is history available? (>= 2× horizon steps) │
                    └──────────────────┬──────────────────────────┘
                                       │
              ┌────────────────────────┴────────────────────┐
              │ NO (cold start)                              │ YES
              ▼                                              ▼
    ┌──────────────────────┐              ┌─────────────────────────────────┐
    │ TimeGPT zero-shot    │              │ Is domain standard?             │
    │ Chronos zero-shot    │              │ (retail, weather, energy, web)  │
    └──────────────────────┘              └──────────────────┬──────────────┘
                                                             │
                                          ┌──────────────────┴───────────────┐
                                          │ YES                               │ NO
                                          ▼                                   ▼
                              ┌─────────────────────┐       ┌──────────────────────────┐
                              │ Zero-shot Chronos   │       │ Train LightGBM or SARIMA │
                              │ → evaluate skill    │       │ (custom feature eng.)    │
                              └──────────┬──────────┘       └──────────────────────────┘
                                         │
                              ┌──────────┴──────────┐
                              │ Skill > 0.05?       │
                              └──────────┬──────────┘
                         ┌───────────────┴──────────────────┐
                         │ YES                               │ NO
                         ▼                                   ▼
              ┌───────────────────────┐       ┌──────────────────────────────┐
              │ Deploy zero-shot      │       │ Train classical (SARIMA/ETS) │
              │ → monitor skill drift │       │ or LightGBM with lag features│
              └───────────────────────┘       └──────────────────────────────┘
```

### 6.1 Quick Decision Checklist

```
Before using a zero-shot model in production, confirm:

[ ] Diagnostic test: skill > 0 vs. seasonal naive
[ ] Walk-forward eval on ≥ 3 windows (not just one hold-out)
[ ] Latency acceptable (API round-trip or local GPU inference)
[ ] Context length sufficient (your series length ≤ model max context)
[ ] Frequency is standard (avoid 3-min, 10-day, bi-weekly — poorly covered)
[ ] No critical covariates that the model can't see
[ ] Regulatory/explainability requirements met
[ ] Monitoring in place (detect when zero-shot degrades post-deployment)
```

---

## 7. Practical Zero-Shot Pipeline

End-to-end production-ready zero-shot pipeline:

```python
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

@dataclass
class ZeroShotForecast:
    """Container for a zero-shot forecast result."""
    model_name:  str
    horizon:     int
    median:      np.ndarray    # (H,)
    lower:       np.ndarray    # (H,) — lower PI bound
    upper:       np.ndarray    # (H,) — upper PI bound
    skill_vs_naive: Optional[float] = None


class ZeroShotPipeline:
    """
    Production-ready zero-shot forecasting pipeline using Amazon Chronos.

    Usage:
        pipeline = ZeroShotPipeline(model_size="small")
        result   = pipeline.forecast(series, horizon=30, eval_naive=True)
        pipeline.plot(series, result)
    """

    CHRONOS_MODELS = {
        "tiny":  "amazon/chronos-t5-tiny",
        "mini":  "amazon/chronos-t5-mini",
        "small": "amazon/chronos-t5-small",
        "base":  "amazon/chronos-t5-base",
        "large": "amazon/chronos-t5-large",
    }

    def __init__(self, model_size: str = "small", num_samples: int = 100):
        try:
            from chronos import ChronosPipeline
            self.pipeline = ChronosPipeline.from_pretrained(
                self.CHRONOS_MODELS[model_size],
                device_map="auto",
                torch_dtype=torch.float32,
            )
            self.num_samples = num_samples
            self.model_name  = f"Chronos-{model_size}"
            print(f"✅ {self.model_name} loaded")
        except ImportError:
            raise ImportError("Run: pip install chronos-forecasting")

    def forecast(
        self,
        series:     np.ndarray,
        horizon:    int,
        pi_level:   float = 0.8,   # 80% prediction interval
        eval_naive: bool  = True,
        period:     int   = 7,
    ) -> ZeroShotForecast:
        """
        Generate a zero-shot probabilistic forecast.
        
        Parameters:
            series:     1D numpy array (complete historical series)
            horizon:    steps to forecast
            pi_level:   prediction interval coverage (0–1)
            eval_naive: compute skill vs seasonal naive on held-out portion
            period:     seasonal period for naive baseline evaluation
        """
        context   = torch.tensor(series, dtype=torch.float32)
        samples   = self.pipeline.predict(context, prediction_length=horizon,
                                          num_samples=self.num_samples)
        s         = samples[0].numpy()  # (num_samples, H)

        alpha  = (1 - pi_level) / 2
        median = np.median(s, axis=0)
        lower  = np.quantile(s, alpha, axis=0)
        upper  = np.quantile(s, 1 - alpha, axis=0)

        skill = None
        if eval_naive and len(series) > horizon + period:
            # Evaluate on last `horizon` steps of training series
            train_for_eval = series[:-horizon]
            actual         = series[-horizon:]
            ctx_eval       = torch.tensor(train_for_eval, dtype=torch.float32)
            s_eval         = self.pipeline.predict(ctx_eval, prediction_length=horizon,
                                                   num_samples=self.num_samples)[0].numpy()
            fc_eval = np.median(s_eval, axis=0)

            snaive_eval = np.tile(train_for_eval[-period:],
                                  horizon // period + 1)[:horizon]
            rmse = lambda a, p: np.sqrt(((a - p)**2).mean())
            skill = (rmse(actual, snaive_eval) - rmse(actual, fc_eval)) / \
                    (rmse(actual, snaive_eval) + 1e-8)
            print(f"Zero-shot skill vs. seasonal naive: {skill:+.3f}")

        return ZeroShotForecast(self.model_name, horizon, median, lower, upper, skill)

    def plot(self, series: np.ndarray, result: ZeroShotForecast,
             n_history: int = 60):
        """Plot forecast with prediction interval."""
        n = len(series)
        hist_idx  = np.arange(n - n_history, n)
        fore_idx  = np.arange(n, n + result.horizon)

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(hist_idx, series[-n_history:],
                color="black", linewidth=2, label="History")
        ax.plot(fore_idx, result.median, color="#D7191C", linewidth=2,
                linestyle="--", label=f"{result.model_name} Median")
        ax.fill_between(fore_idx, result.lower, result.upper,
                        color="#D7191C", alpha=0.2, label="80% PI")
        ax.axvline(n - 0.5, color="black", linewidth=0.8, linestyle=":")

        title = f"Zero-Shot Forecast — {result.model_name}"
        if result.skill_vs_naive is not None:
            title += f"  |  Skill vs. Naive: {result.skill_vs_naive:+.2f}"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig("06_zero_shot_forecast.png", bbox_inches="tight")
        plt.show()
```

---

*← [05 — Moirai & Chronos](./05_moirai_chronos_foundation_models.md) | [Module README](./README.md) | Next: [07 — Fine-Tuning TS LLMs](./07_fine_tuning_ts_llms.md) →*
