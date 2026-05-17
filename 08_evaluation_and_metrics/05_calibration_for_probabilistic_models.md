# 05 — Calibration for Probabilistic Models

> **Module**: 08 Evaluation & Metrics | **File**: 5 of 5
>
> A probabilistic forecast is only useful if it is **calibrated** — its stated confidence levels must match the empirical frequencies. This note covers CRPS, coverage diagnostics, reliability diagrams, sharpness, and the full evaluation pipeline for distributional forecasts.

---

## Table of Contents

1. [The Two Goals: Calibration and Sharpness](#1-the-two-goals-calibration-and-sharpness)
2. [Coverage Metrics](#2-coverage-metrics)
3. [Reliability Diagrams](#3-reliability-diagrams)
4. [CRPS — Continuous Ranked Probability Score](#4-crps--continuous-ranked-probability-score)
5. [Winkler Score and Interval Score](#5-winkler-score-and-interval-score)
6. [Pinball Loss (Quantile Loss)](#6-pinball-loss-quantile-loss)
7. [Sharpness](#7-sharpness)
8. [Full Probabilistic Evaluation Pipeline](#8-full-probabilistic-evaluation-pipeline)

---

## 1. The Two Goals: Calibration and Sharpness

### 1.1 Calibration

A forecast is **calibrated** if the stated probabilities match empirical frequencies:

```
If you say "90% prediction interval" for 100 different forecasts,
~90 of the actual values should fall inside those intervals.

Perfect calibration:
  P(y ∈ [qₐ, q_{1-α}]) = 1 - α   for all α ∈ (0, 1)

Undercoverage (overconfident):
  90% PI contains only 75% of actuals → intervals are too narrow
  → Model underestimates uncertainty

Overcoverage (underconfident):
  90% PI contains 98% of actuals → intervals are too wide
  → Forecasts are useless (could always use ±∞)
```

### 1.2 Sharpness

A forecast is **sharp** if its prediction intervals are narrow while remaining calibrated:

```
Trade-off:
  Wide intervals:   always calibrated (trivially), useless for decisions
  Narrow intervals: hard to maintain calibration, but more decision value

Goal: maximize sharpness SUBJECT TO calibration
  (not at the expense of calibration)

Sharpness = average interval width (lower = sharper)
```

### 1.3 The Murphy Decomposition

Any proper scoring rule (like CRPS) decomposes into:

```
Score = Calibration Error + Sharpness Penalty

Perfect score = 0 (when forecast = true distribution)
A sharp but miscalibrated model may score WORSE than a wide, calibrated one.
```

---

## 2. Coverage Metrics

### 2.1 Marginal Coverage

```
Coverage(α) = fraction of test observations where yₜ ∈ [q̂ₐ(xₜ), q̂_{1-α}(xₜ)]

Target: Coverage(α) ≈ 1 - α

Example: 90% PI → Coverage should be ≈ 0.90
```

### 2.2 Implementation

```python
import numpy as np
import pandas as pd

def coverage_at_level(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal: float,
) -> dict:
    """
    Compute empirical coverage for a single prediction interval level.

    Parameters
    ----------
    y_true  : actual test values (T,)
    lower   : lower bound of prediction interval (T,)
    upper   : upper bound of prediction interval (T,)
    nominal : stated coverage level (e.g., 0.90 for 90% PI)

    Returns
    -------
    dict with coverage statistics
    """
    y_true = np.asarray(y_true, dtype=float)
    lower  = np.asarray(lower, dtype=float)
    upper  = np.asarray(upper, dtype=float)

    in_interval     = (y_true >= lower) & (y_true <= upper)
    empirical_cov   = float(in_interval.mean())
    coverage_error  = empirical_cov - nominal

    return {
        "nominal_coverage":   nominal,
        "empirical_coverage": round(empirical_cov, 4),
        "coverage_error":     round(coverage_error, 4),
        "n_in_interval":      int(in_interval.sum()),
        "n_total":            len(y_true),
        "calibrated":         abs(coverage_error) < 0.02,  # within 2%
        "overconfident":      coverage_error < -0.02,      # too narrow
        "underconfident":     coverage_error > 0.02,       # too wide
    }


def coverage_across_levels(
    y_true: np.ndarray,
    quantile_preds: dict,
    levels: list = None,
) -> pd.DataFrame:
    """
    Compute coverage for multiple PI levels simultaneously.

    Parameters
    ----------
    y_true         : actual values (T,)
    quantile_preds : dict {tau: predictions}, e.g. {0.05: array, 0.95: array}
    levels         : list of PI levels (e.g., [0.80, 0.90, 0.95])
                     requires symmetric quantiles (0.1/0.9 for 80%, etc.)

    Returns
    -------
    DataFrame with one row per PI level
    """
    if levels is None:
        levels = [0.50, 0.80, 0.90, 0.95]

    rows = []
    for level in levels:
        alpha = 1 - level
        tau_lo = alpha / 2
        tau_hi = 1 - alpha / 2

        if tau_lo not in quantile_preds or tau_hi not in quantile_preds:
            continue

        lo  = quantile_preds[tau_lo]
        hi  = quantile_preds[tau_hi]
        res = coverage_at_level(y_true, lo, hi, nominal=level)
        rows.append({
            "PI Level":           f"{int(level*100)}%",
            "Nominal":            level,
            "Empirical":          res["empirical_coverage"],
            "Error":              res["coverage_error"],
            "Status":             ("✅ OK" if res["calibrated"] else
                                   ("⚠ Overconfident" if res["overconfident"] else "⚠ Underconfident")),
            "Mean Width":         float((np.asarray(hi) - np.asarray(lo)).mean()),
        })

    return pd.DataFrame(rows)
```

---

## 3. Reliability Diagrams

### 3.1 What They Show

A **reliability diagram** (also called a calibration plot) plots nominal coverage vs. empirical coverage across all quantile levels. A perfectly calibrated model gives a diagonal line:

```
Perfect calibration:
  Nominal 10% → Empirical 10%  ●
  Nominal 20% → Empirical 20%  ●
  ...
  Nominal 90% → Empirical 90%  ●
  → All points on the diagonal

Overconfident (narrow intervals):
  All points below the diagonal → actual coverage < nominal
  Model is more uncertain than it admits

Underconfident (wide intervals):
  All points above the diagonal → actual coverage > nominal
```

### 3.2 Implementation

```python
import matplotlib.pyplot as plt
import numpy as np

def reliability_diagram(
    y_true: np.ndarray,
    quantile_preds: dict,
    title: str = "Reliability Diagram (Calibration Plot)",
    figsize: tuple = (7, 7),
):
    """
    Plot nominal vs. empirical coverage across all available quantile levels.

    Parameters
    ----------
    y_true         : actual test values (T,)
    quantile_preds : dict {tau → predictions_array} for all quantile levels
    """
    y_true = np.asarray(y_true, dtype=float)
    taus   = sorted(quantile_preds.keys())

    nominal_levels   = []
    empirical_levels = []

    for tau in taus:
        pred = np.asarray(quantile_preds[tau], dtype=float)
        emp  = float((y_true <= pred).mean())   # fraction below this quantile
        nominal_levels.append(tau)
        empirical_levels.append(emp)

    fig, ax = plt.subplots(figsize=figsize)

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration", zorder=1)

    # Model calibration curve
    ax.plot(nominal_levels, empirical_levels, "o-",
            color="#2196F3", linewidth=2.5, markersize=7,
            label="Model", zorder=3)

    # Shading for over/under confidence regions
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05,
                    color="orange", label="Underconfident region")
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.05,
                    color="red", label="Overconfident region")

    ax.set_xlabel("Nominal Coverage (Quantile Level)", fontsize=12)
    ax.set_ylabel("Empirical Coverage", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Calibration error summary
    cal_errors = np.abs(np.array(empirical_levels) - np.array(nominal_levels))
    print(f"Mean Absolute Calibration Error (MACE): {cal_errors.mean():.4f}")
    print(f"Max  Absolute Calibration Error:        {cal_errors.max():.4f}")
    return {
        "mace":     float(cal_errors.mean()),
        "max_error": float(cal_errors.max()),
    }
```

---

## 4. CRPS — Continuous Ranked Probability Score

### 4.1 Definition

CRPS is the gold-standard metric for evaluating probabilistic forecasts. It measures the **distance between the forecast CDF and the empirical CDF** of the observation:

```
CRPS(F, y) = ∫₋∞^∞ [F(z) - 1{z ≥ y}]² dz

Where:
  F(z)    = forecast CDF evaluated at z
  1{z≥y}  = empirical CDF of the observation y (Heaviside function)
  y       = actual observed value

Properties:
  - Strictly proper scoring rule (minimized only by true distribution)
  - Equals MAE when F is a point forecast (degenerate distribution)
  - Lower = better
  - Same units as the target variable
  - Decomposes into Calibration + Sharpness components
```

### 4.2 Analytical CRPS for Gaussian Forecasts

```python
import numpy as np
from scipy.stats import norm

def crps_gaussian(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Analytical CRPS for Gaussian forecast distributions.

    CRPS(N(μ,σ²), y) = σ · [z(Φ(z) - 1) + 2φ(z) - 1/√π]

    Where z = (y - μ) / σ, Φ = normal CDF, φ = normal PDF.

    Parameters
    ----------
    mu    : predicted means (T,)
    sigma : predicted standard deviations (T,) — must be > 0
    y     : actual values (T,)

    Returns
    -------
    crps : per-observation CRPS (T,) — lower = better
    """
    mu    = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    y     = np.asarray(y, dtype=float)

    z    = (y - mu) / (sigma + 1e-12)
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return crps


def crps_from_quantiles(
    y_true: np.ndarray,
    quantile_preds: dict,
) -> np.ndarray:
    """
    Empirical CRPS approximation from quantile forecasts using the PWM estimator.

    CRPS ≈ 2 · Σₖ wₖ · L_τₖ(y, q̂ₖ)

    Where L_τ is the pinball loss and wₖ are integration weights.

    Parameters
    ----------
    y_true         : actual values (T,)
    quantile_preds : dict {tau: predictions_array}

    Returns
    -------
    crps : per-observation CRPS approximation (T,)
    """
    y_true = np.asarray(y_true, dtype=float)
    taus   = sorted(quantile_preds.keys())
    T      = len(y_true)

    crps_sum = np.zeros(T)
    for i, tau in enumerate(taus):
        q       = np.asarray(quantile_preds[tau], dtype=float)
        resid   = y_true - q
        pinball = np.where(resid >= 0, tau * resid, (tau - 1) * resid)

        # Trapezoidal integration weight
        if i == 0:
            w = (taus[1] - taus[0]) / 2
        elif i == len(taus) - 1:
            w = (taus[-1] - taus[-2]) / 2
        else:
            w = (taus[i+1] - taus[i-1]) / 2

        crps_sum += 2 * w * pinball

    return crps_sum


def crps_from_samples(
    samples: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Energy score / CRPS from Monte Carlo samples.

    CRPS = E[|X - y|] - 0.5 · E[|X - X'|]

    Parameters
    ----------
    samples : (n_samples, T) — forecast samples for each test point
    y       : (T,) — actual values

    Returns
    -------
    crps : (T,) — per-observation CRPS
    """
    n_samples, T = samples.shape
    crps_vals    = np.zeros(T)

    for t in range(T):
        s      = samples[:, t]
        term1  = np.abs(s - y[t]).mean()
        term2  = np.abs(s[:, None] - s[None, :]).mean() / 2
        crps_vals[t] = term1 - term2

    return crps_vals
```

### 4.3 CRPS Skill Score

```python
def crps_skill_score(
    crps_model: np.ndarray,
    crps_baseline: np.ndarray,
) -> float:
    """
    CRPS Skill Score relative to a baseline model.

    CRPSS = 1 - mean(CRPS_model) / mean(CRPS_baseline)

    Returns
    -------
    CRPSS ∈ (-inf, 1.0]
    0 = no skill, 1 = perfect, negative = worse than baseline
    """
    mean_model    = crps_model.mean()
    mean_baseline = crps_baseline.mean()
    if mean_baseline == 0:
        return 1.0 if mean_model == 0 else float("-inf")
    return float(1 - mean_model / mean_baseline)
```

---

## 5. Winkler Score and Interval Score

### 5.1 Winkler Score

The **Winkler score** evaluates a single prediction interval `[L, U]` at level `1-α`:

```
W(L, U, y; α) =
  (U - L)                                         if L ≤ y ≤ U
  (U - L) + (2/α)(L - y)                          if y < L   (below interval)
  (U - L) + (2/α)(y - U)                          if y > U   (above interval)

Width term: U - L                  (always paid — sharpness cost)
Penalty:    (2/α) × distance       (paid when missed — calibration cost)

Lower Winkler score = better.
```

```python
def winkler_score(
    lower: np.ndarray,
    upper: np.ndarray,
    y_true: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Winkler Score for prediction interval evaluation.

    Parameters
    ----------
    lower  : lower bound of PI (T,)
    upper  : upper bound of PI (T,)
    y_true : actual values (T,)
    alpha  : miscoverage rate (0.10 for 90% PI)

    Returns
    -------
    scores : per-observation Winkler score (T,) — lower = better
    """
    lower  = np.asarray(lower,  dtype=float)
    upper  = np.asarray(upper,  dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    width   = upper - lower
    penalty = np.where(
        y_true < lower, (2 / alpha) * (lower - y_true),
        np.where(y_true > upper, (2 / alpha) * (y_true - upper), 0.0)
    )
    return width + penalty


def mean_winkler_score(lower, upper, y_true, alpha) -> float:
    """Mean Winkler score across all test observations."""
    return float(winkler_score(lower, upper, y_true, alpha).mean())
```

---

## 6. Pinball Loss (Quantile Loss)

### 6.1 Formula

```
L_τ(y, q̂) = τ · max(y - q̂, 0) + (1-τ) · max(q̂ - y, 0)

           = τ(y - q̂)     if y ≥ q̂
             (τ-1)(y-q̂)   if y < q̂

Mean Pinball Loss (MPL) = (1/T) Σₜ L_τ(yₜ, q̂ₜ)
```

### 6.2 Implementation

```python
def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tau: float,
) -> np.ndarray:
    """
    Per-observation quantile (pinball) loss.

    Parameters
    ----------
    y_true : actual values
    y_pred : quantile forecast (should be τ-th quantile)
    tau    : quantile level in (0, 1)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid  = y_true - y_pred
    return np.where(resid >= 0, tau * resid, (tau - 1) * resid)


def mean_pinball_all_quantiles(
    y_true: np.ndarray,
    quantile_preds: dict,
) -> pd.Series:
    """
    Compute mean pinball loss for all quantile levels.

    Returns
    -------
    pd.Series indexed by quantile level
    """
    return pd.Series({
        tau: float(pinball_loss(y_true, preds, tau).mean())
        for tau, preds in sorted(quantile_preds.items())
    })
```

---

## 7. Sharpness

### 7.1 Sharpness Metrics

```python
def sharpness_metrics(
    quantile_preds: dict,
    levels: list = None,
) -> pd.DataFrame:
    """
    Compute sharpness (interval width) for each PI level.

    Lower width = sharper = more informative (if also calibrated).

    Parameters
    ----------
    quantile_preds : dict {tau: predictions_array}
    levels         : PI levels to evaluate (e.g., [0.80, 0.90])
    """
    if levels is None:
        levels = [0.50, 0.80, 0.90, 0.95]

    rows = []
    for level in levels:
        alpha  = 1 - level
        tau_lo = alpha / 2
        tau_hi = 1 - alpha / 2

        if tau_lo not in quantile_preds or tau_hi not in quantile_preds:
            continue

        lo = np.asarray(quantile_preds[tau_lo], dtype=float)
        hi = np.asarray(quantile_preds[tau_hi], dtype=float)
        w  = hi - lo

        rows.append({
            "PI Level":       f"{int(level*100)}%",
            "Mean Width":     round(w.mean(), 4),
            "Median Width":   round(float(np.median(w)), 4),
            "Std Width":      round(w.std(), 4),
            "Min Width":      round(w.min(), 4),
            "Max Width":      round(w.max(), 4),
        })

    return pd.DataFrame(rows)
```

---

## 8. Full Probabilistic Evaluation Pipeline

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

class ProbabilisticEvaluator:
    """
    Comprehensive evaluation pipeline for probabilistic forecasts.

    Evaluates: calibration, sharpness, CRPS, Winkler score, pinball loss.
    Generates reliability diagram and coverage breakdown.
    """

    def __init__(self, y_true: np.ndarray, quantile_preds: dict,
                 mu: np.ndarray = None, sigma: np.ndarray = None):
        """
        Parameters
        ----------
        y_true         : actual test values (T,)
        quantile_preds : {tau: predictions_array} for all quantile levels
        mu             : predicted means (for Gaussian CRPS; optional)
        sigma          : predicted std devs (for Gaussian CRPS; optional)
        """
        self.y_true  = np.asarray(y_true, dtype=float)
        self.qpreds  = {float(k): np.asarray(v, dtype=float)
                        for k, v in quantile_preds.items()}
        self.mu      = mu
        self.sigma   = sigma

    def run(self, alpha_levels: list = None) -> dict:
        """
        Run full evaluation and return results dictionary.
        """
        if alpha_levels is None:
            alpha_levels = [0.80, 0.90, 0.95]

        results = {}

        # 1. Coverage at each PI level
        cov_rows = []
        for level in alpha_levels:
            alpha  = 1 - level
            tau_lo = round(alpha / 2, 3)
            tau_hi = round(1 - alpha / 2, 3)
            if tau_lo in self.qpreds and tau_hi in self.qpreds:
                lo  = self.qpreds[tau_lo]
                hi  = self.qpreds[tau_hi]
                cov = coverage_at_level(self.y_true, lo, hi, nominal=level)
                cov_rows.append({
                    "PI Level": f"{int(level*100)}%",
                    "Nominal":  level,
                    "Empirical": cov["empirical_coverage"],
                    "Error":     cov["coverage_error"],
                    "Width (mean)": round(float((hi - lo).mean()), 4),
                    "Winkler":   round(mean_winkler_score(lo, hi, self.y_true, alpha), 4),
                })
        results["coverage"] = pd.DataFrame(cov_rows)

        # 2. Pinball loss per quantile
        results["pinball"] = mean_pinball_all_quantiles(self.y_true, self.qpreds)

        # 3. CRPS
        crps_vals = crps_from_quantiles(self.y_true, self.qpreds)
        results["crps_mean"] = float(crps_vals.mean())
        results["crps_std"]  = float(crps_vals.std())

        if self.mu is not None and self.sigma is not None:
            crps_gauss = crps_gaussian(self.mu, self.sigma, self.y_true)
            results["crps_gaussian_mean"] = float(crps_gauss.mean())

        # 4. Mean Absolute Calibration Error
        taus     = sorted(self.qpreds.keys())
        emp_covs = [(self.y_true <= self.qpreds[tau]).mean() for tau in taus]
        mace     = np.abs(np.array(emp_covs) - np.array(taus)).mean()
        results["mace"] = float(mace)

        return results

    def print_report(self, alpha_levels: list = None):
        """Print a formatted evaluation report."""
        res = self.run(alpha_levels)

        print("\n" + "="*60)
        print("PROBABILISTIC FORECAST EVALUATION REPORT")
        print("="*60)

        print("\n📊 Coverage & Interval Width:")
        print(res["coverage"].to_string(index=False))

        print(f"\n📏 CRPS: {res['crps_mean']:.4f} ± {res['crps_std']:.4f}")
        print(f"📏 Mean Absolute Calibration Error (MACE): {res['mace']:.4f}")

        print("\n📌 Pinball Loss by Quantile:")
        print(res["pinball"].round(4).to_string())

        # Calibration verdict
        print("\n🔍 Calibration Verdict:")
        for _, row in res["coverage"].iterrows():
            err = row["Error"]
            if abs(err) < 0.02:
                status = "✅ Well-calibrated"
            elif err < 0:
                status = "⚠️ Overconfident (intervals too narrow)"
            else:
                status = "⚠️ Underconfident (intervals too wide)"
            print(f"  {row['PI Level']}: {status} (error={err:+.3f})")


# ─── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import lightgbm as lgb

    np.random.seed(42)
    N      = 400
    t      = np.arange(N)
    series = 50 + 0.05*t + 8*np.sin(2*np.pi*t/52) + np.random.normal(0, 3, N)

    lags = [1, 2, 7, 14]
    X, y = [], []
    for i in range(max(lags), N):
        X.append([series[i-l] for l in lags])
        y.append(series[i])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    qpreds = {}
    for tau in QUANTILES:
        m = lgb.LGBMRegressor(n_estimators=200, objective="quantile",
                               alpha=tau, verbose=-1)
        m.fit(X_tr, y_tr)
        qpreds[tau] = m.predict(X_te)

    evaluator = ProbabilisticEvaluator(y_te, qpreds)
    evaluator.print_report(alpha_levels=[0.80, 0.90])
    reliability_diagram(y_te, qpreds, title="LightGBM Quantile Regression Calibration")
```

---

*← [04 — Model Comparison & Statistical Tests](./04_model_comparison_and_statistical_tests.md) | [Module README](./README.md)*
