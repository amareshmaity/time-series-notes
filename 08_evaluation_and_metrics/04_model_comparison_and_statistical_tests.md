# 04 — Model Comparison and Statistical Tests

> **Module**: 08 Evaluation & Metrics | **File**: 4 of 5
>
> "Model A has lower RMSE than Model B" is not a conclusion — it's a hypothesis. Statistical tests tell you whether the observed difference is genuine signal or just noise from the evaluation window. This note covers the Diebold-Mariano test, Model Confidence Set, and paired comparison methods.

---

## Table of Contents

1. [Why Statistical Tests Are Necessary](#1-why-statistical-tests-are-necessary)
2. [Diebold-Mariano Test](#2-diebold-mariano-test)
3. [Modified Diebold-Mariano Test](#3-modified-diebold-mariano-test)
4. [Model Confidence Set (MCS)](#4-model-confidence-set-mcs)
5. [Paired t-Test for Forecasting](#5-paired-t-test-for-forecasting)
6. [Giacomini-White Test (Conditional Predictive Ability)](#6-giacomini-white-test-conditional-predictive-ability)
7. [Practical Guidelines](#7-practical-guidelines)
8. [Implementation](#8-implementation)

---

## 1. Why Statistical Tests Are Necessary

### 1.1 The Lucky Test Window Problem

```
Scenario: Model A vs. Model B, evaluated on 20 test windows.
Result: Model A has RMSE = 5.2, Model B has RMSE = 5.5.

Is this a real difference?

With 20 observations, the standard error of the RMSE difference is substantial.
If the true RMSE difference is 0 (models are equally good),
you'd see a 0.3-unit gap ~30% of the time just by chance.

→ Without a statistical test, you cannot know if the improvement is real.
```

### 1.2 Multiple Comparison Problem

Testing many models against a benchmark inflates false positive rate:

```
Testing 20 models at α=0.05 each:
  P(at least one false positive) = 1 - (1-0.05)^20 ≈ 64%

→ With 20 models, you'd expect ~1 model to appear significantly better
   than the baseline purely by chance, even if all models are equally good.

Solutions:
  1. Bonferroni correction: use α/n for each test
  2. Model Confidence Set (MCS): tests all models jointly
```

---

## 2. Diebold-Mariano Test

### 2.1 Setup

The DM test (Diebold & Mariano, 1995) formally tests whether two forecasters have equal predictive accuracy:

```
H₀: E[d̄] = 0   (Models A and B have equal expected loss)
H₁: E[d̄] ≠ 0   (Model A has lower expected loss than B)

Where:
  dₜ = L(eₜᴬ) - L(eₜᴮ)   — loss differential at time t
  eₜᴬ = yₜ - ŷₜᴬ           — forecast error of Model A
  eₜᴮ = yₜ - ŷₜᴮ           — forecast error of Model B
  L(·) = loss function (e.g., squared error, absolute error)

  d̄ = (1/T) Σₜ dₜ    — mean loss differential

DM statistic: DM = d̄ / √(V̂/T)

Where V̂ is a HAC (heteroskedasticity-autocorrelation consistent) variance estimator.
```

### 2.2 Loss Functions for DM

| Loss function | Formula         | Tests equality of |
|---------------|-----------------|-------------------|
| MSE loss      | L(e) = e²       | RMSE              |
| MAE loss      | L(e) = \|e\|    | MAE               |
| Quantile loss | L(e) = pinball  | Quantile accuracy |
| Lin-Lin loss  | Asymmetric      | Asymmetric cost   |

### 2.3 HAC Variance Estimation

For h-step-ahead forecasts, the loss differentials are autocorrelated up to lag h-1. The DM test uses **Newey-West** or **Bartlett kernel** HAC estimation:

```
V̂_NW = γ₀ + 2 Σⱼ₌₁^{h-1} (1 - j/h) γⱼ

Where γⱼ = (1/T) Σₜ (dₜ - d̄)(dₜ₋ⱼ - d̄)   — j-th autocovariance of dₜ
```

### 2.4 Implementation

```python
import numpy as np
from scipy import stats

def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
    loss: str = "mse",
    alternative: str = "two-sided",
) -> dict:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Parameters
    ----------
    errors_a    : forecast errors of model A (y_true - y_pred_A), shape (T,)
    errors_b    : forecast errors of model B (y_true - y_pred_B), shape (T,)
    h           : forecast horizon (controls HAC lag truncation)
    loss        : loss function — 'mse', 'mae', 'mse-mae' (tests RMSE vs MAE tradeoff)
    alternative : 'two-sided' | 'less' (A is less accurate) | 'greater' (A is more accurate)

    Returns
    -------
    dict with DM statistic, p-value, and interpretation
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    T        = len(errors_a)

    # Compute loss differentials
    if loss == "mse":
        d = errors_a**2 - errors_b**2
    elif loss == "mae":
        d = np.abs(errors_a) - np.abs(errors_b)
    else:
        raise ValueError(f"Unknown loss: {loss}. Use 'mse' or 'mae'.")

    d_bar = d.mean()

    # HAC variance (Newey-West with h-1 lags)
    gamma0 = ((d - d_bar)**2).mean()
    hac_var = gamma0

    for lag in range(1, h):
        weight = 1 - lag / h   # Bartlett kernel
        gamma  = ((d[lag:] - d_bar) * (d[:-lag] - d_bar)).mean()
        hac_var += 2 * weight * gamma

    se    = np.sqrt(hac_var / T)
    dm    = d_bar / (se + 1e-12)

    # p-value
    if alternative == "two-sided":
        p_val = 2 * (1 - stats.norm.cdf(abs(dm)))
    elif alternative == "less":
        p_val = stats.norm.cdf(dm)        # P(DM ≤ dm)
    else:
        p_val = 1 - stats.norm.cdf(dm)   # P(DM ≥ dm)

    return {
        "dm_statistic":   float(dm),
        "p_value":        float(p_val),
        "mean_loss_diff": float(d_bar),   # positive = A is worse
        "h":              h,
        "loss":           loss,
        "n_obs":          T,
        "reject_H0_5pct": bool(p_val < 0.05),
        "interpretation": (
            f"p={p_val:.4f} — "
            + ("Reject H₀: Models have significantly different accuracy"
               if p_val < 0.05
               else "Fail to reject H₀: No significant accuracy difference")
        ),
    }
```

---

## 3. Modified Diebold-Mariano Test

### 3.1 Why Modify?

The original DM test is designed for large samples. Harvey, Leybourne & Newbold (1997) proposed a **modified DM (MDM)** test that uses a t-distribution instead of normal, providing better small-sample properties:

```
MDM statistic = DM · √[(T + 1 - 2h + h(h-1)/T) / T]

Under H₀, MDM ~ t(T-1)   (vs. DM ~ N(0,1))

→ MDM has correct size in small samples (T < 30)
→ Practically: use MDM when T < 100
```

```python
def modified_dm_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
    loss: str = "mae",
) -> dict:
    """
    Modified Diebold-Mariano test (Harvey, Leybourne & Newbold, 1997).

    Better small-sample properties than original DM.
    Uses t(T-1) distribution instead of standard normal.
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    T        = len(errors_a)

    if loss == "mse":
        d = errors_a**2 - errors_b**2
    elif loss == "mae":
        d = np.abs(errors_a) - np.abs(errors_b)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    d_bar   = d.mean()
    gamma0  = ((d - d_bar)**2).mean()
    hac_var = gamma0
    for lag in range(1, h):
        w      = 1 - lag / h
        gamma  = ((d[lag:] - d_bar) * (d[:-lag] - d_bar)).mean()
        hac_var += 2 * w * gamma

    # MDM correction factor
    correction = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    se  = np.sqrt(hac_var / T)
    dm  = d_bar / (se + 1e-12)
    mdm = dm * correction

    p_val = 2 * (1 - stats.t.cdf(abs(mdm), df=T-1))

    return {
        "mdm_statistic":  float(mdm),
        "dm_statistic":   float(dm),
        "p_value":        float(p_val),
        "mean_loss_diff": float(d_bar),
        "reject_H0_5pct": bool(p_val < 0.05),
    }
```

---

## 4. Model Confidence Set (MCS)

### 4.1 Concept

The **Model Confidence Set** (Hansen, Lunde & Nason, 2011) tests a **set** of models simultaneously. It returns the subset of models that cannot be statistically distinguished from the best model at a given confidence level:

```
Given M models, MCS procedure:

  1. Start with all M models in the set S
  2. Test H₀: all models in S are equally accurate
  3. If rejected → identify and REMOVE the worst model from S
  4. Repeat steps 2–3 until H₀ is not rejected
  5. S* = surviving models = Model Confidence Set

Guarantee: P(best model ∈ S*) ≥ 1 - α

Interpretation:
  If S* = {A, B}: A and B are statistically indistinguishable from the best
  If S* = {A}:    A is significantly better than all others
```

### 4.2 Implementation (Range Statistic)

```python
import numpy as np
from scipy import stats as scipy_stats

def model_confidence_set(
    losses: np.ndarray,
    alpha: float = 0.10,
    n_bootstrap: int = 1000,
    model_names: list = None,
) -> dict:
    """
    Model Confidence Set using the Range (TR) statistic.

    Parameters
    ----------
    losses      : (T, M) matrix — T time periods, M models
                  Each cell = loss value at time t for model m
    alpha       : significance level (default 0.10 → 90% MCS)
    n_bootstrap : bootstrap replications for critical value estimation
    model_names : list of model name strings

    Returns
    -------
    dict with MCS members and elimination order
    """
    losses = np.asarray(losses, dtype=float)
    T, M   = losses.shape

    if model_names is None:
        model_names = [f"Model_{i}" for i in range(M)]

    # Relative losses: dᵢⱼ = lossᵢ - lossⱼ
    # TR statistic: max over i of |d̄ᵢ| / σ̂ᵢ

    surviving = list(range(M))
    elim_order = []

    while len(surviving) > 1:
        idx         = np.array(surviving)
        sub_losses  = losses[:, idx]
        n_sub       = len(idx)

        # Mean relative performance vs. mean of group
        d_bar = sub_losses.mean(axis=1, keepdims=True)   # (T, 1) mean at each t
        rel   = sub_losses - d_bar                        # (T, n_sub) relative losses
        mu    = rel.mean(axis=0)                          # (n_sub,) mean relative loss

        # Variance via block bootstrap
        boot_means = np.zeros((n_bootstrap, n_sub))
        for b in range(n_bootstrap):
            indices       = np.random.choice(T, size=T, replace=True)
            boot_rel      = rel[indices]
            boot_means[b] = boot_rel.mean(axis=0)

        std_boot = boot_means.std(axis=0) + 1e-12   # (n_sub,)

        # TR statistic = max |t_i|
        t_stats = np.abs(mu / std_boot)
        TR      = t_stats.max()

        # Bootstrap critical value
        boot_TR = np.abs(boot_means / std_boot).max(axis=1)
        crit    = np.quantile(boot_TR, 1 - alpha)

        if TR <= crit:
            break   # Cannot reject H₀ — remaining models form MCS

        # Eliminate model with highest t_stat (worst relative performance)
        worst_local = t_stats.argmax()
        worst_global = idx[worst_local]
        surviving.remove(worst_global)
        elim_order.append(model_names[worst_global])

    mcs_members = [model_names[i] for i in surviving]

    return {
        "mcs_members":       mcs_members,
        "eliminated_order":  elim_order,
        "alpha":             alpha,
        "n_bootstrap":       n_bootstrap,
        "n_models_total":    M,
        "n_models_in_mcs":   len(mcs_members),
    }
```

---

## 5. Paired t-Test for Forecasting

### 5.1 When to Use

The paired t-test is simpler than DM but appropriate when:
- Forecast errors are approximately normally distributed
- Test windows are independent (non-overlapping)
- Sample size is reasonable (T > 30)

```
H₀: μ_A = μ_B  (same expected error)
H₁: μ_A ≠ μ_B

d = |eₜᴬ| - |eₜᴮ|   (paired absolute errors)
t = d̄ / (s_d / √T) ~ t(T-1)   under H₀
```

```python
from scipy import stats

def paired_ttest(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    loss: str = "mae",
    alpha: float = 0.05,
) -> dict:
    """
    Paired t-test for comparing two forecast models.

    Parameters
    ----------
    errors_a : model A forecast errors
    errors_b : model B forecast errors
    loss     : 'mae' uses |errors|, 'mse' uses errors²
    alpha    : significance level

    Returns
    -------
    dict with t-statistic, p-value, and conclusion
    """
    if loss == "mae":
        loss_a = np.abs(errors_a)
        loss_b = np.abs(errors_b)
    elif loss == "mse":
        loss_a = errors_a**2
        loss_b = errors_b**2
    else:
        raise ValueError(f"loss must be 'mae' or 'mse'")

    d      = loss_a - loss_b
    t_stat, p_val = scipy_stats.ttest_1samp(d, popmean=0)

    return {
        "t_statistic":    float(t_stat),
        "p_value":        float(p_val),
        "mean_diff":      float(d.mean()),   # positive = A is worse
        "ci_95":          scipy_stats.t.interval(0.95, len(d)-1,
                                                 loc=d.mean(),
                                                 scale=scipy_stats.sem(d)),
        "reject_H0":      bool(p_val < alpha),
        "n_obs":          len(d),
    }
```

---

## 6. Giacomini-White Test (Conditional Predictive Ability)

### 6.1 Motivation

The DM test evaluates **unconditional** predictive ability — does model A beat model B on average? The **Giacomini-White (GW) test** evaluates **conditional** predictive ability — does model A beat model B given a specific regime (volatility, trend, season)?

```
H₀: E[dₜ | Ωₜ₋₁] = 0   for all t

Where Ωₜ₋₁ = conditioning information (e.g., VIX level, recent volatility, season)

→ GW tests if differences in accuracy are predictable from past information
→ A well-specified model should have unpredictable loss differentials
```

```python
import numpy as np
from scipy import stats

def giacomini_white_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    instruments: np.ndarray,
    loss: str = "mae",
) -> dict:
    """
    Giacomini-White test for conditional predictive ability.

    Tests whether loss differentials are predictable from instruments.

    Parameters
    ----------
    errors_a    : model A forecast errors (T,)
    errors_b    : model B forecast errors (T,)
    instruments : (T, k) matrix of conditioning variables
                  (e.g., lagged loss diff, volatility measure, seasonal dummy)
                  Last row of instruments is for the first test — align carefully.
    loss        : 'mae' or 'mse'

    Returns
    -------
    dict with test statistic, p-value, and degrees of freedom
    """
    if loss == "mae":
        d = np.abs(errors_a) - np.abs(errors_b)
    else:
        d = errors_a**2 - errors_b**2

    T = len(d)
    Z = np.asarray(instruments, dtype=float)  # (T, k)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    # Add intercept
    Z_aug = np.column_stack([np.ones(T), Z])
    k     = Z_aug.shape[1]

    # OLS: regress d on Z_aug
    beta  = np.linalg.lstsq(Z_aug, d, rcond=None)[0]
    d_hat = Z_aug @ beta
    resid = d - d_hat

    # GW statistic ~ chi²(k)
    # Use heteroskedasticity-robust variant
    S = Z_aug.T @ np.diag(resid**2) @ Z_aug / T
    G = (Z_aug * d[:, None]).mean(axis=0)
    gw_stat = T * (G @ np.linalg.pinv(S) @ G)

    p_val = 1 - stats.chi2.cdf(gw_stat, df=k)

    return {
        "gw_statistic": float(gw_stat),
        "p_value":      float(p_val),
        "df":           k,
        "n_instruments": k - 1,
        "reject_H0_5pct": bool(p_val < 0.05),
        "interpretation": (
            "Conditional accuracy differs — loss differential is predictable"
            if p_val < 0.05 else
            "No evidence of conditional predictive ability difference"
        ),
    }
```

---

## 7. Practical Guidelines

### 7.1 Which Test to Use

```
Comparing 2 models:
  Small sample (T < 50):   → Modified DM test
  Large sample (T ≥ 50):   → DM test (or paired t-test if errors ~ normal)

Comparing many models (M > 2):
  → Model Confidence Set (MCS)
  → Avoid multiple DM tests (inflates Type I error)

Testing regime-specific performance:
  → Giacomini-White test

Hyperparameter selection:
  → NEVER use the DM test to select hyperparameters — use it only to compare final models
  → Use time-series CV for hyperparameter selection
```

### 7.2 Reporting Standards

```
Always report alongside DM/MDM results:
  1. Sample size T (affects power)
  2. Horizon h used for HAC truncation
  3. Loss function (MAE/MSE)
  4. Alternative hypothesis direction
  5. Effect size (mean loss differential, % improvement)
  6. Confidence intervals on the metric difference

P-value alone is insufficient — effect size matters.
A statistically significant difference of 0.001 MAE may be practically irrelevant.
```

---

## 8. Implementation

```python
import numpy as np
import pandas as pd
from scipy import stats

def full_model_comparison_report(
    y_true: np.ndarray,
    forecasts: dict,
    baseline_name: str,
    h: int = 1,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compare all models against a baseline using DM test.

    Parameters
    ----------
    y_true        : actual values (T,)
    forecasts     : dict of {model_name: predictions_array}
    baseline_name : name of the baseline model in forecasts dict
    h             : forecast horizon (for DM HAC correction)
    alpha         : significance level

    Returns
    -------
    DataFrame with DM statistics and accuracy metrics per model
    """
    y_true   = np.asarray(y_true, dtype=float)
    baseline = np.asarray(forecasts[baseline_name], dtype=float)
    e_base   = y_true - baseline

    rows = []
    for name, preds in forecasts.items():
        preds  = np.asarray(preds, dtype=float)
        e_mod  = y_true - preds
        mae_m  = np.abs(e_mod).mean()
        rmse_m = np.sqrt((e_mod**2).mean())

        if name == baseline_name:
            row = {
                "Model":    name,
                "MAE":      round(mae_m, 4),
                "RMSE":     round(rmse_m, 4),
                "DM_stat":  "—",
                "p_value":  "—",
                "Sig.":     "baseline",
                "MAE_diff": "—",
            }
        else:
            dm_res = diebold_mariano_test(e_mod, e_base, h=h, loss="mae")
            row = {
                "Model":    name,
                "MAE":      round(mae_m, 4),
                "RMSE":     round(rmse_m, 4),
                "DM_stat":  round(dm_res["dm_statistic"], 3),
                "p_value":  round(dm_res["p_value"], 4),
                "Sig.":     "✅" if dm_res["p_value"] < alpha else "ns",
                "MAE_diff": round(dm_res["mean_loss_diff"], 4),
            }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    print(f"\nModel Comparison Report (baseline: {baseline_name}, α={alpha})")
    print(f"DM_stat < 0 → model has lower MAE than baseline (positive = worse)\n")
    return df


# ─── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(7)
    T      = 100
    y_true = np.cumsum(np.random.randn(T)) + 50

    naive     = np.roll(y_true, 1); naive[0] = y_true[0]
    better    = y_true + np.random.normal(0, 0.8, T)
    similar   = y_true + np.random.normal(0, 2.0, T)
    worse     = y_true + np.random.normal(1, 3.0, T)

    report = full_model_comparison_report(
        y_true,
        forecasts={"Naïve": naive, "Better": better,
                   "Similar": similar, "Worse": worse},
        baseline_name="Naïve",
        h=1,
    )
    print(report.to_string())

    # MCS example
    losses = np.column_stack([
        np.abs(y_true - naive),
        np.abs(y_true - better),
        np.abs(y_true - similar),
        np.abs(y_true - worse),
    ])
    mcs = model_confidence_set(losses, alpha=0.10,
                               model_names=["Naïve","Better","Similar","Worse"])
    print(f"\nModel Confidence Set (90%): {mcs['mcs_members']}")
    print(f"Eliminated order: {mcs['eliminated_order']}")
```

---

*← [03 — Backtesting Design](./03_backtesting_design.md) | [Module README](./README.md) | Next: [05 — Calibration for Probabilistic Models](./05_calibration_for_probabilistic_models.md) →*
