# 02 — Granger Causality

> **Module**: 12 Multivariate & Advanced Topics | **File**: 2 of 6
>
> Granger causality provides a statistical test for predictive precedence: does knowing the past of X improve our forecast of Y beyond what Y's own past already tells us? It is the workhorse of causal screening in economics, neuroscience, and time series analysis.

---

## Table of Contents

1. [Definition and Intuition](#1-definition-and-intuition)
2. [Bivariate Granger Test](#2-bivariate-granger-test)
3. [VAR-Based Multivariate Testing](#3-var-based-multivariate-testing)
4. [PCMCI — Causal Discovery for Time Series](#4-pcmci--causal-discovery-for-time-series)
5. [Nonlinear Extensions](#5-nonlinear-extensions)
6. [Pitfalls and Limitations](#6-pitfalls-and-limitations)
7. [Implementation](#7-implementation)

---

## 1. Definition and Intuition

### 1.1 Granger's Definition

```
X Granger-causes Y at lag k if:

  Var(Y_{t+1} | Y_{past}, X_{past}) < Var(Y_{t+1} | Y_{past})

  i.e., adding past values of X reduces the prediction error for Y.

Operationally (F-test):
  Restricted model (Y only):
    Y_t = α + Σᵢ aᵢ·Y_{t-i} + ε_t

  Unrestricted model (Y + X):
    Y_t = α + Σᵢ aᵢ·Y_{t-i} + Σᵢ bᵢ·X_{t-i} + ε_t

  H₀: b₁ = b₂ = ... = bₚ = 0  (X does NOT Granger-cause Y)
  F-statistic:
    F = [(RSS_R - RSS_U)/p] / [RSS_U/(n - 2p - 1)]
    
  If F > F_{crit} or p-value < α → reject H₀ → X Granger-causes Y.
```

### 1.2 What It Does and Does Not Mean

```
DOES MEAN:
  ✓ Past X provides statistically significant predictive information for Y
  ✓ There is a temporal precedence of X over Y
  ✓ X and Y are statistically linked at specific lags

DOES NOT MEAN:
  ✗ X causes Y in the structural/interventional sense
  ✗ Manipulating X will change Y (could be a spurious association)
  ✗ The relationship is linear (standard test is linear only)
  ✗ Causality is unique (multiple causal paths possible)

Example of false positive:
  "Ice cream sales Granger-cause sunburns" — both driven by summer.
  Controlling for temperature eliminates the effect.
```

---

## 2. Bivariate Granger Test

### 2.1 Implementation with statsmodels

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

def granger_test_pair(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 5,
    alpha: float = 0.05,
) -> dict:
    """
    Bivariate Granger causality test: does X Granger-cause Y?

    Parameters
    ----------
    x, y    : 1D arrays (x = potential cause, y = effect)
              Both must be STATIONARY (difference if needed).
    max_lag : test for lags 1 through max_lag
    alpha   : significance level

    Returns
    -------
    dict with p-values per lag, best lag, and causality verdict
    """
    # Combine into (n, 2) array — statsmodels convention: [y, x]
    data      = np.column_stack([y, x])
    test_lags = list(range(1, max_lag + 1))

    results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

    rows = []
    for lag in test_lags:
        # Extract F-test result
        f_stat = results[lag][0]["ssr_ftest"][0]
        p_val  = results[lag][0]["ssr_ftest"][1]
        rows.append({
            "lag":    lag,
            "F_stat": round(f_stat, 4),
            "p_val":  round(p_val, 5),
            "significant": p_val < alpha,
        })

    df    = pd.DataFrame(rows)
    sig   = df[df["significant"]]
    causal = len(sig) > 0

    return {
        "x_granger_causes_y": causal,
        "best_lag": int(sig["lag"].iloc[0]) if causal else None,
        "min_pvalue": float(df["p_val"].min()),
        "table": df,
    }


def check_stationarity(series: np.ndarray, alpha: float = 0.05) -> tuple:
    """
    ADF test for stationarity. Returns (is_stationary, p_value).
    Difference the series if not stationary before Granger testing.
    """
    result = adfuller(series, autolag="AIC")
    is_stationary = result[1] < alpha
    return is_stationary, round(result[1], 5)
```

### 2.2 Granger Causality Matrix

```python
def granger_causality_matrix(
    df: pd.DataFrame,
    max_lag: int = 5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute pairwise Granger causality for all pairs in df.
    
    Returns a D×D DataFrame where entry (i,j) = min p-value of
    "column i Granger-causes column j".
    Significant entries (p < alpha) indicate potential causal links.

    Parameters
    ----------
    df      : DataFrame of D STATIONARY time series
    max_lag : maximum lags to test
    alpha   : significance threshold

    Returns
    -------
    pval_matrix : D×D DataFrame of minimum p-values
    """
    cols   = df.columns.tolist()
    D      = len(cols)
    pmat   = pd.DataFrame(np.ones((D, D)), index=cols, columns=cols)
    sig_mat = pd.DataFrame(False, index=cols, columns=cols)

    for cause in cols:
        for effect in cols:
            if cause == effect:
                pmat.loc[cause, effect]   = np.nan
                sig_mat.loc[cause, effect] = False
                continue
            try:
                res = granger_test_pair(
                    df[cause].values, df[effect].values, max_lag=max_lag, alpha=alpha
                )
                pmat.loc[cause, effect]    = res["min_pvalue"]
                sig_mat.loc[cause, effect] = res["x_granger_causes_y"]
            except Exception as e:
                pmat.loc[cause, effect] = 1.0

    return pmat, sig_mat
```

---

## 3. VAR-Based Multivariate Testing

### 3.1 Why VAR Extension?

```
Problem with bivariate Granger:
  Testing X → Y ignores all other variables Z.
  If Z → X and Z → Y, then X appears to Granger-cause Y
  even when the true path is Z → {X, Y}.

Solution: Multivariate Granger test via VAR.
  Fit VAR(p) on all D series simultaneously.
  Test: are all coefficients in row j, columns for X equal to 0?

In statsmodels:
  VARResults.test_causality(caused, causing, kind='f')
  Tests whether 'causing' Granger-causes 'caused' in the VAR system.
```

```python
def var_granger_test(df: pd.DataFrame, lag_order: int = 3) -> pd.DataFrame:
    """
    Multivariate Granger causality test via fitted VAR model.

    Tests all pairwise X → Y relationships while conditioning
    on all other variables in the system.

    Parameters
    ----------
    df        : DataFrame of stationary series
    lag_order : VAR lag order (use var_lag_selection to choose)

    Returns
    -------
    DataFrame with cause, effect, F_stat, p_val, significant
    """
    from statsmodels.tsa.vector_ar.var_model import VAR

    model   = VAR(df)
    results = model.fit(lag_order, ic=None)

    cols = df.columns.tolist()
    rows = []
    for cause in cols:
        for effect in cols:
            if cause == effect:
                continue
            try:
                test   = results.test_causality(effect, [cause], kind="f")
                rows.append({
                    "cause":       cause,
                    "effect":      effect,
                    "F_stat":      round(test.test_statistic, 4),
                    "p_val":       round(test.pvalue, 5),
                    "significant": test.pvalue < 0.05,
                })
            except Exception as e:
                rows.append({"cause": cause, "effect": effect,
                             "F_stat": np.nan, "p_val": np.nan, "significant": False})

    result_df = pd.DataFrame(rows).sort_values("p_val")
    print("Significant Granger links (p < 0.05):")
    print(result_df[result_df["significant"]].to_string(index=False))
    return result_df
```

---

## 4. PCMCI — Causal Discovery for Time Series

### 4.1 Overview

```
PCMCI (Peter-Clark Momentary Conditional Independence, Runge et al. 2019):
  - Addresses the confounding problem in Granger causality
  - Uses conditional independence tests to separate direct from indirect links
  - Controls false discovery rate (FDR) for large D

Algorithm:
  1. PC step: identify skeleton of causal graph using conditional independence
  2. MCI step: test momentary conditional independence
       X_t→Y_{t+k} is a link if X_{t-k} ⊥̸ Y_t | {Y_{past}, parents(X)}
  3. Result: time-lagged causal graph with confidence scores

pip install tigramite
```

```python
def run_pcmci(
    df: pd.DataFrame,
    tau_max: int = 5,
    pc_alpha: float = 0.05,
    mci_alpha: float = 0.05,
) -> dict:
    """
    PCMCI causal discovery on multivariate time series.

    Parameters
    ----------
    df        : DataFrame of stationary, standardized time series
    tau_max   : maximum time lag to test
    pc_alpha  : significance level for PC skeleton step
    mci_alpha : significance level for MCI test

    Returns
    -------
    dict with val_matrix (effect strengths), p_matrix, graph
    """
    try:
        import tigramite.data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr

        data    = df.values.astype(float)
        dataobj = pp.DataFrame(data, var_names=df.columns.tolist())

        parcorr = ParCorr(significance="analytic")
        pcmci   = PCMCI(dataframe=dataobj, cond_ind_test=parcorr, verbosity=0)

        results = pcmci.run_pcmci(
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            alpha_level=mci_alpha,
        )

        print("\nSignificant causal links (MCI test):")
        for j, col in enumerate(df.columns):
            for i, parent in enumerate(df.columns):
                for lag in range(1, tau_max + 1):
                    p = results["p_matrix"][i, j, lag]
                    v = results["val_matrix"][i, j, lag]
                    if p < mci_alpha:
                        print(f"  {parent}(t-{lag}) → {col}(t): "
                              f"coeff={v:.4f}, p={p:.5f}")

        return {
            "val_matrix": results["val_matrix"],
            "p_matrix":   results["p_matrix"],
            "graph":      results["graph"],
            "var_names":  df.columns.tolist(),
        }

    except ImportError:
        print("Install tigramite: pip install tigramite")
        return {}
```

---

## 5. Nonlinear Extensions

### 5.1 Transfer Entropy

```
Transfer Entropy (Schreiber, 2000) — information-theoretic Granger causality:

  TE(X→Y) = H(Y_{t+1} | Y_past) - H(Y_{t+1} | Y_past, X_past)

  Where H(·) = Shannon entropy.

  ✅ Model-free — detects nonlinear dependencies
  ✅ Equivalent to Granger causality for Gaussian data
  ✅ Captures higher-order statistical relationships
  ❌ Requires many samples for reliable entropy estimation
  ❌ Computationally intensive
```

```python
def transfer_entropy(x, y, lag=1, n_bins=10):
    """
    Estimate Transfer Entropy from X to Y using binning.
    TE(X→Y) = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})

    For production use: pip install pyinform (more accurate estimators).
    """
    def entropy(p):
        p = p[p > 0]
        return -float(np.sum(p * np.log2(p)))

    def joint_prob(*arrays, bins=n_bins):
        hist, _ = np.histogramdd(np.column_stack(arrays), bins=bins)
        return hist / hist.sum()

    n    = len(x) - lag
    xt   = x[lag:]
    xt_l = x[:-lag]
    yt   = y[lag:]
    yt_l = y[:-lag]

    # H(Y_t | Y_{t-1}) — without X
    p_yyt  = joint_prob(yt, yt_l)
    p_yt_m = p_yyt.sum(axis=1, keepdims=True)
    cond1  = entropy((p_yyt / (p_yt_m + 1e-12)).flatten() * p_yyt.flatten())

    # H(Y_t | Y_{t-1}, X_{t-1}) — with X
    p_xyyt = joint_prob(yt, yt_l, xt_l)
    cond2  = 0.0
    p_xyt  = p_xyyt.sum(axis=0, keepdims=True)
    for i in range(n_bins):
        slice_i = p_xyyt[:, :, i]
        norm    = p_xyt[:, :, i].sum()
        if norm > 0:
            cond2 += entropy((slice_i / (norm + 1e-12)).flatten() * slice_i.flatten())

    return max(0.0, cond1 - cond2)
```

---

## 6. Pitfalls and Limitations

```
PITFALL 1: Testing non-stationary series
  Granger test on I(1) series → spurious results.
  FIX: Difference to stationarity first. Use ADF test.

PITFALL 2: Omitted variable bias
  True causal graph: Z → X, Z → Y
  Bivariate test X → Y falsely rejects H₀.
  FIX: Use multivariate VAR-based Granger test.
  FIX: Use PCMCI which controls for indirect paths.

PITFALL 3: Instantaneous causality
  Granger causality tests LAGGED effects only.
  Same-time effects (shock at t affects Y at t) are invisible.
  FIX: Structural VAR (SVAR) for contemporaneous identification.

PITFALL 4: Nonstationarity of causal links
  The causal structure changes over time (regime shifts).
  Standard test assumes stationarity of coefficients.
  FIX: Rolling-window Granger test or regime-switching VAR.

PITFALL 5: Nonlinearity
  Standard F-test is linear. Nonlinear causal links will be missed.
  FIX: Transfer entropy or kernel-based conditional independence tests.

PITFALL 6: Multiple testing
  Testing D×D pairs → D² tests → many false positives at α=0.05.
  FIX: Bonferroni correction: α_adj = 0.05 / D²
       Or FDR control (Benjamini-Hochberg) as in PCMCI.
```

---

## 7. Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_granger_heatmap(pval_matrix: pd.DataFrame, alpha: float = 0.05) -> None:
    """
    Visualize Granger causality p-value matrix as heatmap.

    Rows = potential causes, Columns = effects.
    Dark cells = significant causality (low p-value).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # P-value heatmap
    ax = axes[0]
    data = pval_matrix.fillna(1.0).values
    im   = ax.imshow(data, vmin=0, vmax=0.1, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(pval_matrix.columns)))
    ax.set_xticklabels(pval_matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pval_matrix.index)))
    ax.set_yticklabels(pval_matrix.index)
    ax.set_title("Granger Causality P-Values\n(row → column)", fontsize=11)
    ax.set_xlabel("Effect"); ax.set_ylabel("Cause")
    plt.colorbar(im, ax=ax, shrink=0.8, label="p-value")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(pval_matrix.iloc[i,j]):
                ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                        fontsize=8, color="white" if data[i,j] < 0.03 else "black")

    # Significant links binary matrix
    ax = axes[1]
    sig  = (data < alpha).astype(float)
    np.fill_diagonal(sig, np.nan)
    im2  = ax.imshow(sig, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(pval_matrix.columns)))
    ax.set_xticklabels(pval_matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pval_matrix.index)))
    ax.set_yticklabels(pval_matrix.index)
    ax.set_title(f"Significant Links (p < {alpha})\n(row → column)", fontsize=11)
    ax.set_xlabel("Effect"); ax.set_ylabel("Cause")
    for i in range(sig.shape[0]):
        for j in range(sig.shape[1]):
            if not np.isnan(sig[i,j]) and sig[i,j] > 0:
                ax.text(j, i, "✓", ha="center", va="center", fontsize=14, color="white")

    plt.tight_layout()
    plt.savefig("granger_causality_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # See code/01_granger_causality.py for full practical
    np.random.seed(42)
    n = 400

    # True causal structure: X1 → X2 (lag 2), X2 → X3 (lag 1), X1 ⊥ X3
    x1 = np.random.normal(0, 1, n)
    x2 = 0.6 * np.roll(x1, 2) + np.random.normal(0, 0.5, n)
    x3 = 0.5 * np.roll(x2, 1) + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    pval_mat, sig_mat = granger_causality_matrix(df, max_lag=4)

    print("\nP-value matrix (row Granger-causes column):")
    print(pval_mat.round(4).to_string())
    print("\nSignificant links:")
    print(sig_mat.to_string())

    plot_granger_heatmap(pval_mat)
```

---

*← [01 — MTS Overview](./01_multivariate_ts_overview.md) | [Module README](./README.md) | Next: [03 — DTW Advanced](./03_dynamic_time_warping_advanced.md) →*
