# 04 — Hierarchical Forecasting

> **Module**: 07 Forecasting Strategies | **File**: 4 of 6
>
> Real-world demand, energy, and logistics data naturally form hierarchies — SKUs roll up to categories, categories to regions, regions to national totals. Hierarchical forecasting ensures that predictions at every level are consistent with each other — a requirement for any production planning system.

---

## Table of Contents

1. [Hierarchy Structures](#1-hierarchy-structures)
2. [The Coherence Problem](#2-the-coherence-problem)
3. [Bottom-Up Reconciliation](#3-bottom-up-reconciliation)
4. [Top-Down Reconciliation](#4-top-down-reconciliation)
5. [Middle-Out Reconciliation](#5-middle-out-reconciliation)
6. [Optimal Reconciliation — MinT](#6-optimal-reconciliation--mint)
7. [Implementation with hierarchicalforecast](#7-implementation-with-hierarchicalforecast)
8. [Grouped Hierarchies](#8-grouped-hierarchies)
9. [Production Considerations](#9-production-considerations)

---

## 1. Hierarchy Structures

### 1.1 Standard Hierarchy

A hierarchy groups series at multiple aggregation levels:

```
Level 0 (Total):                          Total Sales
                                               │
Level 1 (Region):              North           South          East
                                 │               │              │
Level 2 (Category):    Elec  Food  Cloth   Elec  Food    Elec  Food
                         │     │     │       │     │       │     │
Level 3 (SKU):         SKU1 SKU2 SKU3 ...  SKU7 SKU8   SKU12 ...
```

The constraint: **at every point in time, children must sum to parent**.

```
y_Total(t) = y_North(t) + y_South(t) + y_East(t)
y_North(t) = y_Elec_North(t) + y_Food_North(t) + y_Cloth_North(t)
```

### 1.2 Summing Matrix `S`

The **summing matrix** `S` encodes all aggregation relationships:

```
S · y_bottom = y_all_levels

Dimensions:
  y_bottom : (n_bottom, 1)   — bottom-level (SKU) forecasts
  S        : (n_all, n_bottom) — aggregation structure
  y_all    : (n_all, 1)      — all-level forecasts

Example (3 levels, 4 bottom-level series A,B,C,D under 2 mid-level groups):
         A  B  C  D
  Total [1  1  1  1]   y_Total = A + B + C + D
  G1    [1  1  0  0]   y_G1    = A + B
  G2    [0  0  1  1]   y_G2    = C + D
  A     [1  0  0  0]
  B     [0  1  0  0]
  C     [0  0  1  0]
  D     [0  0  0  1]
```

### 1.3 Building the Summing Matrix

```python
import pandas as pd
import numpy as np

def build_summing_matrix(hierarchy_df: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    """
    Build the summing matrix S from a hierarchy specification.

    Parameters
    ----------
    hierarchy_df : DataFrame with one column per level, rows = bottom-level series.
                   Example columns: ['total', 'region', 'category', 'sku']
                   Each row is one SKU with its parent labels filled in.

    Returns
    -------
    S            : summing matrix as DataFrame (index=all nodes, columns=bottom nodes)
    all_nodes    : list of all unique node names (all levels)
    bottom_nodes : list of bottom-level node names
    """
    bottom_col   = hierarchy_df.columns[-1]   # lowest-level column
    bottom_nodes = hierarchy_df[bottom_col].unique().tolist()
    all_nodes    = []

    # Collect all unique nodes at every level
    for col in hierarchy_df.columns:
        all_nodes.extend(hierarchy_df[col].unique().tolist())
    all_nodes = list(dict.fromkeys(all_nodes))  # deduplicate, preserve order

    S = pd.DataFrame(0, index=all_nodes, columns=bottom_nodes)

    # Bottom-level nodes map to themselves
    for node in bottom_nodes:
        S.loc[node, node] = 1

    # Upper-level nodes aggregate their descendants
    for _, row in hierarchy_df.iterrows():
        sku = row[bottom_col]
        for col in hierarchy_df.columns[:-1]:
            parent = row[col]
            S.loc[parent, sku] = 1

    return S, all_nodes, bottom_nodes
```

---

## 2. The Coherence Problem

### 2.1 Why Base Forecasts are Incoherent

When you forecast each level independently:

```
Base forecast for Day 1:
  Total  → ŷ_Total = 1,000
  North  → ŷ_North = 400
  South  → ŷ_South = 350
  East   → ŷ_East  = 260

Sum check: 400 + 350 + 260 = 1,010 ≠ 1,000

→ INCOHERENT — bottom-up total doesn't match top-level forecast
→ Planning system will have conflicting inputs
→ Must reconcile before use
```

### 2.2 Formal Definition

Let `ŷ` be the vector of base forecasts at all levels. Coherent forecasts `ỹ` satisfy:

```
ỹ = S · P · ŷ

Where:
  P    : reconciliation mapping matrix (n_bottom × n_all)
  S    : summing matrix (n_all × n_bottom)
  S·P  : combined mapping that produces coherent forecasts

Different methods choose P differently.
```

---

## 3. Bottom-Up Reconciliation

### 3.1 Concept

Forecast **only the bottom level** (SKUs). Aggregate up to get all other levels. Simple and exact — no incoherence possible.

```
ỹ_Total = Σᵢ ŷ_SKUᵢ
ỹ_Region = Σᵢ∈region ŷ_SKUᵢ
ỹ_Category = Σᵢ∈category ŷ_SKUᵢ
```

The reconciliation mapping `P_BU` simply picks the bottom-level rows:
```
P_BU = [0 ... 0 | I_n_bottom]   (identity for bottom, zeros for upper)
```

### 3.2 Implementation

```python
import pandas as pd
import numpy as np

def bottom_up_reconcile(
    bottom_forecasts: pd.DataFrame,
    summing_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Bottom-up reconciliation: aggregate bottom-level forecasts to all levels.

    Parameters
    ----------
    bottom_forecasts : DataFrame, index=bottom-level series, columns=forecast horizons
    summing_matrix   : S matrix from build_summing_matrix()

    Returns
    -------
    all_level_forecasts : DataFrame with coherent forecasts at all hierarchy levels
    """
    S = summing_matrix.loc[:, bottom_forecasts.index]  # align columns
    reconciled = S.values @ bottom_forecasts.values    # matrix multiplication
    return pd.DataFrame(reconciled, index=summing_matrix.index,
                        columns=bottom_forecasts.columns)


# ── Example ────────────────────────────────────────────────────────────────────
# bottom_forecasts: each row is one SKU, columns are forecast steps
bottom_fc = pd.DataFrame({
    "h1": [100, 80, 150, 200],
    "h2": [105, 82, 148, 210],
}, index=["SKU_A", "SKU_B", "SKU_C", "SKU_D"])

# After building S matrix:
# bu_fc = bottom_up_reconcile(bottom_fc, S)
# → Total, Group, SKU forecasts are all mutually consistent
```

### 3.3 Pros & Cons

| ✅ Pros                                              | ❌ Cons                                                     |
|------------------------------------------------------|-------------------------------------------------------------|
| Guaranteed coherent — no incoherence possible        | Upper-level accuracy may suffer (loses top-level signal)    |
| Simple — only need bottom-level model                | Errors at bottom level propagate up without correction      |
| Preferred when bottom-level data is reliable          | Ignores upper-level patterns                                |

---

## 4. Top-Down Reconciliation

### 4.1 Concept

Forecast **only the top level**. Disaggregate down using historical proportions.

```
ỹ_SKU_A = ŷ_Total × p_A

Where p_A = historical proportion of SKU A in Total
  p_A = mean(y_SKU_A / y_Total)   across historical periods
```

### 4.2 Proportion Methods

```python
def compute_top_down_proportions(
    historical_df: pd.DataFrame,
    id_col: str,
    target_col: str,
    total_id: str = "Total",
    method: str = "average_historical",
) -> pd.Series:
    """
    Compute disaggregation proportions for top-down reconciliation.

    Parameters
    ----------
    historical_df : long-format DataFrame with all levels
    id_col        : series identifier column
    target_col    : target column
    total_id      : identifier of the top-level (total) series
    method        : 'average_historical' | 'proportion_of_averages' | 'forecast_proportions'

    Returns
    -------
    proportions : Series indexed by bottom-level series IDs, values sum to 1.0
    """
    total_series = historical_df[historical_df[id_col] == total_id][target_col]
    total_mean   = total_series.mean()

    bottom_ids   = [uid for uid in historical_df[id_col].unique() if uid != total_id]
    proportions  = {}

    for uid in bottom_ids:
        uid_series = historical_df[historical_df[id_col] == uid][target_col]

        if method == "average_historical":
            # Method 1: average of (series / total) ratios
            aligned = pd.concat([uid_series.reset_index(drop=True),
                                  total_series.reset_index(drop=True)], axis=1)
            aligned.columns = ["uid", "total"]
            aligned = aligned[(aligned["total"] > 0)]
            proportions[uid] = (aligned["uid"] / aligned["total"]).mean()

        elif method == "proportion_of_averages":
            # Method 2: ratio of averages (more stable)
            proportions[uid] = uid_series.mean() / total_mean

    # Normalize so proportions sum to 1
    total_prop = sum(proportions.values())
    return pd.Series({k: v / total_prop for k, v in proportions.items()})
```

### 4.3 Pros & Cons

| ✅ Pros                                    | ❌ Cons                                                      |
|--------------------------------------------|--------------------------------------------------------------|
| Coherent by construction                   | Bottom-level forecasts driven entirely by proportions        |
| Leverages strong top-level signal          | Historical proportions may not hold in the future            |
| Works when bottom-level history is sparse  | Cannot capture bottom-level specific events                  |

---

## 5. Middle-Out Reconciliation

Forecast at an **intermediate level** (e.g., category). Aggregate up to Total, disaggregate down to SKU.

```
1. Forecast at category level → ŷ_Category
2. Aggregate: ỹ_Total = Σ ŷ_Category           (bottom-up from categories)
3. Disaggregate: ỹ_SKU = ŷ_Category × p_SKU|Category  (top-down from category)
```

**Best for**: When the chosen intermediate level has the most reliable forecast signal (e.g., category forecasts are more stable than SKU forecasts but more precise than national totals).

---

## 6. Optimal Reconciliation — MinT

### 6.1 Motivation

Bottom-up and top-down both discard information from some levels. **MinT (Minimum Trace Reconciliation)** uses forecasts at **all levels simultaneously** and finds the mathematically optimal reconciliation.

### 6.2 MinT Formula

```
ỹ = S (SᵀW⁻¹S)⁻¹ SᵀW⁻¹ ŷ

Where:
  ŷ   : base forecasts vector (all levels, shape n_all)
  S   : summing matrix (n_all × n_bottom)
  W   : covariance matrix of base forecast errors
  ỹ   : reconciled (coherent) forecasts

The mapping P_MinT = (SᵀW⁻¹S)⁻¹ Sᵀ W⁻¹ minimizes:
  tr(Var(ỹ - y))   — the trace (total variance) of reconciled forecast errors
```

### 6.3 Covariance Estimators for W

| Method              | W Estimator                               | When to Use                        |
|---------------------|-------------------------------------------|------------------------------------|
| `ols`               | W = I (identity)                          | Simple, fast baseline              |
| `wls_struct`        | W = diag(S·1) (structural weights)        | Default; works well in practice    |
| `wls_var`           | W = diag(σ̂²) (in-sample residual var.)   | When variance estimates are stable |
| `mint_shrink`       | W = shrinkage estimator of covariance     | Best; handles high-dimensional W   |
| `mint_cov`          | W = full empirical covariance             | Risky — needs many residuals        |

---

## 7. Implementation with hierarchicalforecast

```python
import pandas as pd
import numpy as np
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, TopDown, MinTrace
from hierarchicalforecast.utils import aggregate

# ── Step 1: Define the hierarchy ──────────────────────────────────────────────
# hiers: list of aggregation levels (from top to bottom)
# Each sublist = a grouping key path

hiers = [
    ["total"],
    ["total", "region"],
    ["total", "region", "category"],
    ["total", "region", "category", "sku"],   # bottom level
]

# ── Step 2: Prepare panel data in Nixtla format ───────────────────────────────
# df: [unique_id, ds, y]  — bottom-level series only
# S_df, tags = aggregate(df, hiers) aggregates up all levels

S_df, tags = aggregate(df_bottom, hiers)
# S_df: long-format DataFrame with all hierarchy levels
# tags: dict mapping level names to their unique_ids

print(f"Total series: {S_df['unique_id'].nunique()}")
print(f"Bottom series: {len(tags['sku'])}")

# ── Step 3: Fit base forecasters ──────────────────────────────────────────────
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS

# Fit base models on all hierarchy levels
sf = StatsForecast(
    models=[AutoARIMA(season_length=12), AutoETS(season_length=12)],
    freq="M",
    n_jobs=-1,
)
sf.fit(S_df)
base_forecasts = sf.predict(h=12, level=[80, 95])

# ── Step 4: Reconcile with MinT ───────────────────────────────────────────────
hrec = HierarchicalReconciliation(reconcilers=[
    BottomUp(),
    TopDown(method="forecast_proportions"),
    MinTrace(method="mint_shrink"),
    MinTrace(method="ols"),
])

reconciled_df = hrec.reconcile(
    Y_hat_df=base_forecasts,
    Y_df=S_df,
    S=S_df,    # actually use S matrix from aggregate()
    tags=tags,
)

print(reconciled_df.head(20))
# Columns: unique_id, ds, AutoARIMA/BottomUp, AutoARIMA/MinTrace-mint_shrink, ...
```

### 7.1 Evaluating Reconciled Forecasts

```python
from hierarchicalforecast.evaluation import HierarchicalEvaluation

def evaluate_reconciliation(
    Y_test_df: pd.DataFrame,
    Y_hat_df: pd.DataFrame,
    tags: dict,
    metrics: list = None,
) -> pd.DataFrame:
    """
    Evaluate reconciled forecasts at each hierarchy level.

    Parameters
    ----------
    Y_test_df : actual values in long format
    Y_hat_df  : reconciled forecasts
    tags      : level → series_ids mapping from aggregate()
    metrics   : list of metric functions (default: MAE, RMSE)
    """
    from hierarchicalforecast.evaluation import HierarchicalEvaluation
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    if metrics is None:
        metrics = [mean_absolute_error, mean_squared_error]

    heval = HierarchicalEvaluation(evaluators=metrics)
    evaluation = heval.evaluate(
        Y_hat_df=Y_hat_df,
        Y_test_df=Y_test_df,
        tags=tags,
    )
    return evaluation


# Coherence check — are forecasts actually coherent after reconciliation?
def check_coherence(reconciled_df: pd.DataFrame, S: np.ndarray,
                    tags: dict, tol: float = 1e-6) -> bool:
    """Check that reconciled forecasts satisfy S·y_bottom = y_all."""
    bottom_ids = tags[list(tags.keys())[-1]]   # last level = bottom
    for ds, group in reconciled_df.groupby("ds"):
        bottom_vals = group[group["unique_id"].isin(bottom_ids)]["forecast"].values
        expected    = S @ bottom_vals
        actual      = group["forecast"].values
        if not np.allclose(expected, actual, atol=tol):
            return False
    return True
```

---

## 8. Grouped Hierarchies

Not all hierarchies are strictly nested. **Grouped** hierarchies have multiple crossing dimensions:

```
Example: Sales data with dimensions:
  - Region: North, South
  - Category: Electronics, Food

Grouped structure:
  Total
  ├── By Region: North, South
  ├── By Category: Electronics, Food
  ├── Region × Category: North/Electronics, North/Food, South/Electronics, South/Food
  └── (bottom level same as Region × Category for 2-dim case)

→ The bottom level is the Cartesian product of all crossing dimensions
```

```python
# In hierarchicalforecast, grouped hierarchies are specified as:
hiers = [
    ["total"],
    ["total", "region"],
    ["total", "category"],
    ["total", "region", "category"],   # crossing level = bottom
]

# Note: ["total", "region", "category"] vs ["total", "category", "region"]
# creates different summing matrices — order matters for hierarchy interpretation
```

---

## 9. Production Considerations

### 9.1 Retraining Schedule

```
Daily retraining cycle:
  1. Ingest new actuals for all bottom-level series
  2. Update base forecaster on rolling window
  3. Re-aggregate (S matrix is static unless hierarchy changes)
  4. Run MinT reconciliation (fast — just matrix ops)
  5. Push coherent forecasts to downstream planning systems
```

### 9.2 Handling Hierarchy Changes

```python
def update_hierarchy(
    old_S: pd.DataFrame,
    new_bottom_series: list[str],
    hierarchy_spec: pd.DataFrame,
) -> pd.DataFrame:
    """
    Update the summing matrix when hierarchy structure changes.
    (e.g., new SKU launches, category reassignment, regional restructure)

    New series default to equal proportion split from parent.
    """
    # Rebuild S from scratch whenever hierarchy changes
    new_S, _, _ = build_summing_matrix(hierarchy_spec)
    print(f"Hierarchy updated: {old_S.shape[1]} → {new_S.shape[1]} bottom series")
    return new_S
```

### 9.3 Coherence Check in Production

```python
def production_coherence_check(forecasts_df: pd.DataFrame,
                                S: np.ndarray,
                                bottom_ids: list,
                                tolerance: float = 0.01) -> dict:
    """
    Fast vectorized coherence check. Run as data quality step before delivery.

    Returns dict with pass/fail and maximum absolute coherence error.
    """
    results = {"passed": True, "max_coherence_error": 0.0, "failed_timestamps": []}

    for ds, grp in forecasts_df.groupby("ds"):
        bottom_vals = grp[grp["unique_id"].isin(bottom_ids)]["forecast"].values
        all_vals    = grp["forecast"].values
        implied     = S @ bottom_vals

        err = np.abs(implied - all_vals).max()
        results["max_coherence_error"] = max(results["max_coherence_error"], err)

        if err > tolerance:
            results["passed"] = False
            results["failed_timestamps"].append(str(ds))

    return results
```

---

*← [03 — Global vs Local Models](./03_global_vs_local_models.md) | [Module README](./README.md) | Next: [05 — Probabilistic Forecasting](./05_probabilistic_forecasting.md) →*
