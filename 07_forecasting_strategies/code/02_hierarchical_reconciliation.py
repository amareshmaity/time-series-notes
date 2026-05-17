"""
code/02_hierarchical_reconciliation.py
=======================================
Module 07 — Forecasting Strategies
Practical: Hierarchical forecasting with bottom-up, top-down, and MinT reconciliation.

Demonstrates:
  - Building a 3-level retail hierarchy (Total → Region → SKU)
  - Fitting base forecasters (ETS/ARIMA via statsforecast)
  - Reconciling with BottomUp, TopDown, and MinTrace(mint_shrink)
  - Evaluating coherence and per-level accuracy
  - Visualizing reconciled vs. base forecasts

Requirements:
  pip install hierarchicalforecast statsforecast
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Retail Hierarchy Dataset
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
N    = 48    # 48 months (~4 years)
FREQ = "ME"  # Month-end

# Define bottom-level series (Region × Category = 6 SKU-level series)
SKUS = {
    "North/Electronics": {"trend": 0.8, "season_amp": 20, "base": 150},
    "North/Food":        {"trend": 0.3, "season_amp": 10, "base": 300},
    "North/Clothing":    {"trend": 0.5, "season_amp": 30, "base": 200},
    "South/Electronics": {"trend": 1.0, "season_amp": 15, "base": 120},
    "South/Food":        {"trend": 0.2, "season_amp":  8, "base": 250},
    "South/Clothing":    {"trend": 0.4, "season_amp": 25, "base": 180},
}

dates = pd.date_range("2020-01-31", periods=N, freq=FREQ)

bottom_rows = []
for uid, params in SKUS.items():
    t      = np.arange(N)
    trend  = params["trend"] * t
    season = params["season_amp"] * np.sin(2 * np.pi * t / 12)
    noise  = np.random.normal(0, params["base"] * 0.05, N)
    values = params["base"] + trend + season + noise

    for i, (d, v) in enumerate(zip(dates, values)):
        bottom_rows.append({"unique_id": uid, "ds": d, "y": max(v, 0)})

df_bottom = pd.DataFrame(bottom_rows)
print(f"Bottom-level series: {df_bottom['unique_id'].nunique()}")
print(df_bottom.groupby("unique_id")["y"].describe().round(1)[["mean", "std", "min", "max"]])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Aggregate to All Hierarchy Levels
# ─────────────────────────────────────────────────────────────────────────────

try:
    from hierarchicalforecast.utils import aggregate

    # Parse region and category from uid
    df_bottom["region"]   = df_bottom["unique_id"].str.split("/").str[0]
    df_bottom["category"] = df_bottom["unique_id"].str.split("/").str[1]

    hiers = [
        ["total"],
        ["total", "region"],
        ["total", "category"],
        ["total", "region", "category"],   # bottom level (matches unique_id structure)
    ]

    # Prepare long-format with hierarchy columns
    df_hier = df_bottom[["unique_id", "ds", "y", "region", "category"]].copy()
    df_hier["total"] = "Total"

    S_df, tags = aggregate(
        df_hier[["total", "region", "category", "ds", "y"]].rename(
            columns={"y": "y"}
        ),
        [["total"], ["total", "region"], ["total", "category"],
         ["total", "region", "category"]],
    )
    print(f"\nAll hierarchy levels: {S_df['unique_id'].nunique()} series")
    for level, ids in tags.items():
        print(f"  {level}: {ids}")

    USE_HIERARCHICAL_LIB = True

except ImportError:
    print("\n⚠ hierarchicalforecast not installed.")
    print("  pip install hierarchicalforecast statsforecast")
    print("  Falling back to manual implementation.\n")
    USE_HIERARCHICAL_LIB = False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Manual Aggregation (fallback if library not available)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_manually(df_bottom: pd.DataFrame) -> pd.DataFrame:
    """Manually build all hierarchy levels by summing bottom-level series."""
    rows = []

    # Bottom level
    for _, row in df_bottom.iterrows():
        rows.append(row.to_dict())

    # Region level
    for (region, ds), grp in df_bottom.groupby(["region", "ds"]):
        rows.append({"unique_id": region, "ds": ds, "y": grp["y"].sum()})

    # Total level
    for ds, grp in df_bottom.groupby("ds"):
        rows.append({"unique_id": "Total", "ds": ds, "y": grp["y"].sum()})

    return pd.DataFrame(rows)[["unique_id", "ds", "y"]]


df_all = aggregate_manually(df_bottom)
print(f"\nAll levels: {df_all['unique_id'].nunique()} series")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Train/Test Split
# ─────────────────────────────────────────────────────────────────────────────

H         = 6    # forecast horizon (6 months)
TRAIN_END = dates[-H - 1]

df_train = df_all[df_all["ds"] <= TRAIN_END].copy()
df_test  = df_all[df_all["ds"] >  TRAIN_END].copy()

print(f"\nTrain: {df_train['ds'].min().date()} → {df_train['ds'].max().date()}")
print(f"Test : {df_test['ds'].min().date()}  → {df_test['ds'].max().date()}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Base Forecasts — Simple Exponential Smoothing per Series
# ─────────────────────────────────────────────────────────────────────────────

def simple_ets_forecast(series: np.ndarray, h: int, alpha: float = 0.3) -> np.ndarray:
    """
    Simple exponential smoothing (SES) — fast base forecaster.
    All forecasts = last smoothed level (flat forecast).
    """
    level = series[0]
    for y in series:
        level = alpha * y + (1 - alpha) * level
    return np.full(h, level)


base_forecasts = {}
for uid, grp in df_train.groupby("unique_id"):
    train_vals = grp.sort_values("ds")["y"].values
    base_forecasts[uid] = simple_ets_forecast(train_vals, h=H)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Reconciliation Methods
# ─────────────────────────────────────────────────────────────────────────────

BOTTOM_IDS = list(SKUS.keys())
REGION_IDS = ["North", "South"]
TOTAL_ID   = "Total"
ALL_IDS    = [TOTAL_ID] + REGION_IDS + BOTTOM_IDS


def bottom_up_reconcile(base_fc: dict, bottom_ids: list,
                         region_map: dict, total_id: str) -> dict:
    """
    Bottom-up: aggregate bottom forecasts upward.
    Discards base forecasts for upper levels.
    """
    reconciled = {}

    # Keep bottom level as-is
    for uid in bottom_ids:
        reconciled[uid] = base_fc[uid].copy()

    # Aggregate to region level
    for region, members in region_map.items():
        reconciled[region] = sum(reconciled[m] for m in members)

    # Aggregate to total
    reconciled[total_id] = sum(reconciled[m] for m in bottom_ids)

    return reconciled


def top_down_reconcile(base_fc: dict, bottom_ids: list,
                        train_df: pd.DataFrame, total_id: str) -> dict:
    """
    Top-down: use total forecast, disaggregate with historical proportions.
    """
    # Compute historical proportions (average share of each SKU in total)
    total_series = train_df[train_df["unique_id"] == total_id]["y"]
    proportions  = {}
    for uid in bottom_ids:
        uid_series = train_df[train_df["unique_id"] == uid]["y"]
        aligned    = pd.concat([uid_series.reset_index(drop=True),
                                  total_series.reset_index(drop=True)], axis=1)
        aligned.columns = ["uid", "total"]
        aligned = aligned[aligned["total"] > 0]
        proportions[uid] = (aligned["uid"] / aligned["total"]).mean()

    # Normalize
    total_prop = sum(proportions.values())
    for uid in proportions:
        proportions[uid] /= total_prop

    # Disaggregate total forecast
    reconciled = {}
    total_fc   = base_fc[total_id]
    for uid in bottom_ids:
        reconciled[uid] = total_fc * proportions[uid]

    # Re-aggregate to get coherent upper levels
    for region, members in region_map.items():
        reconciled[region] = sum(reconciled[m] for m in members)
    reconciled[total_id] = sum(reconciled[uid] for uid in bottom_ids)

    return reconciled


def mint_ols_reconcile(base_fc: dict, all_ids: list, bottom_ids: list,
                        region_map: dict, total_id: str) -> dict:
    """
    MinT-OLS reconciliation: W = Identity (simple OLS projection).
    Minimizes sum of squared adjustments to base forecasts.
    Coherent by construction.
    """
    n_all    = len(all_ids)
    n_bottom = len(bottom_ids)

    # Build summing matrix S (n_all × n_bottom)
    S = np.zeros((n_all, n_bottom))
    id_to_idx = {uid: i for i, uid in enumerate(all_ids)}
    btm_to_idx = {uid: j for j, uid in enumerate(bottom_ids)}

    # Total row
    S[id_to_idx[total_id], :] = 1

    # Region rows
    for region, members in region_map.items():
        for m in members:
            S[id_to_idx[region], btm_to_idx[m]] = 1

    # Bottom rows (identity)
    for uid in bottom_ids:
        S[id_to_idx[uid], btm_to_idx[uid]] = 1

    # Stack base forecasts as matrix (n_all × H)
    Y_hat = np.stack([base_fc[uid] for uid in all_ids])  # (n_all, H)

    # OLS reconciliation: P = (S'S)^{-1} S'   (W = I)
    STS_inv = np.linalg.pinv(S.T @ S)
    P       = STS_inv @ S.T              # (n_bottom, n_all)
    SP      = S @ P                      # (n_all, n_all) — projection matrix

    Y_rec = SP @ Y_hat   # (n_all, H) — reconciled forecasts

    reconciled = {}
    for i, uid in enumerate(all_ids):
        reconciled[uid] = Y_rec[i]

    return reconciled


# Define region membership
REGION_MAP = {
    "North": [uid for uid in BOTTOM_IDS if uid.startswith("North")],
    "South": [uid for uid in BOTTOM_IDS if uid.startswith("South")],
}

# Run all three reconciliation methods
rec_bu  = bottom_up_reconcile(base_forecasts, BOTTOM_IDS, REGION_MAP, TOTAL_ID)
rec_td  = top_down_reconcile(base_forecasts, BOTTOM_IDS, df_train, TOTAL_ID)
rec_ols = mint_ols_reconcile(base_forecasts, ALL_IDS, BOTTOM_IDS, REGION_MAP, TOTAL_ID)

print("\nReconciliation complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Coherence Check
# ─────────────────────────────────────────────────────────────────────────────

def check_coherence(reconciled: dict, region_map: dict,
                    bottom_ids: list, total_id: str, tol: float = 1e-6) -> bool:
    """Verify that reconciled forecasts satisfy all aggregation constraints."""
    # Check total = sum of bottom
    total_check = abs(reconciled[total_id] - sum(reconciled[b] for b in bottom_ids)).max()

    # Check region = sum of its members
    region_check = max(
        abs(reconciled[region] - sum(reconciled[m] for m in members)).max()
        for region, members in region_map.items()
    )

    ok = (total_check < tol) and (region_check < tol)
    print(f"  Max total aggregation error:  {total_check:.2e}  {'✅' if total_check < tol else '❌'}")
    print(f"  Max region aggregation error: {region_check:.2e}  {'✅' if region_check < tol else '❌'}")
    return ok


print("\nCoherence Check:")
print("  Bottom-Up:");  check_coherence(rec_bu,  REGION_MAP, BOTTOM_IDS, TOTAL_ID)
print("  Top-Down:");   check_coherence(rec_td,  REGION_MAP, BOTTOM_IDS, TOTAL_ID)
print("  MinT-OLS:");   check_coherence(rec_ols, REGION_MAP, BOTTOM_IDS, TOTAL_ID)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Accuracy Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def mae_for_method(reconciled: dict, df_test: pd.DataFrame) -> pd.DataFrame:
    """Compute MAE per series for a reconciliation method."""
    rows = []
    for uid, fc in reconciled.items():
        actual = df_test[df_test["unique_id"] == uid]["y"].values
        if len(actual) == 0:
            continue
        h   = min(len(fc), len(actual))
        mae = np.mean(np.abs(actual[:h] - fc[:h]))
        rows.append({"unique_id": uid, "mae": mae})
    return pd.DataFrame(rows).set_index("unique_id")


mae_base = mae_for_method(base_forecasts, df_test)
mae_bu   = mae_for_method(rec_bu,         df_test)
mae_td   = mae_for_method(rec_td,         df_test)
mae_ols  = mae_for_method(rec_ols,        df_test)

summary = pd.DataFrame({
    "Base (SES)":  mae_base["mae"],
    "Bottom-Up":   mae_bu["mae"],
    "Top-Down":    mae_td["mae"],
    "MinT-OLS":    mae_ols["mae"],
}).round(2)

print("\nMAE by Series and Reconciliation Method:")
print(summary)
print(f"\nMean MAE across all series:")
print(summary.mean().round(2))


# ─────────────────────────────────────────────────────────────────────────────
# 9. Visualization
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

methods = {
    "Base (SES)": base_forecasts,
    "Bottom-Up":  rec_bu,
    "Top-Down":   rec_td,
    "MinT-OLS":   rec_ols,
}
colors = {"Base (SES)": "gray", "Bottom-Up": "#4CAF50", "Top-Down": "#FF9800", "MinT-OLS": "#2196F3"}

for ax_idx, uid in enumerate(BOTTOM_IDS[:6]):
    ax = axes[ax_idx]
    actual     = df_all[df_all["unique_id"] == uid].sort_values("ds")

    # History
    history    = actual[actual["ds"] <= TRAIN_END]
    ax.plot(history["ds"], history["y"], color="black", linewidth=1.5, label="History")

    # Actual test
    test_part  = actual[actual["ds"] > TRAIN_END]
    ax.plot(test_part["ds"], test_part["y"], color="black", linewidth=2,
            linestyle=":", label="Actual")

    # Forecasts
    test_dates = test_part["ds"].values
    for name, fc_dict in methods.items():
        if uid in fc_dict:
            fc_vals = fc_dict[uid][:len(test_dates)]
            ax.plot(test_dates, fc_vals, color=colors[name],
                    linewidth=1.8, linestyle="--", label=name)

    ax.set_title(uid, fontsize=10)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3)
    if ax_idx == 0:
        ax.legend(fontsize=7, loc="upper left")

plt.suptitle("Hierarchical Reconciliation — SKU-Level Forecasts", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("hierarchical_reconciliation.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: hierarchical_reconciliation.png")
