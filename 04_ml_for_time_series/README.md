# 📘 Module 04 — Machine Learning for Time Series

> **Level**: 🟡 Intermediate | **Prerequisites**: [Module 02](../02_data_engineering/README.md) (feature engineering), [Module 03](../03_statistical_models/README.md) (statistical baselines)
>
> Tree-based ML models (XGBoost, LightGBM) consistently win time series Kaggle competitions. But applying them correctly requires time-safe cross-validation, leakage-free feature engineering, and careful hyperparameter tuning. This module covers it all.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Explain when ML models outperform statistical models for time series
- Implement walk-forward cross-validation correctly (no leakage)
- Build production-grade XGBoost, LightGBM, and CatBoost forecasting pipelines
- Tune hyperparameters using Optuna with time-safe CV
- Generate distribution-free prediction intervals using Conformal Prediction
- Stack multiple models into an ensemble for improved accuracy
- Compare all models on a fair leaderboard

---

## 🔗 Prerequisites

- [Module 02 — Data Engineering](../02_data_engineering/README.md) — lag/rolling/Fourier features
- [Module 03 — Statistical Models](../03_statistical_models/README.md) — seasonal naive baseline
- scikit-learn, XGBoost, LightGBM basics

---

## 📂 Module Contents

### 📒 Theory Notes

| File | Topic | Description |
|------|-------|-------------|
| [`01_ml_vs_statistical_models.md`](./01_ml_vs_statistical_models.md) | ML vs. Statistical | When ML wins, when ARIMA wins, hybrid strategies |
| [`02_timeseries_crossvalidation.md`](./02_timeseries_crossvalidation.md) | Time Series CV | Walk-forward, expanding window, purged CV — the right way |
| [`03_gradient_boosting_xgboost_lgbm.md`](./03_gradient_boosting_xgboost_lgbm.md) | XGBoost, LightGBM & CatBoost | Gradient boosting internals, CatBoost ordered boosting, TS-specific tuning |
| [`04_random_forest_and_tree_models.md`](./04_random_forest_and_tree_models.md) | Random Forest & Trees | RF, ExtraTrees, feature importance for TS |
| [`05_linear_models.md`](./05_linear_models.md) | Linear Models + Conformal PI | Ridge, Lasso, ElasticNet, Bayesian Ridge, conformal prediction intervals |
| [`06_model_stacking_and_ensembles.md`](./06_model_stacking_and_ensembles.md) | Ensembles & Stacking | Blending, stacking, weighted averaging, ML + statistical hybrid |

### 💻 Code Practicals

| File | What It Demonstrates |
|------|----------------------|
| [`code/01_ts_crossvalidation.py`](./code/01_ts_crossvalidation.py) | Walk-forward CV, expanding window, TimeSeriesSplit visualization |
| [`code/02_xgboost_lgbm_pipeline.py`](./code/02_xgboost_lgbm_pipeline.py) | Full XGBoost + LightGBM pipeline: features → fit → forecast → diagnostics |
| [`code/03_hyperparameter_tuning.py`](./code/03_hyperparameter_tuning.py) | Optuna-based tuning for LightGBM with walk-forward CV objective |
| [`code/04_model_comparison.py`](./code/04_model_comparison.py) | Full leaderboard: Linear → RF → XGBoost → LGBM → Ensemble vs. baselines |

---

## 🗺️ Learning Path

```
01_ml_vs_statistical_models.md         ← Understand the landscape
        ↓
02_timeseries_crossvalidation.md        ← Master CV before anything else
        ↓
03_gradient_boosting_xgboost_lgbm.md   ← Core ML model family
        ↓
04_random_forest_and_tree_models.md
        ↓
05_linear_models.md
        ↓
06_model_stacking_and_ensembles.md
        ↓
code/01 → code/02 → code/03 → code/04
```

---

## ⚡ The ML for TS Golden Rules

1. **Never use random K-Fold CV** — always use walk-forward / TimeSeriesSplit
2. **Shift before rolling** — all rolling features must use `.shift(1).rolling(w)`
3. **Encode on train only** — target encoding and normalization must fit on train set
4. **Lag features are the most important features** — lag 1, lag 7, lag 28, lag 365
5. **Beat seasonal naive first** — if XGBoost can't beat seasonal naive, the features are wrong
6. **CatBoost ordered boosting** is the only gradient boosting that natively respects temporal order during training
7. **Conformal prediction** gives valid prediction intervals for any ML model without distributional assumptions

---

## 🧠 ML Model Family Overview

| Model | Handles Non-linearity | Multi-step | External Regressors | Interpretability |
|-------|----------------------|-----------|--------------------|--------------------|
| Ridge / Lasso | ❌ | Via MIMO | ✅ | High |
| Random Forest | ✅ | Via MIMO | ✅ | Medium (feature importance) |
| **XGBoost** | ✅ | Via MIMO | ✅ | Medium (SHAP) |
| **LightGBM** | ✅ | Via MIMO | ✅ | Medium (SHAP) |
| **CatBoost** | ✅ | Via MIMO | ✅ | Medium (SHAP, ordered boosting) |
| Stacked Ensemble | ✅ | Via MIMO | ✅ | Low |

---

## 📖 Further Reading

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Forecasting with Gradient Boosting — Skforecast](https://skforecast.org/latest/)
- [SHAP for Time Series Interpretation](https://shap.readthedocs.io/)

---

*← [Module 03 — Statistical Models](../03_statistical_models/README.md) | Back to [Master README](../README.md) | Next: [Module 05 — Deep Learning](../05_deep_learning_models/README.md) →*
