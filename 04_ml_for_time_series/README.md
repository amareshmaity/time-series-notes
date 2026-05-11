# 📘 Module 04 — Machine Learning for Time Series

> Frame time series as supervised learning problems and leverage the power of gradient boosting, random forests, and other ML algorithms — while respecting the temporal structure of data.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Reframe a time series forecasting task as a tabular regression problem
- Engineer lag, rolling, and calendar features for tree-based models
- Train and tune XGBoost and LightGBM models on time series data
- Apply time-series-aware cross-validation (TimeSeriesSplit, walk-forward)
- Avoid data leakage in feature construction and validation
- Interpret feature importance for temporal predictions
- Combine ML predictions with statistical residuals (hybrid models)

---

## 🔗 Prerequisites

- [Module 02 — Data Engineering](../02_data_engineering/README.md)
- [Module 03 — Statistical Models](../03_statistical_models/README.md)
- Familiarity with scikit-learn API

---

## 📂 Module Contents

### Theory Notes
| File | Topic |
|------|-------|
| `01_ml_framing_regression_approach.md` | Supervised learning framing, target variable, train/test design |
| `02_feature_engineering_for_ml.md` | Lag features, rolling stats, calendar encodings, Fourier terms |
| `03_xgboost_lightgbm_for_ts.md` | Tree-based models, hyperparameter tuning, early stopping for TS |
| `04_random_forest_ts.md` | Random forest for forecasting, feature importance, limitations |
| `05_cross_validation_for_ts.md` | TimeSeriesSplit, walk-forward validation, gap strategy |
| `06_target_encoding_and_lags.md` | Target encoding, mean encoding, lag target leakage prevention |

### Code Practicals
| File | What It Demonstrates |
|------|----------------------|
| `code/01_ml_framing.py` | Converting TS to supervised dataset with lag matrix construction |
| `code/02_xgboost_ts.py` | XGBoost training, Optuna tuning, multi-step forecasting |
| `code/03_lightgbm_ts.py` | LightGBM with early stopping, SHAP feature importance |
| `code/04_ts_cv.py` | TimeSeriesSplit, walk-forward CV, gap parameter, performance curves |

---

## 🧠 Key Concepts

- **Supervised Framing** — Convert `[y(t), y(t-1), ..., y(t-p)]` into feature matrix `X` and target `y`.
- **Lag Features** — Most important features for tree-based models; capture autocorrelation.
- **Walk-Forward Validation** — Repeatedly train on past, predict next window; mimics production reality.
- **Data Leakage** — Using rolling features without respecting the temporal split boundary.
- **SHAP Values** — Model-agnostic feature attributions that explain which lags drive predictions.

---

## 📌 Key Takeaways

1. LightGBM/XGBoost are among the **most competitive** models in Kaggle TS competitions.
2. Feature engineering matters more than model choice — lags and rolling windows dominate.
3. **Never use standard k-fold CV** on time series — always use TimeSeriesSplit or walk-forward.
4. Use a **hold-out gap** between train and validation to prevent look-ahead bias.
5. Hybrid models (statistical for trend + ML for residuals) often outperform both individually.

---

## 📖 Further Reading

- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/stable/)
- [Kaggle — M5 Forecasting Winners Write-up](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion)
- [Skforecast Library](https://skforecast.org/) — scikit-learn + TS forecasting

---

*← [Module 03](../03_statistical_models/README.md) | Back to [Master README](../README.md) | Next: [Module 05](../05_deep_learning_models/README.md) →*
