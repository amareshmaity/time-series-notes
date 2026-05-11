# 📘 Module 08 — Evaluation & Metrics

> Knowing how to measure a forecast is just as important as making one. This module covers every metric, backtesting design principle, and statistical test used in production time series systems.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Calculate and interpret MAE, RMSE, MAPE, SMAPE, MASE, and wMAPE
- Choose the right metric for your domain and data characteristics
- Design a rigorous backtesting pipeline that mimics production conditions
- Run statistical significance tests to compare two or more models
- Evaluate and calibrate probabilistic forecasts using CRPS and coverage metrics

---

## 🔗 Prerequisites

- [Module 03 — Statistical Models](../03_statistical_models/README.md)
- [Module 04 — ML for Time Series](../04_ml_for_time_series/README.md)

---

## 📂 Module Contents

### Theory Notes
| File | Topic |
|------|-------|
| `01_error_metrics_MAE_RMSE_MAPE.md` | Point forecast metrics, their properties, edge cases, and failure modes |
| `02_skill_scores_and_relative_metrics.md` | MASE, wMAPE, skill scores, comparing across scales |
| `03_backtesting_design.md` | Rolling window, expanding window, gap strategy, avoiding leakage |
| `04_model_comparison_and_statistical_tests.md` | Diebold-Mariano test, Model Confidence Set, paired t-test |
| `05_calibration_for_probabilistic_models.md` | CRPS, coverage, sharpness, reliability diagrams |

### Code Practicals
| File | What It Demonstrates |
|------|----------------------|
| `code/01_metrics_implementation.py` | All metrics implemented from scratch + sklearn comparison |
| `code/02_backtesting_pipeline.py` | Rolling origin backtesting framework with parallel evaluation |
| `code/03_statistical_tests.py` | Diebold-Mariano test, Model Confidence Set with statsmodels |

---

## 🧠 Key Concepts

| Metric | Formula | Best When |
|--------|---------|-----------|
| **MAE** | Mean Absolute Error | Robust to outliers, interpretable |
| **RMSE** | Root Mean Squared Error | Penalizes large errors, same unit as target |
| **MAPE** | Mean Absolute % Error | Relative, but breaks when `y=0` |
| **SMAPE** | Symmetric MAPE | Avoids asymmetry in MAPE |
| **MASE** | Mean Absolute Scaled Error | Scale-free, uses naïve benchmark |
| **CRPS** | Continuous Ranked Probability Score | Evaluates probabilistic forecasts |

---

## 📌 Key Takeaways

1. **Never use MAPE** when actual values can be zero or near-zero.
2. **MASE** is the recommended metric for comparing across series of different scales.
3. Backtesting window count matters more than the split ratio — use at least 5–10 origins.
4. Use the **Diebold-Mariano test** to formally assess if one model is significantly better.
5. For probabilistic models, low CRPS with high coverage indicates a well-calibrated model.

---

## 📖 Further Reading

- [Forecasting: Principles and Practice — Accuracy Measures (Ch. 5)](https://otexts.com/fpp3/accuracy.html)
- [Another Look at Forecast Accuracy Metrics (Hyndman & Koehler, 2006)](https://robjhyndman.com/papers/mase.pdf)
- [Diebold-Mariano Test — statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.stattools.durbin_watson.html)

---

*← [Module 07](../07_forecasting_strategies/README.md) | Back to [Master README](../README.md) | Next: [Module 09](../09_anomaly_detection/README.md) →*
