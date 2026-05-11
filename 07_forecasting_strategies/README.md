# 📘 Module 07 — Forecasting Strategies

> Beyond fitting a single model — learn the strategies that determine how predictions are generated, reconciled, and made uncertainty-aware. These strategies are what separate production systems from prototypes.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Implement Direct, Recursive, and MIMO multi-step forecasting strategies
- Choose the right strategy based on horizon length and computational budget
- Build hierarchical forecasting pipelines with bottom-up and optimal reconciliation
- Generate probabilistic forecasts with prediction intervals
- Apply conformal prediction for distribution-free uncertainty quantification
- Understand global vs. local model trade-offs for multi-series forecasting

---

## 🔗 Prerequisites

- [Module 03 — Statistical Models](../03_statistical_models/README.md)
- [Module 04 — ML for Time Series](../04_ml_for_time_series/README.md)

---

## 📂 Module Contents

### Theory Notes
| File | Topic |
|------|-------|
| `01_direct_vs_recursive_vs_MIMO.md` | Strategy comparison, error accumulation, computational cost |
| `02_multi_step_forecasting.md` | Horizon selection, chained prediction, direct multi-output |
| `03_global_vs_local_models.md` | One model per series vs. one model for all series |
| `04_hierarchical_forecasting.md` | Hierarchy structures, bottom-up, top-down, MinT reconciliation |
| `05_probabilistic_forecasting.md` | Quantile regression, prediction intervals, conformal coverage |
| `06_conformal_prediction_for_ts.md` | Split conformal, adaptive conformal, EnbPI for TS |

### Code Practicals
| File | What It Demonstrates |
|------|----------------------|
| `code/01_multi_step_strategies.py` | Direct vs Recursive vs MIMO comparison on the same dataset |
| `code/02_hierarchical_reconciliation.py` | MinT reconciliation with hierarchicalforecast |
| `code/03_probabilistic_forecast.py` | Quantile regression + conformal prediction intervals |

---

## 🧠 Key Concepts

- **Direct** — Train one model per horizon step. No error accumulation. High memory cost.
- **Recursive** — Train one model, feed predictions back as inputs. Simple but errors compound.
- **MIMO** — One model predicts all steps simultaneously. Best trade-off for many use cases.
- **Hierarchical Forecasting** — Forecasting at multiple aggregation levels (e.g., SKU → category → total).
- **MinT Reconciliation** — Minimum trace reconciliation that makes hierarchy forecasts coherent.
- **Conformal Prediction** — Provides coverage-guaranteed intervals without distributional assumptions.

---

## 📌 Key Takeaways

1. For horizons > 12, prefer **Direct or MIMO** over Recursive to avoid error accumulation.
2. **Global models** (one model, many series) are the modern standard — they leverage cross-series patterns.
3. Hierarchical forecasting must always be **reconciled** — incoherent forecasts create operational chaos.
4. Conformal prediction is the most **reliable** way to generate prediction intervals in practice.

---

## 📖 Further Reading

- [Forecasting: Principles and Practice — Hierarchical (Ch. 11)](https://otexts.com/fpp3/hierarchical.html)
- [hierarchicalforecast Library (Nixtla)](https://nixtlaverse.nixtla.io/hierarchicalforecast/)
- [Conformal Prediction for Time Series (Angelopoulos et al.)](https://arxiv.org/abs/2107.07511)

---

*← [Module 06](../06_transformer_and_foundation_models/README.md) | Back to [Master README](../README.md) | Next: [Module 08](../08_evaluation_and_metrics/README.md) →*
