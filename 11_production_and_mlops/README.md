# 📘 Module 11 — Production & MLOps for Time Series

> Bridge the gap between a notebook model and a live production system. This module covers the full lifecycle — pipelines, feature stores, monitoring, drift detection, retraining, and serving.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Design end-to-end time series ML pipelines with proper abstraction layers
- Implement a feature store strategy for temporal features without leakage
- Version models and datasets using MLflow for reproducibility
- Detect data drift and concept drift in production TS streams
- Build automated retraining triggers based on performance degradation
- Serve forecasting models via REST APIs with low-latency responses

---

## 🔗 Prerequisites

- [Module 04 — ML for Time Series](../04_ml_for_time_series/README.md)
- [Module 08 — Evaluation & Metrics](../08_evaluation_and_metrics/README.md)
- Basic Docker / API concepts

---

## 📂 Module Contents

### Theory Notes
| File | Topic |
|------|-------|
| `01_ts_pipeline_architecture.md` | Pipeline design, ingestion → features → train → serve → monitor |
| `02_feature_stores_for_ts.md` | Point-in-time correct joins, offline vs. online stores, Feast |
| `03_model_registry_and_versioning.md` | MLflow tracking, model registry, experiment comparison |
| `04_drift_detection_and_monitoring.md` | Data drift (PSI, KS test), concept drift (performance monitoring) |
| `05_retraining_strategies.md` | Scheduled retrain, triggered retrain, online learning, warm start |
| `06_serving_ts_models.md` | FastAPI serving, batch vs. real-time, caching, latency optimization |

### Code Practicals
| File | What It Demonstrates |
|------|----------------------|
| `code/01_pipeline_template.py` | Modular sklearn-compatible TS pipeline with transformers |
| `code/02_drift_detection.py` | PSI-based drift detector, performance monitoring dashboard |
| `code/03_serving_api.py` | FastAPI endpoint for model inference with caching |

---

## 🧠 Key Concepts

- **Point-in-Time Correct Join** — Feature lookups must use only data available at prediction time.
- **Data Drift** — Input distribution changes (e.g., new seasonality pattern emerges).
- **Concept Drift** — Relationship between inputs and target changes (model becomes stale).
- **PSI (Population Stability Index)** — Measures distribution shift between train and serving data.
- **Warm Start** — Retrain incrementally from the previous model rather than from scratch.
- **Online Learning** — Update model weights continuously with each new observation.

---

## 📌 Key Takeaways

1. A TS pipeline without **point-in-time correct** feature serving will silently leak future data.
2. Monitor **both data drift and model performance** — they signal different types of issues.
3. **Schedule retraining** at minimum; build trigger-based retraining for high-stakes systems.
4. Use **MLflow** for every experiment — you will forget which config produced that good result.
5. For high-throughput serving, pre-compute and cache forecasts rather than computing on-demand.

---

## 📖 Further Reading

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Evidently AI — Data & Model Monitoring](https://docs.evidentlyai.com/)
- [Feast Feature Store](https://docs.feast.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

*← [Module 10](../10_classification_and_clustering/README.md) | Back to [Master README](../README.md) | Next: [Module 12](../12_multivariate_and_advanced_topics/README.md) →*
