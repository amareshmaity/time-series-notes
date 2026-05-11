# 📘 Module 09 — Anomaly Detection in Time Series

> Detect unusual events, outliers, and structural breaks in temporal data using statistical methods, machine learning, and deep learning approaches — both offline and in real-time.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Apply statistical control-chart methods (Z-score, IQR, CUSUM) for anomaly detection
- Use Isolation Forest and One-Class SVM for unsupervised anomaly detection
- Build autoencoder-based detectors for complex multivariate anomalies
- Implement LSTM-based anomaly detection using reconstruction error
- Deploy online anomaly detection for real-time streaming data
- Perform root cause analysis to trace anomalies back to their source

---

## 🔗 Prerequisites

- [Module 02 — Data Engineering](../02_data_engineering/README.md)
- [Module 05 — Deep Learning Models](../05_deep_learning_models/README.md)

---

## 📂 Module Contents

### Theory Notes
| File | Topic |
|------|-------|
| `01_statistical_anomaly_detection.md` | Z-score, IQR, CUSUM, Bollinger Bands, STL residuals |
| `02_isolation_forest_for_ts.md` | Isolation Forest, One-Class SVM, Local Outlier Factor for TS |
| `03_autoencoder_anomaly_detection.md` | Reconstruction error, threshold selection, VAE for TS |
| `04_lstm_based_anomaly_detection.md` | LSTM prediction error as anomaly signal, LSTMAD |
| `05_online_anomaly_detection.md` | Streaming anomaly detection, ADWIN, HBOS, river library |
| `06_root_cause_analysis.md` | Correlation networks, Granger causality for RCA, contribution analysis |

### Code Practicals
| File | What It Demonstrates |
|------|----------------------|
| `code/01_statistical_methods.py` | Z-score, rolling stats, CUSUM with visualization |
| `code/02_isolation_forest.py` | Isolation Forest on TS features, threshold tuning |
| `code/03_autoencoder_ad.py` | LSTM autoencoder, reconstruction error anomaly scoring |
| `code/04_online_detection.py` | Streaming anomaly detection with river library |

---

## 🧠 Key Concepts

- **Point Anomaly** — A single data point is anomalous (e.g., a sudden spike).
- **Contextual Anomaly** — A value is normal globally but anomalous given its context (time of day).
- **Collective Anomaly** — A sequence of values is anomalous (e.g., flatline in a sensor).
- **Reconstruction Error** — Autoencoders trained on normal data produce high error on anomalies.
- **CUSUM** — Cumulative sum chart that detects persistent shifts in mean.

---

## 📌 Key Takeaways

1. Start with **STL residual + 3-sigma rule** — catches most anomalies with no tuning.
2. Autoencoder approaches work best on **multivariate and collective anomalies**.
3. Threshold selection is the hardest part — always calibrate on labelled holdout if available.
4. For streaming systems, use **online algorithms** (ADWIN, HBOS) that update incrementally.
5. Anomaly detection without RCA is often useless in production — always trace the root cause.

---

## 📖 Further Reading

- [Anomaly Detection for Time Series — A Survey](https://arxiv.org/abs/2101.09372)
- [PyOD Library — Outlier Detection](https://pyod.readthedocs.io/en/latest/)
- [river — Online ML Library](https://riverml.xyz/)

---

*← [Module 08](../08_evaluation_and_metrics/README.md) | Back to [Master README](../README.md) | Next: [Module 10](../10_classification_and_clustering/README.md) →*
