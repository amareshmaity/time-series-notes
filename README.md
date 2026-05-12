# 📈 Time Series Notes — Industry-Standard Curriculum

> A comprehensive, structured knowledge base for AI Engineers covering Time Series Analysis and Forecasting — from fundamentals to production-grade systems.

---

## 🎯 About This Curriculum

This repository is organized as a **self-contained learning path** that covers:
- **Theory** — concepts, math intuition, and model internals
- **Practical** — runnable Python code with real datasets
- **Industry Patterns** — production pipelines, MLOps, and serving

Each module is independent but builds on the previous ones. Every module has its own `README.md` as an entry point.

---

## 📚 Curriculum Map

| # | Module | Core Topics | Level |
|---|--------|-------------|-------|
| [01](./01_foundations/README.md) | **Foundations** | What is TS, components, stationarity, ACF/PACF, decomposition | 🟢 Beginner |
| [02](./02_data_engineering/README.md) | **Data Engineering** | Resampling, missing values, outliers, feature engineering, windowing | 🟢 Beginner |
| [03](./03_statistical_models/README.md) | **Statistical Models** | AR, ARIMA, SARIMA, ARIMAX, ETS, VAR, Kalman filters, Prophet | 🟡 Intermediate |
| [04](./04_ml_for_time_series/README.md) | **ML for Time Series** | XGBoost, LightGBM, lag features, time-series CV | 🟡 Intermediate |
| [05](./05_deep_learning_models/README.md) | **Deep Learning Models** | LSTM, GRU, TCN, Seq2Seq, N-BEATS, TFT, data augmentation | 🟠 Advanced |
| [06](./06_transformer_and_foundation_models/README.md) | **Transformers & Foundation Models** | Informer, PatchTST, Chronos, TimeGPT, Moirai, zero-shot, fine-tuning | 🔴 Expert |
| [07](./07_forecasting_strategies/README.md) | **Forecasting Strategies** | Direct/Recursive/MIMO, hierarchical, probabilistic, conformal | 🟠 Advanced |
| [08](./08_evaluation_and_metrics/README.md) | **Evaluation & Metrics** | MAE/RMSE/MAPE, residual diagnostics, CRPS, backtesting, stat tests | 🟡 Intermediate |
| [09](./09_anomaly_detection/README.md) | **Anomaly Detection** | Statistical, Isolation Forest, autoencoders, LSTM-AD, online AD | 🟠 Advanced |
| [10](./10_classification_and_clustering/README.md) | **Classification & Clustering** | DTW, ROCKET, deep classifiers, TS clustering | 🟠 Advanced |
| [11](./11_production_and_mlops/README.md) | **Production & MLOps** | Pipelines, feature stores, drift detection, serving, retraining | 🔴 Expert |
| [12](./12_multivariate_and_advanced_topics/README.md) | **Multivariate & Advanced** | Granger causality, PCMCI causal discovery, GNNs, diffusion models, synthetic TS | 🔴 Expert |
| [13](./13_projects_and_case_studies/README.md) | **Projects & Case Studies** | Stock, energy demand, retail, sensor anomaly — end-to-end | 🔴 Expert |

---

## 🗂️ Repository Structure

```
time-series-notes/
│
├── README.md           ← You are here
├── requirements.txt    ← Install all dependencies
│
├── 01_foundations/
├── 02_data_engineering/
├── 03_statistical_models/
├── 04_ml_for_time_series/
├── 05_deep_learning_models/
├── 06_transformer_and_foundation_models/
├── 07_forecasting_strategies/
├── 08_evaluation_and_metrics/
├── 09_anomaly_detection/
├── 10_classification_and_clustering/
├── 11_production_and_mlops/
├── 12_multivariate_and_advanced_topics/
└── 13_projects_and_case_studies/
```

Each module follows this internal layout:
```
XX_module_name/
├── README.md           ← Module index, objectives, prerequisites
├── 01_topic.md         ← Theory notes
├── 02_topic.md
├── ...
└── code/
    ├── 01_demo.py      ← Practical code
    └── 02_demo.py
```

---

## 🛠️ Libraries & Tools Used

| Category | Libraries |
|----------|-----------|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Statistical Models | `statsmodels`, `prophet` |
| ML Models | `scikit-learn`, `xgboost`, `lightgbm` |
| Deep Learning | `pytorch`, `tensorflow/keras` |
| TS Frameworks | `darts`, `sktime`, `neuralforecast`, `statsforecast` |
| Foundation Models | `chronos-forecasting`, `nixtla (TimeGPT)` |
| Hyperparameter Tuning | `optuna` |
| MLOps | `mlflow`, `evidently` |

---

## 🧭 Recommended Learning Path

```
01 Foundations
    ↓
02 Data Engineering
    ↓
03 Statistical Models
    ↓
04 ML for Time Series
    ↓
05 Deep Learning Models
    ↓
06 Transformers & Foundation Models
    ↓
07 Forecasting Strategies
    ↓
11 Production & MLOps
    ↓
12 Multivariate & Advanced
    ↓
13 Projects & Case Studies

Side tracks (use alongside any module above):
  08 Evaluation & Metrics     ← apply after every model you train
  09 Anomaly Detection        ← after Module 05
  10 Classification & Clustering ← after Module 04/05
```

---

## 📌 Prerequisites

- Python 3.9+
- Basic knowledge of NumPy and Pandas
- Familiarity with machine learning concepts (regression, cross-validation)

---

*Built for AI Engineers who want to master Time Series — from first principles to production.*
