# 📘 Module 13 — Projects & Case Studies

> Apply everything you've learned end-to-end on real-world problems. Each project goes from raw data → EDA → modeling → evaluation → deployment-ready output.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Execute a complete TS forecasting project from raw data to production artifact
- Choose the right model family and strategy for a given business problem
- Build anomaly detection pipelines on real sensor and industrial data
- Communicate results and uncertainty to non-technical stakeholders
- Evaluate multiple models head-to-head on identical backtesting setups

---

## 🔗 Prerequisites

All previous modules — this is the capstone module.

---

## 📂 Module Contents

### Project Guides
| File | Domain | Problem Type |
|------|--------|-------------|
| `01_stock_price_forecasting.md` | Finance | Forecasting + uncertainty quantification |
| `02_energy_demand_forecasting.md` | Energy | Multi-step, probabilistic, hierarchical |
| `03_retail_sales_forecasting.md` | Retail / Supply Chain | Global model, many series, hierarchical |
| `04_sensor_anomaly_detection.md` | IoT / Manufacturing | Multivariate anomaly detection, RCA |
| `05_patient_monitoring_system.md` | Healthcare | Classification + anomaly + real-time |

### Code Projects
| Folder | What's Inside |
|--------|---------------|
| `code/01_stock_project/` | Data pipeline, ARIMA + TFT + Chronos comparison, confidence intervals |
| `code/02_energy_project/` | Hierarchical forecasting, MinT reconciliation, LightGBM + N-BEATS |
| `code/03_retail_project/` | 50,000+ SKU global model, LightGBM, walk-forward backtest, MLflow tracking |

---

## 🧠 Project Workflow (Applied to Each Case Study)

```
1. Problem Definition
       ↓
2. Data Collection & EDA
       ↓
3. Preprocessing & Feature Engineering
       ↓
4. Baseline Model (Naïve / ARIMA / ETS)
       ↓
5. Advanced Models (ML / DL / Foundation)
       ↓
6. Backtesting & Model Comparison
       ↓
7. Probabilistic Forecasting & Uncertainty
       ↓
8. Production Packaging (Pipeline + API)
       ↓
9. Monitoring & Retraining Design
```

---

## 📋 Project Summaries

### 📈 Project 1 — Stock Price Forecasting
- **Dataset**: Yahoo Finance (yfinance)
- **Models**: ARIMA, LightGBM, TFT, Chronos (zero-shot)
- **Key Challenges**: Non-stationarity, regime changes, near-random-walk behavior
- **Output**: Multi-horizon forecast with prediction intervals

### ⚡ Project 2 — Energy Demand Forecasting
- **Dataset**: UCI Electricity Load / Open Power System Data
- **Models**: SARIMA, Prophet, LightGBM, N-BEATS, hierarchical reconciliation
- **Key Challenges**: Multiple seasonality (daily + weekly + yearly), holidays
- **Output**: Hourly probabilistic forecasts at country and zone levels

### 🛒 Project 3 — Retail Sales Forecasting
- **Dataset**: M5 Forecasting Competition (Walmart)
- **Models**: LightGBM global model, CrossValidation, MLflow tracking
- **Key Challenges**: Intermittent demand, promotions, 50,000+ SKUs
- **Output**: 28-day ahead point + quantile forecasts per SKU

### 🔧 Project 4 — Sensor Anomaly Detection
- **Dataset**: NASA SMAP / MSL benchmark
- **Models**: STL residuals, Isolation Forest, LSTM Autoencoder
- **Key Challenges**: Multivariate, collective anomalies, class imbalance
- **Output**: Anomaly score stream + alert pipeline

### 🏥 Project 5 — Patient Monitoring System
- **Dataset**: PhysioNet (ICU vital signs)
- **Models**: LSTM classifier, ROCKET, online anomaly detection
- **Key Challenges**: Real-time constraints, missing data, high stakes
- **Output**: Real-time alert system with confidence scoring

---

## 📌 Key Takeaways

1. Every project starts with **EDA and a naïve baseline** — never skip this step.
2. The best model is the one that performs well on **production-like backtesting**, not in-sample.
3. Communicating **uncertainty** (prediction intervals) is as important as the point forecast.
4. Real projects are 80% data wrangling, 20% modeling — plan accordingly.
5. Always package your model into a **reproducible pipeline** from day one.

---

## 📖 Dataset Sources

| Dataset | Source |
|---------|--------|
| Stock data | [yfinance](https://pypi.org/project/yfinance/) |
| Energy data | [Open Power System Data](https://open-power-system-data.org/) |
| Retail / M5 | [Kaggle M5 Competition](https://www.kaggle.com/c/m5-forecasting-accuracy) |
| Sensor anomaly | [NASA SMAP/MSL](https://github.com/khundman/telemanom) |
| ICU vitals | [PhysioNet](https://physionet.org/) |

---

*← [Module 12](../12_multivariate_and_advanced_topics/README.md) | Back to [Master README](../README.md)*
