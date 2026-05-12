# Time Series Notes — Implementation Plan

## Goal

Build a comprehensive, industry-standard time series curriculum for an AI Engineer — covering theory + practical code from fundamentals to production-grade systems. Organized as a modular knowledge repository with `README.md` as the entry point for each module.

---

## Curriculum Structure

```
time-series-notes/
│
├── README.md                          ← Master index
├── requirements.txt                   ← Reproducible environment setup
│
├── 01_foundations/
│   ├── README.md
│   ├── 01_what_is_time_series.md
│   ├── 02_components_trend_seasonality.md
│   ├── 03_stationarity.md
│   ├── 04_autocorrelation_acf_pacf.md
│   ├── 05_decomposition.md
│   └── code/
│       ├── 01_basics_exploration.py
│       └── 02_decomposition_demo.py
│
├── 02_data_engineering/
│   ├── README.md
│   ├── 01_data_collection_sources.md
│   ├── 02_resampling_and_frequency.md
│   ├── 03_handling_missing_values.md
│   ├── 04_outlier_detection_and_treatment.md
│   ├── 05_feature_engineering_for_ts.md
│   ├── 06_windowing_and_rolling_features.md
│   └── code/
│       ├── 01_resampling_demo.py
│       ├── 02_missing_values.py
│       ├── 03_outlier_handling.py
│       └── 04_feature_engineering.py
│
├── 03_statistical_models/
│   ├── README.md
│   ├── 01_naive_baseline_models.md
│   ├── 02_exponential_smoothing_ETS.md
│   ├── 03_ar_ma_arma_arima_sarima.md          ← AR, MA, ARMA, ARIMA, SARIMA — full family
│   ├── 04_arimax_sarimax_exogenous.md         ← Exogenous variables, ARIMAX, SARIMAX
│   ├── 05_var_vector_autoregression.md
│   ├── 06_state_space_kalman_filters.md       ← State space models + Kalman filter deep dive
│   ├── 07_prophet.md                          ← Meta's Prophet: changepoints, seasonality, holidays
│   ├── 08_model_selection_and_diagnostics.md
│   └── code/
│       ├── 01_naive_models.py
│       ├── 02_ets_models.py
│       ├── 03_ar_ma_arma_arima_sarima.py       ← Practicals: AR → MA → ARMA → ARIMA → SARIMA
│       ├── 04_arimax_sarimax.py
│       ├── 05_var_models.py
│       ├── 06_prophet_demo.py
│       └── 07_diagnostics.py
│
├── 04_ml_for_time_series/
│   ├── README.md
│   ├── 01_ml_framing_regression_approach.md
│   ├── 02_feature_engineering_for_ml.md
│   ├── 03_xgboost_lightgbm_for_ts.md
│   ├── 04_random_forest_ts.md
│   ├── 05_cross_validation_for_ts.md
│   ├── 06_target_encoding_and_lags.md
│   └── code/
│       ├── 01_ml_framing.py
│       ├── 02_xgboost_ts.py
│       ├── 03_lightgbm_ts.py
│       └── 04_ts_cv.py
│
├── 05_deep_learning_models/
│   ├── README.md
│   ├── 01_rnn_and_lstm_basics.md
│   ├── 02_gru_architecture.md
│   ├── 03_seq2seq_encoder_decoder.md
│   ├── 04_temporal_convolutional_networks.md
│   ├── 05_nbeats_and_nhits.md
│   ├── 06_tft_temporal_fusion_transformer.md
│   ├── 07_data_augmentation_for_ts.md         ← Jittering, window warping, synthetic augmentation
│   └── code/
│       ├── 01_lstm_forecasting.py
│       ├── 02_seq2seq_ts.py
│       ├── 03_tcn_ts.py
│       ├── 04_tft_demo.py
│       └── 05_nbeats_demo.py
│
├── 06_transformer_and_foundation_models/
│   ├── README.md
│   ├── 01_attention_for_ts.md
│   ├── 02_informer_autoformer_fedformer.md
│   ├── 03_patchtst_timesnet.md               ← PatchTST & TimesNet — patch-based transformers
│   ├── 04_timegpt_and_lag_llama.md
│   ├── 05_moirai_chronos_foundation_models.md
│   ├── 06_zero_shot_forecasting.md
│   ├── 07_fine_tuning_ts_llms.md
│   └── code/
│       ├── 01_informer_demo.py
│       ├── 02_chronos_inference.py
│       ├── 03_patchtst_demo.py
│       └── 04_zero_shot_example.py
│
├── 07_forecasting_strategies/
│   ├── README.md
│   ├── 01_direct_vs_recursive_vs_MIMO.md
│   ├── 02_multi_step_forecasting.md
│   ├── 03_global_vs_local_models.md
│   ├── 04_hierarchical_forecasting.md
│   ├── 05_probabilistic_forecasting.md
│   ├── 06_conformal_prediction_for_ts.md
│   └── code/
│       ├── 01_multi_step_strategies.py
│       ├── 02_hierarchical_reconciliation.py
│       └── 03_probabilistic_forecast.py
│
├── 08_evaluation_and_metrics/
│   ├── README.md
│   ├── 01_error_metrics_MAE_RMSE_MAPE.md
│   ├── 02_skill_scores_and_relative_metrics.md
│   ├── 03_backtesting_design.md
│   ├── 04_residual_diagnostics.md             ← Ljung-Box, DW test, ACF of residuals
│   ├── 05_model_comparison_and_statistical_tests.md
│   ├── 06_calibration_for_probabilistic_models.md
│   ├── 07_crps_and_distributional_accuracy.md ← CRPS, pinball loss, WIS for probabilistic models
│   └── code/
│       ├── 01_metrics_implementation.py
│       ├── 02_backtesting_pipeline.py
│       ├── 03_statistical_tests.py
│       └── 04_crps_calibration.py
│
├── 09_anomaly_detection/
│   ├── README.md
│   ├── 01_statistical_anomaly_detection.md
│   ├── 02_isolation_forest_for_ts.md
│   ├── 03_autoencoder_anomaly_detection.md
│   ├── 04_lstm_based_anomaly_detection.md
│   ├── 05_online_anomaly_detection.md
│   ├── 06_root_cause_analysis.md
│   └── code/
│       ├── 01_statistical_methods.py
│       ├── 02_isolation_forest.py
│       ├── 03_autoencoder_ad.py
│       └── 04_online_detection.py
│
├── 10_classification_and_clustering/
│   ├── README.md
│   ├── 01_ts_classification_overview.md
│   ├── 02_distance_based_methods_DTW.md
│   ├── 03_feature_based_classification.md
│   ├── 04_deep_learning_classification.md
│   ├── 05_ts_clustering_methods.md
│   └── code/
│       ├── 01_dtw_classification.py
│       ├── 02_rocket_classifier.py
│       └── 03_ts_clustering.py
│
├── 11_production_and_mlops/
│   ├── README.md
│   ├── 01_ts_pipeline_architecture.md
│   ├── 02_feature_stores_for_ts.md
│   ├── 03_model_registry_and_versioning.md
│   ├── 04_drift_detection_and_monitoring.md
│   ├── 05_retraining_strategies.md
│   ├── 06_serving_ts_models.md
│   └── code/
│       ├── 01_pipeline_template.py
│       ├── 02_drift_detection.py
│       └── 03_serving_api.py
│
├── 12_multivariate_and_advanced_topics/
│   ├── README.md
│   ├── 01_multivariate_ts_overview.md
│   ├── 02_granger_causality.md
│   ├── 03_causal_discovery_pcmci.md           ← PCMCI, PC algorithm, causal structure learning
│   ├── 04_dynamic_time_warping_advanced.md
│   ├── 05_graph_neural_networks_for_ts.md
│   ├── 06_diffusion_models_for_ts.md
│   ├── 07_synthetic_ts_generation.md
│   └── code/
│       ├── 01_granger_causality.py
│       ├── 02_causal_discovery.py
│       ├── 03_gnn_ts.py
│       └── 04_ts_generation.py
│
└── 13_projects_and_case_studies/
    ├── README.md
    ├── 01_stock_price_forecasting.md
    ├── 02_energy_demand_forecasting.md
    ├── 03_retail_sales_forecasting.md
    ├── 04_sensor_anomaly_detection.md
    ├── 05_patient_monitoring_system.md
    └── code/
        ├── 01_stock_project/
        ├── 02_energy_project/
        └── 03_retail_project/
```

---

## Module Summary

| # | Module | Focus |
|---|--------|-------|
| 01 | Foundations | Core concepts, stationarity, ACF/PACF, decomposition |
| 02 | Data Engineering | Preprocessing, missing values, feature engineering |
| 03 | Statistical Models | AR, ARIMA, SARIMA, ARIMAX, ETS, VAR, Kalman, Prophet |
| 04 | ML for Time Series | XGBoost, LightGBM, lag features, CV strategies |
| 05 | Deep Learning | LSTM, TCN, Seq2Seq, N-BEATS, TFT, data augmentation |
| 06 | Transformers & Foundation Models | Informer, PatchTST, Chronos, TimeGPT, zero-shot, fine-tuning |
| 07 | Forecasting Strategies | Hierarchical, probabilistic, MIMO, conformal prediction |
| 08 | Evaluation & Metrics | MAE/RMSE/MAPE, residual diagnostics, CRPS, backtesting |
| 09 | Anomaly Detection | Statistical, Isolation Forest, autoencoder, LSTM-AD, online |
| 10 | Classification & Clustering | DTW, ROCKET, deep classifiers, TS clustering |
| 11 | Production & MLOps | Pipelines, feature stores, drift detection, serving, retraining |
| 12 | Multivariate & Advanced | Granger, PCMCI causal discovery, GNNs, diffusion, synthesis |
| 13 | Projects & Case Studies | Stock, energy, retail, sensor anomaly — end-to-end |

---

## Pedagogical Design

Each module README follows this pattern:
1. **Learning Objectives** — what you'll know after the module
2. **Prerequisites** — what to know before starting
3. **Theory Notes** — deep-dive markdown files
4. **Code Files** — runnable Python examples with comments
5. **Key Takeaways** — summary of important concepts
6. **Further Reading** — links to papers, blogs, libraries

---

## Libraries Covered

- `pandas`, `numpy` — data manipulation
- `statsmodels` — ARIMA, ETS, VAR
- `scikit-learn` — ML models, pipelines
- `xgboost`, `lightgbm` — gradient boosting
- `pytorch`, `tensorflow/keras` — deep learning
- `neuralforecast`, `statsforecast` — Nixtla stack
- `darts` — unified forecasting library
- `sktime` — scikit-learn compatible TS toolkit
- `prophet` — Meta's forecasting tool
- `optuna` — hyperparameter tuning
- `mlflow` — experiment tracking

