# 📘 Module 12 — Multivariate & Advanced Topics

> Push the boundaries — explore causality, graph neural networks, generative models for TS synthesis, and the latest research-level techniques for complex, real-world temporal systems.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Test for and interpret Granger causality between time series
- Apply Dynamic Time Warping in advanced settings (multivariate, barycenter)
- Build Graph Neural Networks (GNNs) for traffic, sensor, and spatial TS
- Understand diffusion models adapted for time series generation
- Generate synthetic time series for data augmentation and privacy preservation
- Apply changepoint detection to identify structural breaks

---

## 🔗 Prerequisites

- [Module 05 — Deep Learning Models](../05_deep_learning_models/README.md)
- [Module 07 — Forecasting Strategies](../07_forecasting_strategies/README.md)
- Graph theory basics (nodes, edges, adjacency matrix)

---

## 📂 Module Contents

### Theory Notes
| File | Topic |
|------|-------|
| `01_multivariate_ts_overview.md` | Cross-series dependencies, correlation vs. causality, VARS |
| `02_granger_causality.md` | Granger causality test, VAR-based testing, pitfalls and limitations |
| `03_dynamic_time_warping_advanced.md` | Multivariate DTW, DTW barycenter averaging, soft-DTW |
| `04_graph_neural_networks_for_ts.md` | Spatial-temporal GNNs, DCRNN, WaveNet on graphs |
| `05_diffusion_models_for_ts.md` | Score-based diffusion, TimeGrad, CSDI for TS imputation |
| `06_synthetic_ts_generation.md` | TimeGAN, CTGAN for TS, augmentation strategies |

### Code Practicals
| File | What It Demonstrates |
|------|----------------------|
| `code/01_granger_causality.py` | VAR-based Granger causality test with statsmodels |
| `code/02_gnn_ts.py` | Spatial-temporal GNN with PyTorch Geometric Temporal |
| `code/03_ts_generation.py` | Synthetic TS generation with TimeGAN |

---

## 🧠 Key Concepts

- **Granger Causality** — X Granger-causes Y if past X improves prediction of Y beyond past Y alone.
- **Spatial-Temporal GNN** — Models that capture both graph topology and temporal dynamics simultaneously.
- **DCRNN** — Diffusion Convolutional RNN: uses graph diffusion to model spatial dependencies in traffic data.
- **Diffusion Models for TS** — Denoising score matching to learn the data distribution of a TS.
- **TimeGAN** — GAN architecture with temporal dynamics: trains a generator on TS sequences.

---

## 📌 Key Takeaways

1. **Correlation ≠ Causality** — Granger causality is still associational; true causality requires intervention.
2. GNNs are essential when data has explicit **spatial or relational structure** (traffic, power grids).
3. Synthetic TS generation is increasingly important for **class imbalance** and **privacy** use cases.
4. Diffusion models are the new frontier in TS generation — outperforming GANs in fidelity.
5. Always validate synthetic data with **train-on-synthetic, test-on-real** methodology.

---

## 📖 Further Reading

- [Granger Causality — statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html)
- [DCRNN Paper (Li et al., 2018)](https://arxiv.org/abs/1707.01926)
- [TimeGAN Paper (Yoon et al., 2019)](https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html)
- [CSDI Paper — Diffusion for TS Imputation (Tashiro et al., 2021)](https://arxiv.org/abs/2107.03502)

---

*← [Module 11](../11_production_and_mlops/README.md) | Back to [Master README](../README.md) | Next: [Module 13](../13_projects_and_case_studies/README.md) →*
