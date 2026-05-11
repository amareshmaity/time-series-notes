# 📘 Module 06 — Transformers & Foundation Models for Time Series

> Explore the cutting-edge frontier: transformer architectures adapted for time series, and the new generation of pre-trained foundation models enabling zero-shot forecasting.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Understand how self-attention is adapted for temporal data
- Distinguish Informer, Autoformer, and FEDformer and their efficiency improvements
- Use TimeGPT for zero-shot and few-shot forecasting via API
- Run Amazon Chronos for probabilistic zero-shot forecasting
- Understand Moirai and Lag-Llama as open-source foundation models
- Fine-tune pre-trained time series models on domain-specific data

---

## 🔗 Prerequisites

- [Module 05 — Deep Learning Models](../05_deep_learning_models/README.md)
- Understanding of Transformer architecture (self-attention, positional encoding)

---

## 📂 Module Contents

### Theory Notes
| File | Topic |
|------|-------|
| `01_attention_for_ts.md` | Adapting self-attention to TS: positional encoding, causal masking |
| `02_informer_autoformer_fedformer.md` | ProbSparse attention, AutoCorrelation, FEDformer decomposition |
| `03_timegpt_and_lag_llama.md` | TimeGPT architecture, Lag-Llama (LLM backbone for TS) |
| `04_moirai_chronos_foundation_models.md` | Moirai (Salesforce), Chronos (Amazon) — zero-shot models |
| `05_zero_shot_forecasting.md` | Zero-shot vs. few-shot, when to use foundation models |
| `06_fine_tuning_ts_llms.md` | Fine-tuning Chronos/Moirai on custom data, LoRA for TS transformers |

### Code Practicals
| File | What It Demonstrates |
|------|----------------------|
| `code/01_informer_demo.py` | Informer model for long-sequence forecasting |
| `code/02_chronos_inference.py` | Zero-shot probabilistic forecasting with Amazon Chronos |
| `code/03_zero_shot_example.py` | TimeGPT API call, comparison with ARIMA |

---

## 🧠 Key Concepts

| Model | Organization | Key Feature |
|-------|-------------|-------------|
| **Informer** | Beihang Univ. | ProbSparse attention — O(L log L) complexity |
| **Autoformer** | Tsinghua | AutoCorrelation replacing self-attention |
| **TimeGPT** | Nixtla | First foundation model for TS, API-based |
| **Chronos** | Amazon | Tokenizes TS values, T5 backbone, probabilistic |
| **Moirai** | Salesforce | Universal TS model trained on LOTSA data |
| **Lag-Llama** | Mila | LLaMA backbone for probabilistic TS |

---

## 📌 Key Takeaways

1. Zero-shot models (Chronos, TimeGPT) can outperform ARIMA out-of-the-box on many datasets.
2. Foundation models shine on cold-start problems where historical data is scarce.
3. Fine-tuning Chronos on domain data typically yields best-of-both-worlds performance.
4. Always compare against a tuned LightGBM baseline — often faster and competitive.

---

## 📖 Further Reading

- [Chronos Paper (Ansari et al., 2024)](https://arxiv.org/abs/2403.07815)
- [Moirai Paper (Woo et al., 2024)](https://arxiv.org/abs/2402.02592)
- [TimeGPT — Nixtla](https://docs.nixtla.io/)
- [Informer Paper (Zhou et al., 2021)](https://arxiv.org/abs/2012.07436)

---

*← [Module 05](../05_deep_learning_models/README.md) | Back to [Master README](../README.md) | Next: [Module 07](../07_forecasting_strategies/README.md) →*
