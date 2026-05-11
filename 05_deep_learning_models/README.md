# 📘 Module 05 — Deep Learning Models for Time Series

> Dive into the architecture and intuition behind RNNs, LSTMs, TCNs, and modern architectures like N-BEATS and Temporal Fusion Transformer — the deep learning toolkit for time series.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Understand RNN and LSTM architectures and their role in sequence modeling
- Implement GRU as a lightweight alternative to LSTM
- Build Seq2Seq encoder-decoder models for multi-step forecasting
- Apply Temporal Convolutional Networks (TCN) as a parallelizable RNN alternative
- Use N-BEATS and N-HiTS for interpretable neural forecasting
- Implement Temporal Fusion Transformer (TFT) for multi-horizon, multi-variate forecasting
- Understand PatchTST and TimesNet as modern patch-based architectures

---

## 🔗 Prerequisites

- [Module 03 — Statistical Models](../03_statistical_models/README.md)
- [Module 04 — ML for Time Series](../04_ml_for_time_series/README.md)
- Basic deep learning (backpropagation, PyTorch or Keras)

---

## 📂 Module Contents

### Theory Notes
| File | Topic |
|------|-------|
| `01_rnn_and_lstm_basics.md` | Vanishing gradients, LSTM gates (forget/input/output), hidden state |
| `02_gru_architecture.md` | GRU vs LSTM, reset/update gates, when to prefer GRU |
| `03_seq2seq_encoder_decoder.md` | Encoder-decoder framework, teacher forcing, attention mechanism |
| `04_temporal_convolutional_networks.md` | Causal convolutions, dilated convolutions, residual blocks |
| `05_nbeats_and_nhits.md` | Block architecture, basis expansion, interpretable decomposition |
| `06_tft_temporal_fusion_transformer.md` | Variable selection, LSTM encoder, multi-head attention, quantile outputs |
| `07_patchtst_timesnet.md` | Patch-based tokenization, channel independence vs mixing |

### Code Practicals
| File | What It Demonstrates |
|------|----------------------|
| `code/01_lstm_forecasting.py` | LSTM forecasting in PyTorch with sliding window dataset |
| `code/02_seq2seq_ts.py` | Encoder-decoder model with teacher forcing |
| `code/03_tcn_ts.py` | TCN implementation with dilated causal convolutions |
| `code/04_tft_demo.py` | TFT using PyTorch Forecasting / neuralforecast |
| `code/05_nbeats_demo.py` | N-BEATS using neuralforecast with interpretable stacks |

---

## 🧠 Key Concepts

| Architecture | Key Idea | Best For |
|-------------|----------|----------|
| **LSTM** | Gated recurrent unit with memory cell | Medium-length sequences, univariate |
| **GRU** | Simplified LSTM with fewer parameters | Faster training, similar performance |
| **Seq2Seq** | Encoder compresses history, decoder predicts future | Multi-step forecasting |
| **TCN** | Dilated causal convolutions, fully parallelizable | Long sequences, fast training |
| **N-BEATS** | Pure MLP with doubly residual stacks | Interpretable, univariate |
| **TFT** | Attention + LSTM + gating + variable selection | Multi-variate, multi-horizon |
| **PatchTST** | Patches as tokens for Transformer | Long-term forecasting |

---

## 📌 Key Takeaways

1. **LSTMs** are still competitive but often beaten by simpler TCNs or MLPs on tabular TS.
2. **TFT** is the go-to architecture for production multi-variate forecasting with covariates.
3. **N-BEATS** is the gold standard for pure univariate forecasting benchmarks.
4. TCNs train faster than RNNs due to parallelism — prefer them for large datasets.
5. Use `neuralforecast` or `darts` for production-grade, well-tested DL model implementations.

---

## 📖 Further Reading

- [N-BEATS Paper (Oreshkin et al., 2020)](https://arxiv.org/abs/1905.10437)
- [TFT Paper (Lim et al., 2021)](https://arxiv.org/abs/1912.09363)
- [PatchTST Paper (Nie et al., 2023)](https://arxiv.org/abs/2211.14730)
- [neuralforecast Library](https://nixtlaverse.nixtla.io/neuralforecast/)
- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/)

---

*← [Module 04](../04_ml_for_time_series/README.md) | Back to [Master README](../README.md) | Next: [Module 06](../06_transformer_and_foundation_models/README.md) →*
