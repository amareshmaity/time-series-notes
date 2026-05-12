# 📘 Module 05 — Deep Learning Models for Time Series

> **Level**: 🟠 Advanced | **Prerequisites**: [Module 02](../02_data_engineering/README.md) (windowing/features), [Module 04](../04_ml_for_time_series/README.md) (ML fundamentals), PyTorch basics
>
> Deep learning unlocks long-range dependency modeling, multi-variate attention, and probabilistic forecasting at scale. This module covers the full DL stack — RNN/LSTM, TCN, Seq2Seq, N-BEATS, and TFT — with working implementations.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Understand the vanishing gradient problem and why LSTM/GRU solve it
- Build sliding-window `Dataset` classes for supervised DL forecasting
- Implement TCN with dilated causal convolutions for long-range patterns
- Design Seq2Seq encoder-decoder architectures with attention
- Understand the N-BEATS doubly residual stacking architecture
- Configure and train the Temporal Fusion Transformer (TFT) with all covariate types
- Build the `create_future_df` scaffold for TFT inference with known-future covariates
- Apply training best practices: gradient clipping, LR scheduling, early stopping
- Write reusable `train_epoch` / `evaluate` training utilities

---

## 🔗 Prerequisites

- [Module 02 — Data Engineering](../02_data_engineering/README.md) — sliding window datasets
- [Module 04 — ML for TS](../04_ml_for_time_series/README.md) — feature engineering, CV
- Python: PyTorch, numpy; optional: neuralforecast, pytorch-forecasting

---

## 📂 Module Contents

### 📒 Theory Notes

| File | Topic | Description |
|------|-------|-------------|
| [`01_rnn_lstm_gru.md`](./01_rnn_lstm_gru.md) | RNN, LSTM, GRU | Vanishing gradient, LSTM gates, GRU simplification, implementation |
| [`02_temporal_convolutional_networks.md`](./02_temporal_convolutional_networks.md) | TCN | Dilated causal convolutions, receptive field, WaveNet connection |
| [`03_seq2seq_and_attention.md`](./03_seq2seq_and_attention.md) | Seq2Seq + Attention | Encoder-decoder, Bahdanau attention, teacher forcing, scheduled sampling |
| [`04_nbeats_and_nhits.md`](./04_nbeats_and_nhits.md) | N-BEATS & N-HiTS | Doubly residual stacking, interpretable decomposition, N-HiTS multi-rate hierarchy |
| [`05_temporal_fusion_transformer.md`](./05_temporal_fusion_transformer.md) | TFT | VSN, GRN, LSTM encoder/decoder, attention heatmap, `create_future_df` helper |
| [`06_training_best_practices.md`](./06_training_best_practices.md) | Training Best Practices | RevIN, Huber loss, gradient clipping, LR scheduling, `train_epoch`/`evaluate` utilities |

### 💻 Code Practicals

| File | What It Demonstrates |
|------|----------------------|
| [`code/01_lstm_gru_demo.py`](./code/01_lstm_gru_demo.py) | LSTM & GRU from scratch in PyTorch — sliding window, training loop, forecast |
| [`code/02_tcn_demo.py`](./code/02_tcn_demo.py) | TCN with dilated causal convolutions — build, train, compare with LSTM |
| [`code/03_nbeats_demo.py`](./code/03_nbeats_demo.py) | N-BEATS using neuralforecast — interpretable trend/seasonal decomposition |
| [`code/04_tft_demo.py`](./code/04_tft_demo.py) | TFT using neuralforecast — covariates, attention weights, prediction intervals |

---

## 🗺️ Learning Path

```
01_rnn_lstm_gru.md              ← Sequential foundations
        ↓
02_temporal_convolutional_networks.md
        ↓
03_seq2seq_and_attention.md
        ↓
04_nbeats_and_nhits.md          ← Modern pure-DL forecasters
        ↓
05_temporal_fusion_transformer.md  ← State-of-the-art
        ↓
06_training_best_practices.md
        ↓
code/01 → 02 → 03 → 04
```

---

## 🧠 DL Model Architecture Overview

| Model | Temporal Mechanism | Multi-step | Interpretable | Covariates | Best For |
|-------|-------------------|-----------|--------------|-----------|---------|
| **LSTM** | Hidden state (recurrent) | Via Seq2Seq | ❌ | ✅ | Short-to-medium sequences |
| **GRU** | Hidden state (recurrent) | Via Seq2Seq | ❌ | ✅ | Faster LSTM alternative |
| **TCN** | Dilated causal conv | Direct | ❌ | ✅ | Long-range parallel training |
| **Seq2Seq** | LSTM + decoder | Native | ❌ | ✅ | Multi-step structured forecast |
| **N-BEATS** | FC blocks + residuals | Native | ✅ (generic) | ❌ | Pure univariate, interpretable |
| **N-HiTS** | Multi-rate sampling | Native | ✅ | ❌ (v1) | Long-horizon efficiency |
| **TFT** | LSTM + Multi-head attn | Native | ✅ | ✅ | Production, covariates, intervals |

---

## 📦 Required Libraries

```bash
pip install torch torchvision torchaudio          # PyTorch
pip install neuralforecast                        # N-BEATS, N-HiTS, TFT (high-level)
pip install pytorch-forecasting                   # TFT (detailed)
pip install lightning                             # PyTorch Lightning (training)
pip install matplotlib seaborn pandas numpy
```

---

## 📖 Further Reading

- [Hochreiter & Schmidhuber (1997) — Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Bai et al. (2018) — TCN: An Empirical Evaluation of Generic Convolutional Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)
- [Oreshkin et al. (2020) — N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting](https://arxiv.org/abs/1905.10437)
- [Lim et al. (2021) — Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)
- [Neuralforecast Documentation](https://nixtlaverse.nixtla.io/neuralforecast/)

---

*← [Module 04 — ML for TS](../04_ml_for_time_series/README.md) | Back to [Master README](../README.md) | Next: [Module 06 — Transformers & Foundation Models](../06_transformer_and_foundation_models/README.md) →*
