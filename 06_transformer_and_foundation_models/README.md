# 📘 Module 06 — Transformers & Foundation Models for Time Series

> Explore the cutting-edge frontier: transformer architectures adapted for time series, and the new generation of pre-trained foundation models enabling zero-shot forecasting.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Adapt self-attention (positional encoding, causal masking, patch tokenization) to time series
- Distinguish Informer, Autoformer, and FEDformer and their O(T log T) efficiency improvements
- Understand PatchTST's channel-independent design and implement it from scratch
- Implement DLinear and understand why simple models often beat complex transformers
- Use Amazon Chronos for zero-shot probabilistic forecasting (open weights)
- Use TimeGPT via API for zero-shot and fine-tuned forecasting
- Understand Moirai's LOTSA corpus and mixture distribution head
- Apply Lag-Llama for probabilistic forecasting using lag-feature tokenization
- Evaluate zero-shot models fairly with walk-forward CV and skill scores
- Fine-tune Chronos and apply LoRA to time series transformers
- Apply the production decision framework: zero-shot → fine-tune → train from scratch

---

## 🔗 Prerequisites

- [Module 05 — Deep Learning Models](../05_deep_learning_models/README.md)
- Familiarity with PyTorch `nn.Module`, `Dataset`, `DataLoader`
- Understanding of Transformer self-attention (covered in `01_attention_for_ts.md`)

---

## 📂 Module Contents

### Theory Notes

| File | Topic | Description |
|------|-------|-------------|
| [`01_attention_for_ts.md`](./01_attention_for_ts.md) | Self-Attention for TS | Tokenization, positional encoding, causal masking, MHA, O(T²) complexity |
| [`02_informer_autoformer_fedformer.md`](./02_informer_autoformer_fedformer.md) | Informer / Autoformer / FEDformer | ProbSparse, AutoCorrelation, frequency decomposition — O(T log T) |
| [`03_patchtst_timesnet.md`](./03_patchtst_timesnet.md) | PatchTST & TimesNet | Patch tokenization, channel independence, 2D temporal modeling, DLinear |
| [`04_timegpt_and_lag_llama.md`](./04_timegpt_and_lag_llama.md) | TimeGPT & Lag-Llama | GPT/LLaMA for TS, Nixtla API, lag-feature tokenization |
| [`05_moirai_chronos_foundation_models.md`](./05_moirai_chronos_foundation_models.md) | Moirai & Chronos | Open-source foundation models, LOTSA corpus, mixture distributions |
| [`06_zero_shot_forecasting.md`](./06_zero_shot_forecasting.md) | Zero-Shot Forecasting | When it works, fair evaluation, skill scoring, production decision framework |
| [`07_fine_tuning_ts_llms.md`](./07_fine_tuning_ts_llms.md) | Fine-Tuning | Full fine-tune Chronos, LoRA for TS transformers, TimeGPT API fine-tuning |

### 💻 Code Practicals

| File | What It Demonstrates |
|------|----------------------|
| [`code/01_informer_demo.py`](./code/01_informer_demo.py) | Informer on long-horizon hourly data; walk-forward CV; vs. N-HiTS |
| [`code/02_chronos_inference.py`](./code/02_chronos_inference.py) | Chronos zero-shot; batch inference; PI coverage evaluation |
| [`code/03_patchtst_demo.py`](./code/03_patchtst_demo.py) | PatchTST from scratch with RevIN; vs. DLinear and LSTM |
| [`code/04_zero_shot_benchmark.py`](./code/04_zero_shot_benchmark.py) | Walk-forward benchmark across 3 dataset types; decision framework |

---

## 🧠 Model Quick Reference

| Model | Org | Year | Open? | Key Feature | Complexity | Covariates |
|-------|-----|------|-------|-------------|-----------|------------|
| **Informer** | Beihang | 2021 | ✅ | ProbSparse attention | O(T log T) | ✅ |
| **Autoformer** | Tsinghua | 2021 | ✅ | AutoCorrelation + decomposition | O(T log T) | ✅ |
| **FEDformer** | USTC | 2022 | ✅ | Frequency-enhanced decomposition | O(T log T) | ✅ |
| **PatchTST** | Princeton | 2023 | ✅ | Patch tokens + channel independence | O((T/P)²) | ❌ |
| **TimesNet** | Tsinghua | 2023 | ✅ | 1D→2D temporal convolution | O(T·k) | ❌ |
| **DLinear** | UESTC | 2023 | ✅ | Decomposition linear (simple!) | O(T) | ❌ |
| **TimeGPT** | Nixtla | 2023 | ❌ | First commercial foundation model | — | ✅ |
| **Chronos** | Amazon | 2024 | ✅ | T5 + value tokenization | — | ❌ |
| **Moirai** | Salesforce | 2024 | ✅ | LOTSA + mixture distribution | — | ✅ |
| **Lag-Llama** | Mila | 2024 | ✅ | LLaMA + lag-feature tokenization | — | ❌ |

---

## 📌 Key Takeaways

1. **DLinear beats early transformers**: A simple linear model (DLinear) often outperforms Informer/Autoformer on standard benchmarks — always benchmark against simple baselines first.
2. **PatchTST is the transformer sweet spot**: Patch tokenization (P=16) reduces O(T²) to O((T/P)²) — 256× fewer attention pairs — while actually improving accuracy.
3. **Chronos is the best zero-shot starting point**: Open weights, probabilistic output, easy to use (`pip install chronos-forecasting`).
4. **Zero-shot skill score > 0.05 → deploy it**: If Chronos improves over seasonal naive by >5%, the engineering effort of training a custom model may not be worth it.
5. **LoRA > full fine-tuning for small data**: 10–100× fewer trainable parameters prevents catastrophic forgetting and overfitting.

---

## 🔑 Golden Rules for Production

```
1. Always compare zero-shot against seasonal naive FIRST
   → If zero-shot skill < 0, don't use it

2. Use walk-forward CV (≥ 3 windows) — never a single test split
   → Single splits have high variance; you need multiple measurements

3. Prefer Chronos (small) over Chronos (large) to start
   → Speed 10× faster; accuracy usually similar for short horizons

4. If zero-shot struggles: fine-tune for 50–200 steps with LoRA
   → More steps = risk of catastrophic forgetting

5. TFT > Chronos when you have strong known-future covariates
   → Foundation models don't see your promotions/holidays

6. For H > 168 steps: PatchTST or N-HiTS → better than Informer
   → ProbSparse attention doesn't scale as well as patches at very long H
```

---

## 📖 Further Reading

- [Chronos Paper (Ansari et al., 2024)](https://arxiv.org/abs/2403.07815)
- [Moirai Paper (Woo et al., 2024)](https://arxiv.org/abs/2402.02592)
- [PatchTST Paper (Nie et al., 2023)](https://arxiv.org/abs/2211.14730)
- [Are Transformers Effective? DLinear (Zeng et al., 2023)](https://arxiv.org/abs/2205.13504)
- [TimeGPT — Nixtla](https://docs.nixtla.io/)
- [Informer Paper (Zhou et al., 2021)](https://arxiv.org/abs/2012.07436)
- [Lag-Llama (Rasul et al., 2024)](https://arxiv.org/abs/2310.08278)
- [TimesNet (Wu et al., 2023)](https://arxiv.org/abs/2210.02186)

---

*← [Module 05](../05_deep_learning_models/README.md) | Back to [Master README](../README.md) | Next: [Module 07](../07_forecasting_strategies/README.md) →*
