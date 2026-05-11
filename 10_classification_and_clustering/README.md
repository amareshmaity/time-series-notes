# 📘 Module 10 — Time Series Classification & Clustering

> Go beyond forecasting — learn to classify time series by shape, pattern, or label, and cluster them into meaningful groups using distance-based, feature-based, and deep learning approaches.

---

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- Frame and solve time series classification problems
- Apply Dynamic Time Warping (DTW) as a similarity measure for TS
- Use ROCKET and MiniRocket as the state-of-the-art fast classifiers
- Build deep learning classifiers (ResNet, FCN, InceptionTime)
- Cluster time series using k-Shape, k-Means with DTW, and hierarchical methods
- Evaluate clustering quality with silhouette score and domain-specific criteria

---

## 🔗 Prerequisites

- [Module 02 — Data Engineering](../02_data_engineering/README.md)
- [Module 05 — Deep Learning Models](../05_deep_learning_models/README.md)

---

## 📂 Module Contents

### Theory Notes
| File | Topic |
|------|-------|
| `01_ts_classification_overview.md` | Problem framing, benchmark datasets (UCR/UEA), evaluation |
| `02_distance_based_methods_DTW.md` | Euclidean distance, DTW, LCSS, EDR — warping constraints |
| `03_feature_based_classification.md` | tsfresh, catch22, BOSS, Shapelet Transform |
| `04_deep_learning_classification.md` | ResNet, FCN, InceptionTime, LSTM classifiers |
| `05_ts_clustering_methods.md` | k-Shape, k-Means DTW, hierarchical clustering, HDBSCAN for TS |

### Code Practicals
| File | What It Demonstrates |
|------|----------------------|
| `code/01_dtw_classification.py` | DTW distance computation, kNN-DTW classifier with tslearn |
| `code/02_rocket_classifier.py` | ROCKET and MiniRocket with sktime, benchmark vs. DTW |
| `code/03_ts_clustering.py` | k-Shape and hierarchical clustering with silhouette evaluation |

---

## 🧠 Key Concepts

- **DTW** — Dynamic Time Warping: elastic distance that allows non-linear alignment of two series.
- **ROCKET** — Random Convolutional Kernel Transform: 10k random kernels + linear classifier; state-of-the-art accuracy in milliseconds.
- **k-Shape** — Clustering algorithm using shape-based distance (SBD), correlation-based.
- **tsfresh** — Automatically extracts hundreds of statistical features from TS for classification.
- **Shapelet** — A subsequence that is maximally discriminative between classes.

---

## 📌 Key Takeaways

1. **ROCKET / MiniRocket** is the default go-to — fastest and most accurate for most datasets.
2. DTW is powerful but slow at scale — use **FastDTW** or **LB_Keogh lower bound** to prune.
3. Feature-based methods (tsfresh) are interpretable and work well with small datasets.
4. For clustering, **k-Shape** outperforms k-Means + Euclidean significantly on shaped TS.
5. Always evaluate on **UCR/UEA benchmark archives** when developing new TS classifiers.

---

## 📖 Further Reading

- [ROCKET Paper (Dempster et al., 2020)](https://arxiv.org/abs/1910.13051)
- [UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
- [sktime — TS Classification](https://www.sktime.net/en/stable/api_reference/classification.html)
- [tslearn Library](https://tslearn.readthedocs.io/en/stable/)

---

*← [Module 09](../09_anomaly_detection/README.md) | Back to [Master README](../README.md) | Next: [Module 11](../11_production_and_mlops/README.md) →*
