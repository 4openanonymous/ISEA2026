# W1 Stability Report

## Environment
| Library | Version |
|---|---|
| python | 3.13.5 |
| numpy | 2.1.3 |
| sklearn | 1.6.1 |
| torch | 2.8.0 |
| transformers | 4.57.0 |
| tokenizers | 0.22.1 |
| sbert | 5.1.1 |

## Dataset
- Input: `/Users/mmsssskk/Desktop/Project/ISEA/data/clean/ncaf_2021.csv`
- Size: 125
- Text column: `text`

## Model & Settings
- Model: `paraphrase-multilingual-MiniLM-L12-v2` (Dim: 384)
- Mode: **KNN**
- kNN neighbors: 10

## Stability Metrics
### 1. Embedding Reproducibility
| Metric | Description | Value |
|---|---|---|
| Cosine Repro. | Mean pairwise cosine between repeated encodings | **1.000** |
### 2. $\hat{d}$ Reproducibility (Across Re-embeddings)
| Metric | Correlation Type | Value |
|---|---|---|
| $\hat{d}$ Spearman $ho$ | Rank-order consistency | **1.000** |
| $\hat{d}$ Pearson $r$ | Linear correlation | **1.000** |

## Outputs
- Primary data: `clusters_knn.csv` (contains $\hat{d}_{knn}$)
- Visualization: `pca_latent_knn.png` (colored by $\hat{d}_{knn}$)

## Notes and Critical Interpretation
- Reference thresholds: Cosine $\ge 0.95$; Spearman $ho \ge 0.90$（可依語料調整）