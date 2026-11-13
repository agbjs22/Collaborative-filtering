# 🎬 Product Recommendation System using Collaborative Filtering

> A comprehensive Data Mining Techniques (DMT) project: User-Based, Item-Based, and Matrix Factorization (SVD) Collaborative Filtering benchmarked on **MovieLens 100K** with rigorous offline evaluation, sparsity experiments, cold-start analysis, and publication-ready academic visualizations.

![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.14%2B-blue)
![Course](https://img.shields.io/badge/Course-ITB1303%20Data%20Mining%20Techniques-orange)
![Tests](https://img.shields.io/badge/Tests-7%20Passed-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-MovieLens%20100K-yellow)

---

## 1. Project Overview & Problem Statement

Modern e-commerce and streaming catalogs contain tens of thousands to millions of items. Users face acute **information overload**, and presenting the same catalog to all visitors diminishes discovery and relevance. Recommender systems mitigate this overload by predicting individual preferences from historical interaction patterns.

**Collaborative Filtering (CF)** is the cornerstone recommendation methodology because it models user preferences purely from observed behavior (ratings, co-ratings) without requiring handcrafted content attributes. 

However, real-world user-item interaction matrices suffer from extreme **data sparsity** ($> 90\%$), **cold-start bottlenecks** for new users or items, and popularity bias. This project conducts a systematic, reproducible data mining benchmark comparing:
1. **User-Based Collaborative Filtering** (Cosine & Pearson correlation)
2. **Item-Based Collaborative Filtering** (Cosine & Adjusted Cosine)
3. **Singular Value Decomposition (SVD)** Matrix Factorization (Latent factor modeling)
4. **Popularity Baseline** (Reference point & cold-start fallback)

---

## 2. Key Academic Objectives

- **Matrix Construction & Sparsity Audit**: Re-index raw interaction logs into a dense user-item matrix $R \in \mathbb{R}^{943 \times 1682}$ and quantify sparsity ($94.96\%$ in training).
- **From-Scratch Vectorized Implementations**: Implement User-Based and Item-Based CF from first principles using optimized NumPy operations, ensuring co-rated-only calculations and numerical stability.
- **Dimensionality Reduction**: Train a regularized biased SVD latent factor model using Stochastic Gradient Descent (SGD).
- **Multi-Metric Evaluation Protocol**: Evaluate rating error (RMSE, MAE) and ranking quality (Precision@K, Recall@K, F1@K, NDCG@K) on held-out test ratings with relevance threshold $r \ge 4.0$.
- **Empirical Factor Sweeps**: Quantify performance under variable neighborhood sizes ($k \in [5, 10, 20, 40, 80]$), varying dataset sparsity ($20\%$ to $100\%$), and cold-start interaction constraints ($1$ to $20$ ratings).

---

## 3. Tech Stack & Scope

- **Language**: Python 3.10+ (tested on Python 3.10, 3.11, and modern Linux environments)
- **Data Engineering**: Pandas, NumPy, SciPy
- **Algorithms**: Scikit-Learn (splitting & preprocessing), Scikit-Surprise (SVD) with pure NumPy SGD fallback
- **Visualizations**: Matplotlib (academic publication charts, multi-panel figures)
- **Testing**: Pytest

---

## 4. Mathematical Foundations

### 4.1 Similarity Formulations

1. **Cosine Similarity (Row-wise / Column-wise)**:
   $$\text{sim}(u, v) = \frac{\mathbf{r}_u \cdot \mathbf{r}_v}{\|\mathbf{r}_u\|_2 \|\mathbf{r}_v\|_2}$$
   Measures angular alignment over non-negative interaction vectors.

2. **Pearson Correlation Coefficient (Co-rated Items Only)**:
   $$\text{sim}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}$$
   Centers user ratings around their historical mean $\bar{r}_u$ to correct for individual grading bias (harsh vs. generous raters). Evaluated strictly over co-rated items $I_{uv}$.

3. **Adjusted Cosine Similarity (Item-Item)**:
   $$\text{sim}(i, j) = \frac{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)(r_{uj} - \bar{r}_u)}{\sqrt{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{u \in U_{ij}} (r_{uj} - \bar{r}_u)^2}}$$
   Subtracts the user mean $\bar{r}_u$ from item vectors before cosine calculation to eliminate cross-user scale differences.

### 4.2 Rating Prediction Formulas

- **User-Based CF**:
  $$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in \mathcal{N}_i(u)} \text{sim}(u, v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in \mathcal{N}_i(u)} |\text{sim}(u, v)|}$$
  where $\mathcal{N}_i(u)$ is the set of top-$k$ nearest neighbors to user $u$ who rated item $i$.

- **Item-Based CF**:
  $$\hat{r}_{ui} = \bar{r}_i + \frac{\sum_{j \in \mathcal{N}_u(i)} \text{sim}(i, j) \cdot (r_{uj} - \bar{r}_j)}{\sum_{j \in \mathcal{N}_u(i)} |\text{sim}(i, j)|}$$
  where $\mathcal{N}_u(i)$ is the set of top-$k$ items rated by user $u$ most similar to target item $i$.

- **Biased SVD Matrix Factorization**:
  $$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i$$
  Objective function minimized via Stochastic Gradient Descent (SGD):
  $$\min \sum_{(u, i) \in R_{\text{train}}} (r_{ui} - \hat{r}_{ui})^2 + \lambda (b_u^2 + b_i^2 + \|\mathbf{p}_u\|_2^2 + \|\mathbf{q}_i\|_2^2)$$

---

## 5. System Architecture & Workflow

```
+-------------------------------------------------------------------------+
|                          DATA SOURCE LAYER                              |
|   MovieLens 100K: 100,000 ratings | 943 users | 1,682 movies            |
+-------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------+
|                         PREPROCESSING LAYER                             |
|   ID Re-indexing -> 80/20 Deterministic Train/Test Split (Seed 42)      |
|   User-Item Matrix R (943x1682) -> Sparsity Analysis (94.96%)          |
+-------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------+
|                           MODELING LAYER                                |
|   - User-Based CF (Cosine & Pearson, k=40)                              |
|   - Item-Based CF (Cosine & Adjusted Cosine, k=40)                      |
|   - SVD Matrix Factorization (50 factors, 30 epochs, lr=0.005, reg=0.02)|
|   - Popularity Baseline (frequency + score ranking)                     |
+-------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------+
|                          EVALUATION LAYER                               |
|   Rating Error: RMSE, MAE                                               |
|   Top-N Ranking (K=10): Precision@10, Recall@10, F1@10, NDCG@10         |
|   Diagnostic Sweeps: k-sweep, Sparsity (20%-100%), Cold Start (1-20)    |
+-------------------------------------------------------------------------+
                                    │
                                    ▼
+-------------------------------------------------------------------------+
|                           OUTPUT DELIVERABLES                           |
|   results/results_main.csv, results_k_sweep.csv, results_table.md       |
|   Publication Graphs: figures/fig_11_4_6_*.png, fig_11_4_7_*.png        |
+-------------------------------------------------------------------------+
```

---

## 6. Experimental Results & Verification

All results were produced on a deterministic $80/20$ split (`random_state=42`) over MovieLens 100K.

### 6.1 Main Model Comparison Benchmark

| Model | RMSE | MAE | Precision@10 | Recall@10 | F1@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| **User-based CF (cosine)** | 0.9416 | 0.7358 | 0.0132 | 0.0120 | 0.0101 | 0.0182 |
| **User-based CF (Pearson)** | 0.9422 | 0.7376 | 0.0051 | 0.0039 | 0.0038 | 0.0066 |
| **Item-based CF (cosine)** | **0.9178** | **0.7200** | 0.0083 | 0.0052 | 0.0051 | 0.0106 |
| **Item-based CF (adjusted cosine)** | 0.9197 | 0.7204 | 0.0054 | 0.0039 | 0.0037 | 0.0064 |
| **SVD** | 0.9391 | 0.7374 | **0.0730** | **0.0526** | **0.0521** | **0.0900** |
| **Popularity baseline** | *NaN* | *NaN* | 0.1359 | 0.1372 | 0.1121 | 0.1763 |

#### Key Analytical Insights:
1. **Best Rating Predictor**: **Item-Based CF (cosine)** achieves the lowest rating error ($\text{RMSE} = 0.9178$). Item similarities are more stable than user similarities because movies accumulate more ratings over time.
2. **Best Personalized Ranking**: **SVD** achieves the highest personalized top-10 quality ($\text{NDCG@10} = 0.0900$, $\text{Precision@10} = 0.0730$), vastly outperforming neighborhood methods in surfacing relevant unobserved items.
3. **Cold-Start Fallback**: The **Popularity baseline** provides the highest raw top-10 retrieval for new users, making a hybrid strategy (popularity fallback when interactions $< 5$) optimal in production.

---

## 7. Project Directory Structure

```
Collaborative-filtering/
├── README.md                      # Academic documentation & manual
├── requirements.txt               # Pinned dependencies
├── run_all.py                     # Master execution pipeline
├── data/                          # Auto-downloaded MovieLens 100K data
│   └── ml-100k/
│       ├── u.data                 # Ratings
│       └── u.item                 # Movie metadata
├── src/                           # Modular data mining source code
│   ├── __init__.py
│   ├── data.py                    # Download, validation, reindexing, split
│   ├── similarity.py              # Cosine, Pearson, Adjusted Cosine
│   ├── user_based.py              # User-based CF model
│   ├── item_based.py              # Item-based CF model
│   ├── svd_model.py               # Biased SVD model (Surprise & SGD fallback)
│   ├── popularity.py              # Unpersonalized popularity baseline
│   ├── evaluate.py                # RMSE, MAE, Precision@K, Recall@K, F1@K, NDCG@K
│   └── experiments.py             # Sweeps, sparsity, cold-start & plotting
├── tests/                         # Automated test suite
│   ├── conftest.py
│   └── test_basic.py              # 7 test cases covering all edge cases
└── results/                       # Generated outputs and figures
    ├── results_main.csv           # Model benchmark metrics
    ├── results_k_sweep.csv        # Neighborhood sweep metrics
    ├── results_sparsity.csv       # Sparsity degradation metrics
    ├── results_cold_start.csv     # Cold-start simulation metrics
    ├── results_table.md           # Formatted markdown table
    ├── summary.md                 # Executive summary
    ├── sample_recommendations.md  # Example user recommendations
    └── figures/                   # Academic plots (170+ DPI)
        ├── fig_11_4_6_k_similarity_graphs.png
        ├── fig_11_4_7_sparsity_coldstart_graphs.png
        ├── rmse_mae_comparison.png
        ├── precision_recall_f1_comparison.png
        ├── rmse_vs_k.png
        ├── rmse_vs_sparsity.png
        └── cold_start.png
```

---

## 8. Installation & Execution Guide

### 8.1 Setup Virtual Environment

```bash
# 1. Clone repository and navigate to root
cd Collaborative-filtering

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 8.2 Run Automated Tests

Run the test suite covering matrix symmetry, bounds, edge cases, and leakage:
```bash
pytest tests/ -v
```

### 8.3 Run End-to-End Pipeline

Execute the complete academic pipeline with a single command:
```bash
python run_all.py
```

*For quick development testing (reduced sweeps), run:*
```bash
python run_all.py --quick
```

---

## 9. Academic Limitations & Future Work

- **Offline vs. Online Gap**: Results evaluate offline historical accuracy. Real user satisfaction often requires serendipity, diversity, and freshness.
- **Sparsity Bottlenecks**: Neighborhood methods degrade rapidly when interaction density falls below $97\%$.
- **Future Extensions**:
  - Incorporating implicit feedback (clicks, watch-time) alongside explicit ratings.
  - Hybrid neural collaborative filtering (NCF) and graph neural networks (LightGCN).
  - Multi-stakeholder evaluation incorporating catalog coverage and intra-list diversity.
