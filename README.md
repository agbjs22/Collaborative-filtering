# 🎬 Product Recommendation System using Collaborative Filtering

> A full-stack data mining project: user-based, item-based and SVD collaborative filtering recommenders on **MovieLens 100K**, benchmarked end to end and served through an interactive **web application**.

![Python](https://img.shields.io/badge/Python-3.10%20|%203.11-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![React](https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript-61dafb)

**Course:** Data Mining Techniques (DMT)
**Author:** Guha Bava Jaya Sre ([@agbjs22](https://github.com/agbjs22)) · B.Tech Information Technology, K. Ramakrishnan College of Engineering

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Features](#2-key-features)
3. [Tech Stack](#3-tech-stack)
4. [Project Structure](#4-project-structure)
5. [Dataset and Preprocessing](#5-dataset-and-preprocessing)
6. [Models and Algorithms](#6-models-and-algorithms)
7. [Evaluation Methodology](#7-evaluation-methodology)
8. [Experiments](#8-experiments)
9. [Web Application](#9-web-application)
10. [REST API Reference](#10-rest-api-reference)
11. [Installation and Running](#11-installation-and-running)
12. [Docker Deployment](#12-docker-deployment)
13. [Outputs](#13-outputs)
14. [Code Quality Standards](#14-code-quality-standards)
15. [Testing and CI](#15-testing-and-ci)
16. [Acceptance Criteria](#16-acceptance-criteria)
17. [Results Summary](#17-results-summary)
18. [Limitations and Future Work](#18-limitations-and-future-work)
19. [References](#19-references)
20. [License](#20-license)

---

## 1. Overview

Recommender systems drive product discovery in e-commerce, streaming and social platforms. This project builds and rigorously compares the main **collaborative filtering (CF)** families on the MovieLens 100K benchmark (used as a stand-in for product ratings; the approach transfers directly to e-commerce data).

**Models implemented and compared**

| # | Model | Implementation |
|---|---|---|
| 1 | User-based CF (cosine, Pearson) | From scratch, NumPy |
| 2 | Item-based CF (cosine, adjusted cosine) | From scratch, NumPy |
| 3 | SVD matrix factorization | `scikit-surprise` (NumPy SGD fallback) |
| 4 | Popularity baseline | Reference point and cold-start fallback |

**Additional analyses:** neighbourhood-size sweep, similarity-metric comparison, data-sparsity impact, cold-start handling (including a hybrid strategy), and beyond-accuracy metrics (coverage, diversity, novelty).

**Web layer:** a FastAPI backend exposes the trained models; a React + TypeScript frontend lets users get recommendations, rate movies to bootstrap a new profile, see *why* an item was recommended, and explore the experiment results.

---

## 2. Key Features

### Core (data mining pipeline)
- Automatic dataset download, validation and preprocessing
- Vectorized NumPy similarity functions (cosine, Pearson, adjusted cosine), with no loops over pairs
- Top-k neighbourhood prediction with fallbacks and clipping to [1, 5]
- SVD with biases and latent factors (surprise or NumPy fallback, same interface)
- Rating-prediction metrics (RMSE, MAE) and top-N metrics (Precision@K, Recall@K, NDCG@K)
- Five experiments: main comparison, k sweep, similarity comparison, sparsity, cold start
- Auto-generated results table, metrics CSV, 150 dpi figures and a written summary

### Added on top of the base spec
- **Web application** with interactive recommendations and dashboards
- **New-user onboarding:** rate a few movies and get live recommendations (real cold-start handling)
- **Explainable recommendations:** "Because users similar to you liked…" / "Because you rated *X* highly"
- **Beyond-accuracy metrics:** catalog coverage, intra-list diversity (genre-based), novelty
- **5-fold cross-validation** with mean ± std for the main comparison (single 80/20 split kept as the headline)
- **Hyperparameter tuning:** grid search for SVD (`n_factors`, `lr_all`, `reg_all`) and best-k selection
- **Model persistence** (`joblib`) and precomputed neighbour caches for fast API responses
- **Statistical significance:** paired comparison of per-user NDCG between models
- **Genre-based filtering** and movie search in the UI
- **Dockerfile + docker-compose** for one-command deployment
- **GitHub Actions CI** running tests on every push
- Optional movie posters via the TMDB API (disabled by default)

---

## 3. Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 or 3.11 (3.12+ may break `scikit-surprise`) |
| Data / ML | pandas, numpy, scipy, scikit-learn, scikit-surprise (SVD only) |
| Visualization | matplotlib (report figures), Recharts (web charts) |
| Backend | FastAPI, Uvicorn, Pydantic, joblib |
| Frontend | React, TypeScript, Vite, Tailwind CSS |
| Testing | pytest, httpx (API tests) |
| DevOps | Docker, docker-compose, GitHub Actions |

No deep learning libraries are required.

---

## 4. Project Structure

```
cf-recommender/
├── README.md
├── requirements.txt
├── run_all.py                 # runs the full pipeline end to end
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml   # pytest on push / PR
├── data/                      # dataset downloaded here (git-ignored)
├── models/                    # persisted trained models (git-ignored)
├── results/
│   ├── metrics.csv            # all model metrics
│   ├── results_table.md       # final comparison table
│   ├── summary.md             # written findings for the report/slides
│   └── figures/               # all PNG plots
├── notebooks/
│   └── exploration.ipynb      # optional EDA walkthrough
├── src/
│   ├── __init__.py
│   ├── data.py                # download, load, preprocess, split
│   ├── similarity.py          # cosine, pearson, adjusted cosine
│   ├── user_based.py          # user-based CF (NumPy)
│   ├── item_based.py          # item-based CF (NumPy)
│   ├── svd_model.py           # SVD via surprise (+ NumPy fallback)
│   ├── popularity.py          # popularity baseline + cold-start fallback
│   ├── evaluate.py            # RMSE, MAE, Precision/Recall/NDCG@K, coverage, diversity, novelty
│   ├── experiments.py         # sparsity, similarity comparison, k sweep, cold start, CV
│   └── explain.py             # explanation generation for recommendations
├── api/
│   ├── main.py                # FastAPI app and routes
│   ├── schemas.py             # Pydantic request/response models
│   └── service.py             # loads models, serves predictions, caches neighbours
├── frontend/                  # React + TypeScript (Vite)
│   ├── src/
│   │   ├── pages/             # Recommend, Explore, Onboarding, Compare, Experiments
│   │   ├── components/
│   │   └── api.ts
│   └── package.json
└── tests/
    ├── test_basic.py          # sanity tests for the ML code
    └── test_api.py            # endpoint tests
```

---

## 5. Dataset and Preprocessing

- **Source:** <https://files.grouplens.org/datasets/movielens/ml-100k.zip>
- `data.py` downloads and unzips it into `data/ml-100k/` automatically if missing.
- Files used: `u.data` (tab-separated: `user_id, item_id, rating, timestamp`), `u.item` (pipe-separated, latin-1, for titles and genres).
- **Facts verified and printed at runtime:** 943 users, 1,682 items, 100,000 ratings.

### Preprocessing steps (`data.py`)

1. Load ratings into a DataFrame; check and report missing values and duplicates.
2. Compute and print: users / items / ratings, **sparsity** = `1 - ratings / (users × items)`, rating distribution, ratings per user and per item (mean, median, min, max).
3. Re-index user and item IDs to contiguous 0-based integers.
4. **Train/test split:** 80/20 random split of ratings, `random_state=42`. The same split is used for all models.
5. Build the train user-item matrix as a dense NumPy array (943 × 1682), with `0` meaning "not rated", plus a boolean mask of observed entries.

---

## 6. Models and Algorithms

### 6.1 Similarity functions (`similarity.py`)

Vectorized NumPy; each function takes a rating matrix and returns a full similarity matrix.

| Metric | Definition |
|---|---|
| **Cosine** | Zero-filled vectors (documented choice, chosen for speed). Mean-centered variant optional. |
| **Pearson** | Mean-center each row by its own mean over rated entries, then cosine on the centered matrix with masked entries. |
| **Adjusted cosine** | Subtract each **user's** mean rating, then item-item cosine (item-based only). |

The diagonal is set to 0 so a user or item is never its own neighbour.

### 6.2 User-based CF (`user_based.py`)

```
r_hat(u,i) = mean_u + Σ_v sim(u,v) · (r(v,i) − mean_v) / Σ_v |sim(u,v)|
```

- `v` ranges over the top-`k` most similar users **who rated item `i`**.
- Default `k = 40`. If no neighbour rated the item, fall back to the user's mean (or the global mean for unknown users).
- Predictions are clipped to [1, 5].
- Parameters: `metric` (`"cosine"` or `"pearson"`) and `k`.

### 6.3 Item-based CF (`item_based.py`)

```
r_hat(u,i) = mean_i + Σ_j sim(i,j) · (r(u,j) − mean_j) / Σ_j |sim(i,j)|
```

- `j` ranges over the top-`k` most similar items that user `u` rated.
- Default `k = 40`; same fallbacks and clipping.
- Parameters: `metric` (`"cosine"` or `"adjusted_cosine"`) and `k`.

### 6.4 SVD (`svd_model.py`)

- `surprise.SVD(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.02, random_state=42)` trained on the same train split.
- If `scikit-surprise` fails to install, a NumPy SGD implementation (user/item biases + latent factors, same hyperparameters) is used automatically behind the same interface.

### 6.5 Popularity baseline (`popularity.py`)

- Items ranked by number of ratings, tie-broken by mean rating.
- Serves as both a baseline and the **cold-start fallback**.

---

## 7. Evaluation Methodology

### Rating prediction
- **RMSE** and **MAE** on the test set.

### Top-N recommendation (K = 10; also K = 5 and 20)
- For each user with at least one relevant test item, rank all items **not in the user's training set**.
- An item is **relevant** if its test rating is ≥ 4.
- Compute **Precision@K**, **Recall@K** and **NDCG@K** per user, then average across users.
- Scores for ranking come from each model's rating predictions (popularity uses its popularity score).
- All models are evaluated on the same split and the same users.

### Beyond accuracy
- **Catalog coverage:** fraction of items that appear in at least one user's top-10.
- **Intra-list diversity:** average pairwise genre dissimilarity inside a top-10 list.
- **Novelty:** mean negative log popularity of recommended items.

### Robustness
- **5-fold cross-validation** reports mean ± std for RMSE and NDCG@10.
- **Paired significance test** on per-user NDCG@10 between the best CF model and each other model.

---

## 8. Experiments

| # | Experiment | Description |
|---|---|---|
| 1 | **Main comparison** | User-based (cosine, Pearson), item-based (cosine, adjusted cosine), SVD, popularity. RMSE, MAE, Precision@10, Recall@10, NDCG@10. |
| 2 | **Neighbourhood size sweep** | `k ∈ {5, 10, 20, 40, 80}` for user-based and item-based; plot RMSE vs `k`. |
| 3 | **Similarity comparison** | Cosine vs Pearson (user-based), cosine vs adjusted cosine (item-based), at the best `k`. |
| 4 | **Sparsity analysis** | Keep 20 / 40 / 60 / 80 / 100% of training ratings (random, seed 42), retrain every model, plot RMSE vs training fraction. Shows SVD degrading more gracefully. |
| 5 | **Cold start** | Pick 10% of users, keep only 3 of their training ratings, compare (a) pure CF, (b) popularity fallback, (c) hybrid (popularity if the user has fewer than 5 ratings, else CF). Report RMSE and Precision@10 for these users. |
| 6 | **Sample recommendations** | For 3 example users, print top-rated training movies and the top-10 recommendations with titles. |
| 7 | **Hyperparameter tuning** *(added)* | Grid search for SVD; best-`k` selection using a validation slice of train. |

---

## 9. Web Application

The app is served by FastAPI (API) and React (UI).

| Page | What it does |
|---|---|
| **Recommend** | Pick an existing user (0–942), choose a model (user-CF, item-CF, SVD, popularity) and `k`, and see top-N movies with predicted rating and an explanation. Shows the user's top-rated training movies side by side. |
| **New User Onboarding** | Search and rate 5+ movies; the app builds a temporary profile and returns live recommendations. Uses the hybrid strategy (popularity below 5 ratings, CF above). |
| **Explore Movies** | Search by title, filter by genre, view average rating, number of ratings and similar movies (item-item neighbours). |
| **Model Comparison** | Dashboard of the results table and charts: RMSE, MAE, Precision/Recall/NDCG, coverage, diversity, novelty. |
| **Experiments** | Interactive plots for RMSE vs `k`, similarity comparison, RMSE vs sparsity, and cold-start results. |
| **Dataset Insights** | Rating distribution, ratings per user/item, sparsity statistics. |

**Explainability examples**
- *User-based:* "Recommended because 12 of your 40 nearest neighbours rated this 4+."
- *Item-based:* "Because you rated *Toy Story* 5, and it is similar to this movie (sim 0.71)."
- *SVD:* "Matches your strongest latent factors (genre affinity)."

---

## 10. REST API Reference

Base URL: `http://localhost:8000` · Interactive docs: `/docs` (Swagger UI)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Service and model status |
| GET | `/api/users/{user_id}/history` | Top-rated training movies for a user |
| GET | `/api/recommend/{user_id}?model=svd&k=40&n=10` | Top-N recommendations with scores and explanations |
| POST | `/api/recommend/new-user` | Body: `{ "ratings": [{"item_id": 50, "rating": 5}, ...], "model": "svd", "n": 10 }` |
| GET | `/api/predict?user_id=1&item_id=50&model=item_cf` | Predicted rating for one pair |
| GET | `/api/movies?query=star&genre=Sci-Fi&limit=20` | Search/filter movies |
| GET | `/api/movies/{item_id}/similar?n=10` | Most similar movies (item-item) |
| GET | `/api/metrics` | Contents of `metrics.csv` as JSON |
| GET | `/api/experiments/{name}` | Experiment results (`k_sweep`, `similarity`, `sparsity`, `cold_start`) |

---

## 11. Installation and Running

### Prerequisites
- Python 3.10 or 3.11
- Node.js 18+ (for the frontend)

### 1. Run the ML pipeline

```bash
git clone https://github.com/agbjs22/cf-recommender.git
cd cf-recommender

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python run_all.py
```

`run_all.py` downloads data, preprocesses, trains all models, runs all experiments, writes everything listed in [Outputs](#13-outputs), saves trained models to `models/`, and prints a final summary table. Expected runtime: **under ~10 minutes** on a normal laptop.

### 2. Start the backend

```bash
uvicorn api.main:app --reload --port 8000
```

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

### `requirements.txt`

```
numpy
pandas
scipy
scikit-learn
scikit-surprise
matplotlib
tqdm
pytest
fastapi
uvicorn[standard]
pydantic
joblib
httpx
```

> **Troubleshooting:** if `scikit-surprise` fails to build (common on Python 3.12+ or Windows without C++ build tools), the project automatically falls back to the NumPy SVD. You can also install a C++ toolchain or use Python 3.10/3.11.

---

## 12. Docker Deployment

```bash
docker compose up --build
```

- Backend: <http://localhost:8000>
- Frontend: <http://localhost:5173>

The image runs `run_all.py` on first start if `models/` is empty, then launches the API.

---

## 13. Outputs

Everything is written under `results/`:

- **`metrics.csv`**: one row per model/config with all metrics.
- **`results_table.md`**: markdown table in this exact column order:

  | Model | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 |
  |---|---|---|---|---|---|
  | User-based CF (cosine) | | | | | |
  | User-based CF (Pearson) | | | | | |
  | Item-based CF | | | | | |
  | SVD | | | | | |

  (The item-based row uses the better of cosine / adjusted cosine; the other is listed in `metrics.csv`.)
- **`summary.md`**: plain-text findings (5 to 8 sentences) for the report and slides.
- **`figures/`** (PNG, 150 dpi, labelled axes, titles, readable fonts):
  - `rating_distribution.png`
  - `ratings_per_user.png`
  - `rmse_vs_k.png`
  - `similarity_comparison.png`
  - `rmse_vs_sparsity.png`
  - `cold_start.png`
  - `model_comparison.png` (bar chart of RMSE per model)

---

## 14. Code Quality Standards

- Fixed random seed (**42**) everywhere for reproducibility.
- Docstrings on every function; short comments on non-obvious maths.
- No hardcoded absolute paths; `pathlib` relative to the project root.
- Vectorized NumPy; no per-pair Python loops for similarity. Per-prediction loops are fine if the run stays within the time limit (top-k neighbours are precomputed).
- Small, testable functions. No global state.
- Type hints on public functions; `ruff`/`black` formatting.

---

## 15. Testing and CI

Run locally:

```bash
pytest
```

**`tests/test_basic.py`**
- Similarity matrices are symmetric with zero diagonal and values in [-1, 1].
- Predictions are clipped to [1, 5].
- Train/test split has no overlapping (user, item) pairs and sizes are 80/20.
- RMSE of a perfect predictor is 0.
- Precision / Recall / NDCG are in [0, 1].

**`tests/test_api.py`**
- `/api/health` returns 200.
- Recommendations never include items already rated by the user.
- Invalid user IDs and model names return 4xx with a clear message.
- New-user endpoint works with 1 rating (popularity fallback) and 10 ratings (CF).

**CI:** `.github/workflows/ci.yml` installs dependencies and runs `pytest` on every push and pull request.

---

## 16. Acceptance Criteria

The project is done when:

1. `python run_all.py` completes without errors from a clean environment.
2. `results/results_table.md`, `results/metrics.csv`, `results/summary.md` and all listed figures exist.
3. **Sanity ranges:** SVD RMSE roughly **0.93 to 0.95**; user-based and item-based CF roughly **0.95 to 1.02**; the popularity baseline is clearly worse on ranking metrics than the CF models on the full dataset. If numbers are far outside these ranges, debug before finishing.
4. `pytest` passes.
5. The final console output prints the comparison table and the paths of saved results.
6. The web app starts, `/docs` loads, and all six pages render with real data.

---

## 17. Results Summary

> Fill this table from `results/results_table.md` after running `run_all.py`.

| Model | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|---|---|
| User-based CF (cosine) | – | – | – | – | – |
| User-based CF (Pearson) | – | – | – | – | – |
| Item-based CF | – | – | – | – | – |
| SVD | – | – | – | – | – |
| Popularity (baseline) | – | – | – | – | – |

`results/summary.md` covers which model performed best and why, how sparsity affected each model, which similarity measure worked best, and how well the cold-start fallback helped. These findings feed the "Experimental Results" and "Conclusions" slides.

---

## 18. Limitations and Future Work

**Limitations**
- MovieLens 100K is small and movie-specific; absolute numbers will differ on real e-commerce data.
- Random rating split ignores time; a temporal split would be more realistic.
- Dense matrices work at this scale but not for millions of users.
- Pure CF cannot recommend brand-new items (item cold start).

**Future work**
- Hybrid models using item content (genres, descriptions)
- Implicit-feedback methods (ALS, BPR) and sparse-matrix implementations
- Temporal splits and time-aware models
- A/B testing simulation and online learning
- Larger datasets (MovieLens 1M/25M, Amazon reviews)

---

## 19. References

1. Sarwar, B., Karypis, G., Konstan, J., Riedl, J. (2001). *Item-based Collaborative Filtering Recommendation Algorithms.* WWW '01.
2. Koren, Y., Bell, R., Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems.* IEEE Computer.
3. Herlocker, J. et al. (1999). *An Algorithmic Framework for Performing Collaborative Filtering.* SIGIR '99.
4. Harper, F. M., Konstan, J. A. (2015). *The MovieLens Datasets: History and Context.* ACM TiiS.
5. Hug, N. (2020). *Surprise: A Python library for recommender systems.* JOSS.

---

## 20. License

Released under the [MIT License](LICENSE). The MovieLens dataset is provided by [GroupLens](https://grouplens.org/datasets/movielens/) and is licensed for research use; do not redistribute it with this repository.
