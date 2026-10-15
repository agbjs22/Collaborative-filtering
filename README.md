# Product Recommendation System using Collaborative Filtering

Data Mining Techniques (DMT) project. This README is the full specification: build everything described here, run it, and produce the listed outputs.

## 1. Goal

Build and compare collaborative filtering (CF) recommenders on the MovieLens 100K dataset (used as the benchmark for product recommendation; the approach transfers to e-commerce data).

Models to implement and compare:

1. User-based CF (cosine and Pearson similarity), **implemented from scratch with NumPy**
2. Item-based CF (cosine and adjusted cosine similarity), **implemented from scratch with NumPy**
3. SVD matrix factorization (using the `surprise` library, with a NumPy fallback, see section 4)
4. Popularity baseline (reference point only)

Plus three extra analyses: sparsity impact, similarity-metric comparison, and cold-start handling.

## 2. Tech stack

- Python 3.10 or 3.11 (3.12+ may break `scikit-surprise`)
- pandas, numpy, scipy, scikit-learn, matplotlib, tqdm
- scikit-surprise (for SVD only)
- No deep learning libraries needed

## 3. Project structure

Create exactly this layout:

```
cf-recommender/
├── README.md
├── requirements.txt
├── run_all.py                 # runs the full pipeline end to end
├── data/                      # dataset downloaded here (git-ignored)
├── results/
│   ├── metrics.csv            # all model metrics
│   ├── results_table.md       # final comparison table
│   └── figures/               # all PNG plots
├── notebooks/
│   └── exploration.ipynb      # optional: EDA walkthrough
├── src/
│   ├── __init__.py
│   ├── data.py                # download, load, preprocess, split
│   ├── similarity.py          # cosine, pearson, adjusted cosine
│   ├── user_based.py          # user-based CF (NumPy)
│   ├── item_based.py          # item-based CF (NumPy)
│   ├── svd_model.py           # SVD via surprise (+ fallback)
│   ├── popularity.py          # popularity baseline + cold-start fallback
│   ├── evaluate.py            # RMSE, MAE, Precision@K, Recall@K, NDCG@K
│   └── experiments.py         # sparsity, similarity comparison, k sweep, cold start
└── tests/
    └── test_basic.py          # sanity tests
```

## 4. Data

- Source: `https://files.grouplens.org/datasets/movielens/ml-100k.zip`
- `data.py` must download and unzip it into `data/ml-100k/` automatically if missing.
- Use `u.data` (tab-separated: `user_id, item_id, rating, timestamp`) and `u.item` (pipe-separated, latin-1 encoding, for movie titles).
- Dataset facts to verify and print: 943 users, 1,682 items, 100,000 ratings.

### Preprocessing (`data.py`)

1. Load ratings into a pandas DataFrame; check and report missing values and duplicates.
2. Compute and print: number of users/items/ratings, **sparsity** = `1 - ratings / (users * items)`, rating distribution, ratings per user and per item (mean, median, min, max).
3. Re-index user and item IDs to contiguous 0-based integers.
4. Train/test split: **80/20 random split of ratings**, `random_state=42`. Use the same split for all models.
5. Build the train user-item matrix as a dense NumPy array (943 x 1682 is small), with `0` meaning "not rated", plus a boolean mask of observed entries.

## 5. Models

### 5.1 Similarity functions (`similarity.py`)

Vectorized NumPy, no loops over pairs. Each function takes a rating matrix and returns a full similarity matrix.

- **Cosine** over co-rated or zero-filled vectors (document which; use zero-filled for speed, mean-centered variant optional).
- **Pearson**: correlate over co-rated entries only (mean-center each row by its own mean over rated entries, then cosine on the centered matrix with masked entries).
- **Adjusted cosine** (for item-based): subtract each **user's** mean rating before computing item-item cosine.
- Set the diagonal to 0 so a user/item is never its own neighbor.

### 5.2 User-based CF (`user_based.py`)

Prediction for user `u`, item `i`:

```
r_hat(u,i) = mean_u + sum_v( sim(u,v) * (r(v,i) - mean_v) ) / sum_v( |sim(u,v)| )
```

- `v` ranges over the top-`k` most similar users **who rated item `i`**.
- Default `k = 40`. If no neighbor rated the item, fall back to the user's mean (or global mean for unknown users).
- Clip predictions to [1, 5].
- Must support similarity metric as a parameter (`"cosine"` or `"pearson"`) and `k`.

### 5.3 Item-based CF (`item_based.py`)

Mirror of user-based, over item-item similarity:

```
r_hat(u,i) = mean_i + sum_j( sim(i,j) * (r(u,j) - mean_j) ) / sum_j( |sim(i,j)| )
```

- `j` ranges over the top-`k` most similar items that user `u` rated.
- Default `k = 40`. Same fallbacks and clipping.
- Support `"cosine"` and `"adjusted_cosine"`.

### 5.4 SVD (`svd_model.py`)

- Use `surprise.SVD` (`n_factors=50`, `n_epochs=30`, `lr_all=0.005`, `reg_all=0.02`, `random_state=42`) trained on the same train split.
- If `scikit-surprise` fails to install, implement SVD matrix factorization with SGD in NumPy (user/item biases + latent factors, same hyperparameters). Expose the same interface either way.

### 5.5 Popularity baseline (`popularity.py`)

- Rank items by number of ratings, tie-break by mean rating. Used both as a baseline and as the **cold-start fallback**.

## 6. Evaluation (`evaluate.py`)

### Rating prediction
- **RMSE** and **MAE** on the test set, computed from predicted vs true ratings.

### Top-N recommendation
For `K = 10` (also report K = 5 and 20 if cheap):

- For each user with at least one relevant test item, rank all items **not in the user's training set**.
- An item is **relevant** if its test rating is >= 4.
- Compute **Precision@K**, **Recall@K**, **NDCG@K** per user, then average across users.
- Predicted scores for ranking come from each model's rating predictions (popularity uses its popularity score).

All models are evaluated on the same split and the same users.

## 7. Experiments (`experiments.py`)

1. **Main comparison**: user-based (cosine), user-based (Pearson), item-based (cosine, adjusted cosine), SVD, popularity. Report RMSE, MAE, Precision@10, Recall@10, NDCG@10.
2. **Neighbourhood size sweep**: `k` in `[5, 10, 20, 40, 80]` for user-based and item-based; plot RMSE vs `k`.
3. **Similarity comparison**: cosine vs Pearson (user-based), cosine vs adjusted cosine (item-based), at the best `k`.
4. **Sparsity analysis**: keep 20%, 40%, 60%, 80%, 100% of the training ratings (random, seed 42), retrain every model, plot RMSE vs training-data fraction. Show that SVD degrades more gracefully.
5. **Cold start**: pick 10% of users, remove all but 3 of their training ratings, and compare (a) pure CF, (b) popularity fallback, (c) hybrid (use popularity when the user has fewer than 5 ratings, else CF). Report RMSE/Precision@10 for these users.
6. **Sample recommendations**: for 3 example users, print their top-rated training movies and the top-10 recommendations with movie titles.

## 8. Outputs (required)

Save everything under `results/`:

- `metrics.csv`: one row per model/config with all metrics
- `results_table.md`: markdown table in this exact column order:

  | Model | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 |
  |---|---|---|---|---|---|
  | User-based CF (cosine) | | | | | |
  | User-based CF (Pearson) | | | | | |
  | Item-based CF | | | | | |
  | SVD | | | | | |

  (Item-based row uses the better of cosine / adjusted cosine; list the other in `metrics.csv`.)
- `figures/` (PNG, 150 dpi, labelled axes, titles, readable fonts):
  - `rating_distribution.png`
  - `ratings_per_user.png`
  - `rmse_vs_k.png`
  - `similarity_comparison.png`
  - `rmse_vs_sparsity.png`
  - `cold_start.png`
  - `model_comparison.png` (bar chart of RMSE per model)

## 9. Run instructions

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run_all.py
```

`run_all.py` should: download data, preprocess, train all models, run all experiments, write everything in section 8, and print a final summary table to the console. Total runtime should be under about 10 minutes on a normal laptop.

`requirements.txt`:
```
numpy
pandas
scipy
scikit-learn
scikit-surprise
matplotlib
tqdm
pytest
```

## 10. Code quality requirements

- Fixed random seed (42) everywhere for reproducibility.
- Docstrings on every function; short comments on the non-obvious maths.
- No hardcoded absolute paths; use `pathlib` relative to the project root.
- Vectorize with NumPy; avoid per-pair Python loops for similarity. Per-prediction loops are acceptable if the full run stays within the time limit (precompute top-k neighbors to keep it fast).
- Keep functions small and testable. No global state.

## 11. Tests (`tests/test_basic.py`)

- Similarity matrices are symmetric with zero diagonal and values in [-1, 1].
- Predictions are clipped to [1, 5].
- Train/test split has no overlapping (user, item) pairs and the sizes are 80/20.
- RMSE of a perfect predictor is 0.
- Precision/Recall/NDCG are in [0, 1].

Run with `pytest`.

## 12. Acceptance criteria

The project is done when:

1. `python run_all.py` completes without errors from a clean environment.
2. `results/results_table.md`, `results/metrics.csv` and all figures listed in section 8 exist.
3. Sanity check: SVD RMSE is roughly 0.93 to 0.95, user-based and item-based CF roughly 0.95 to 1.02, and the popularity baseline is clearly worse on ranking metrics than the CF models on the full dataset. If the numbers are far outside these ranges, debug before finishing.
4. `pytest` passes.
5. The final console output prints the comparison table and the paths of the saved results.

## 13. Notes for the report (do not skip)

After the run, also write `results/summary.md` (plain, no decoration) with 5 to 8 sentences covering: which model performed best and why, how sparsity affected each model, which similarity measure worked best, and how well the cold-start fallback helped. These findings go straight into the "Experimental Results" and "Conclusions" slides.
#   C o l l a b o r a t i v e - f i l t e r i n g  
 