"""Experiment helpers and reproducible result charts."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .user_based import UserBasedCF
from .item_based import ItemBasedCF
from .svd_model import SVDRecommender
from .popularity import PopularityRecommender
from .evaluate import rating_metrics, ranking_metrics


def save_bar_chart(records: list[dict], path: Path) -> None:
    """Save an RMSE comparison bar chart."""
    names, values = [x["Model"] for x in records], [x["RMSE"] for x in records]
    plt.figure(figsize=(9, 5)); plt.bar(names, values, color="#1976B8"); plt.ylabel("RMSE"); plt.title("Model comparison"); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def save_distribution_plots(ratings, figures: Path) -> None:
    """Save ratings and ratings-per-user distribution charts."""
    plt.figure(figsize=(7, 4)); ratings.rating.value_counts().sort_index().plot.bar(color="#43B6E8"); plt.xlabel("Rating"); plt.ylabel("Count"); plt.title("Rating distribution"); plt.tight_layout(); plt.savefig(figures / "rating_distribution.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 4)); ratings.groupby("user_id").size().plot.hist(bins=30, color="#1976B8"); plt.xlabel("Ratings per user"); plt.ylabel("Users"); plt.title("Ratings per user"); plt.tight_layout(); plt.savefig(figures / "ratings_per_user.png", dpi=150); plt.close()


def _line_plot(x, series: dict[str, list[float]], xlabel: str, ylabel: str, title: str, path: Path) -> None:
    """Write a consistently styled multi-series line chart."""
    plt.figure(figsize=(8, 4.8))
    for label, values in series.items(): plt.plot(x, values, marker="o", linewidth=2, label=label)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def neighbourhood_sweep(matrix, test, figures: Path) -> None:
    """Evaluate RMSE over configured user and item neighbourhood sizes."""
    ks = [5, 10, 20, 40, 80]
    user_rmse, item_rmse = [], []
    for k in ks:
        user_rmse.append(rating_metrics(UserBasedCF("cosine", k).fit(matrix), test)["RMSE"])
        item_rmse.append(rating_metrics(ItemBasedCF("adjusted_cosine", k).fit(matrix), test)["RMSE"])
    _line_plot(ks, {"User CF (cosine)": user_rmse, "Item CF (adjusted cosine)": item_rmse}, "Neighbourhood size (k)", "RMSE", "RMSE vs neighbourhood size", figures / "rmse_vs_k.png")


def similarity_comparison(records: list[dict], figures: Path) -> None:
    """Compare RMSE for the similarity metrics already evaluated in the main run."""
    selected = [record for record in records if "User-based" in record["Model"] or "Item-based" in record["Model"]]
    plt.figure(figsize=(8, 4.8)); plt.bar([r["Model"] for r in selected], [r["RMSE"] for r in selected], color=["#1976B8", "#43B6E8", "#80C9E8", "#124D75"]); plt.ylabel("RMSE"); plt.title("Similarity comparison"); plt.xticks(rotation=15, ha="right"); plt.tight_layout(); plt.savefig(figures / "similarity_comparison.png", dpi=150); plt.close()


def sparsity_analysis(matrix, test, figures: Path) -> None:
    """Retrain recommenders after randomly retaining fractions of observed training data."""
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]; rng = np.random.default_rng(42)
    observed = np.argwhere(matrix > 0); series = {"User CF": [], "Item CF": [], "SVD": []}
    for fraction in fractions:
        keep = observed[rng.choice(len(observed), size=max(1, int(len(observed) * fraction)), replace=False)]
        sparse = np.zeros_like(matrix); sparse[keep[:, 0], keep[:, 1]] = matrix[keep[:, 0], keep[:, 1]]
        series["User CF"].append(rating_metrics(UserBasedCF().fit(sparse), test)["RMSE"])
        series["Item CF"].append(rating_metrics(ItemBasedCF().fit(sparse), test)["RMSE"])
        series["SVD"].append(rating_metrics(SVDRecommender().fit(sparse), test)["RMSE"])
    _line_plot(fractions, series, "Training-data fraction", "RMSE", "RMSE vs training-data fraction", figures / "rmse_vs_sparsity.png")


def cold_start_analysis(matrix, test, figures: Path, max_users: int | None = None) -> None:
    """Compare pure CF, popularity, and a threshold-based hybrid for sparse users."""
    rng = np.random.default_rng(42); users = rng.choice(matrix.shape[0], size=max(1, int(matrix.shape[0] * 0.1)), replace=False)
    sparse = matrix.copy()
    for user in users:
        rated = np.flatnonzero(sparse[user] > 0)
        if len(rated) > 3: sparse[user, rated[3:]] = 0
    subset = test[test.user_idx.isin(users)]
    cf, pop = UserBasedCF().fit(sparse), PopularityRecommender().fit(sparse)
    class Hybrid:
        def predict(self, user, item): return pop.predict(user, item) if (sparse[user] > 0).sum() < 5 else cf.predict(user, item)
        def score_all_items(self, user): return pop.score_all_items(user) if (sparse[user] > 0).sum() < 5 else cf.score_all_items(user)
    models = {"Pure CF": cf, "Popularity fallback": pop, "Hybrid": Hybrid()}
    labels, rmse, precision = [], [], []
    for label, model in models.items():
        labels.append(label); rmse.append(rating_metrics(model, subset)["RMSE"])
        precision.append(ranking_metrics(model, sparse, subset, max_users=max_users)["Precision@10"])
    x = np.arange(len(labels)); width = 0.36
    plt.figure(figsize=(8, 4.8)); plt.bar(x - width / 2, rmse, width, label="RMSE", color="#1976B8"); plt.bar(x + width / 2, precision, width, label="Precision@10", color="#43B6E8"); plt.xticks(x, labels); plt.title("Cold-start comparison"); plt.legend(); plt.tight_layout(); plt.savefig(figures / "cold_start.png", dpi=150); plt.close()
