"""Small in-memory service layer used by the FastAPI application."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import build_matrix, load_ratings, reindex_ratings, split_ratings
from src.item_based import ItemBasedCF
from src.popularity import PopularityRecommender
from src.svd_model import SVDRecommender
from src.user_based import UserBasedCF


class RecommendationService:
    """Load the dataset once and expose safe, presentation-ready recommendations."""

    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.root = root
        self.ratings, self.movies = load_ratings(root / "data")
        indexed, self.user_map, self.item_map = reindex_ratings(self.ratings)
        self.inverse_items = {index: item for item, index in self.item_map.items()}
        train, _ = split_ratings(indexed)
        self.matrix, _ = build_matrix(train, len(self.user_map), len(self.item_map))
        
        # Fit models for live evaluation & recommendations
        self.models = {
            "svd": SVDRecommender().fit(self.matrix),
            "item_cf": ItemBasedCF("cosine").fit(self.matrix),
            "item_cf_cosine": ItemBasedCF("cosine").fit(self.matrix),
            "item_cf_adj": ItemBasedCF("adjusted_cosine").fit(self.matrix),
            "user_cf": UserBasedCF("pearson").fit(self.matrix),
            "user_cf_pearson": UserBasedCF("pearson").fit(self.matrix),
            "user_cf_cosine": UserBasedCF("cosine").fit(self.matrix),
            "popularity": PopularityRecommender().fit(self.matrix),
        }
        self.movie_by_id = self.movies.set_index("item_id").title.to_dict()

    def validate_user(self, user_id: int) -> int:
        if user_id not in self.user_map:
            raise ValueError("Unknown user ID. Choose a value from 1 to 943.")
        return self.user_map[user_id]

    def validate_model(self, model: str) -> Any:
        if model not in self.models:
            raise ValueError(f"Unknown model '{model}'. Choose from: {list(self.models.keys())}")
        return self.models[model]

    def movie(self, item_index: int, score: float | None = None, explanation: str | None = None) -> dict[str, Any]:
        item_id = int(self.inverse_items[item_index])
        result: dict[str, Any] = {"item_id": item_id, "title": self.movie_by_id.get(item_id, "Unknown movie")}
        if score is not None:
            result["score"] = round(float(score), 2)
        if explanation:
            result["explanation"] = explanation
        return result

    def history(self, user_id: int) -> list[dict[str, Any]]:
        user = self.validate_user(user_id)
        rated = np.flatnonzero(self.matrix[user])
        ordered = rated[np.argsort(self.matrix[user, rated])[::-1]][:8]
        return [self.movie(i, self.matrix[user, i]) for i in ordered]

    def recommend(self, user_id: int, model_name: str, n: int) -> list[dict[str, Any]]:
        user = self.validate_user(user_id)
        model = self.validate_model(model_name)
        scores = model.score_all_items(user).astype(float)
        scores[self.matrix[user] > 0] = -np.inf
        items = np.argsort(scores)[::-1][:n]
        
        explanations = {
            "svd": "Latent factor decomposition matched your implicit viewing preferences.",
            "item_cf": "Resembles high-rated films from your history (Item Cosine CF).",
            "item_cf_cosine": "Resembles high-rated films from your history (Item Cosine CF).",
            "item_cf_adj": "Matches co-rated preferences with user-mean centering (Adjusted Cosine).",
            "user_cf": "Similar viewers with matching taste rated this highly (Pearson User CF).",
            "user_cf_pearson": "Similar viewers with matching taste rated this highly (Pearson User CF).",
            "user_cf_cosine": "Viewers with similar raw rating profiles watched this (Cosine User CF).",
            "popularity": "High community engagement and strong mean rating across MovieLens.",
        }
        exp = explanations.get(model_name, "Recommended by collaborative filtering.")
        return [self.movie(i, scores[i], exp) for i in items]

    def search_movies(self, query: str, limit: int) -> list[dict[str, Any]]:
        table = self.movies
        if query:
            table = table[table.title.str.contains(query, case=False, na=False, regex=False)]
        return [{"item_id": int(row.item_id), "title": row.title} for row in table.head(limit).itertuples()]

    def new_user(self, ratings: list[dict[str, float]], model_name: str, n: int) -> list[dict[str, Any]]:
        self.validate_model(model_name)
        profile = np.zeros(self.matrix.shape[1], dtype=float)
        for rating in ratings:
            item_id = int(rating["item_id"])
            if item_id in self.item_map:
                profile[self.item_map[item_id]] = float(np.clip(rating["rating"], 1, 5))
        
        is_sparse = np.count_nonzero(profile) < 5
        model = self.models["popularity"] if is_sparse else ItemBasedCF("cosine").fit(np.vstack((self.matrix, profile)))
        user = 0 if is_sparse else self.matrix.shape[0]
        scores = model.score_all_items(user).astype(float)
        scores[profile > 0] = -np.inf
        items = np.argsort(scores)[::-1][:n]
        message = "Popular fallback (cold-start bootstrap < 5 ratings)." if is_sparse else "Personalized to your rated taste profile."
        return [self.movie(i, scores[i], message) for i in items]

    def metrics(self) -> list[dict[str, Any]]:
        path = self.root / "results" / "metrics.csv"
        if not path.exists():
            path = self.root / "results" / "results_main.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path).round(4)
        return df.replace({np.nan: None}).to_dict(orient="records")

    def analytics_summary(self) -> dict[str, Any]:
        """Aggregate real dataset properties and distributions."""
        n_users = int(self.ratings["user_id"].nunique())
        n_items = int(self.ratings["item_id"].nunique())
        total_ratings = int(len(self.ratings))
        avg_rating = round(float(self.ratings["rating"].mean()), 2)
        full_sparsity = round(float(1.0 - total_ratings / (n_users * n_items)) * 100.0, 2)
        train_sparsity = round(float(1.0 - np.count_nonzero(self.matrix) / self.matrix.size) * 100.0, 2)

        # Rating distribution
        vc = self.ratings["rating"].value_counts().sort_index()
        dist = [{"rating": int(r), "count": int(c), "percentage": round(c / total_ratings * 100.0, 2)} for r, c in vc.items()]

        # Most-rated movies top 10
        counts = self.ratings.groupby("item_id").size().rename("count")
        means = self.ratings.groupby("item_id")["rating"].mean().round(2).rename("avg_rating")
        top_df = pd.concat([counts, means], axis=1).sort_values(by="count", ascending=False).head(10).reset_index()
        top_df["title"] = top_df["item_id"].map(self.movie_by_id).fillna("Unknown movie")
        most_rated = top_df[["item_id", "title", "count", "avg_rating"]].to_dict(orient="records")

        return {
            "total_users": n_users,
            "total_movies": n_items,
            "total_ratings": total_ratings,
            "average_rating": avg_rating,
            "full_sparsity_pct": full_sparsity,
            "train_sparsity_pct": train_sparsity,
            "sparsity_formula": "1.0 - (total_ratings / (total_users * total_movies))",
            "rating_distribution": dist,
            "most_rated_movies": most_rated,
        }

    def export_ratings_csv(self) -> str:
        """Export raw ratings to CSV string."""
        buf = io.StringIO()
        self.ratings[["user_id", "item_id", "rating", "timestamp"]].to_csv(buf, index=False)
        return buf.getvalue()

    def export_recommendations_csv(self, user_id: int, model_name: str, n: int) -> str:
        """Export current recommendation list to CSV."""
        recs = self.recommend(user_id, model_name, n)
        df = pd.DataFrame([
            {
                "rank": idx + 1,
                "item_id": r["item_id"],
                "title": r["title"],
                "predicted_score": r.get("score"),
                "explanation": r.get("explanation", ""),
            }
            for idx, r in enumerate(recs)
        ])
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue()

    def export_metrics_csv(self) -> str:
        """Export benchmark evaluation metrics."""
        path = self.root / "results" / "results_main.csv"
        if not path.exists():
            path = self.root / "results" / "metrics.csv"
        return path.read_text(encoding="utf-8") if path.exists() else "Model,RMSE,MAE,Precision@10,Recall@10,F1@10,NDCG@10\n"

    def export_analytics_csv(self) -> str:
        """Export aggregated analytics summary."""
        stats = self.analytics_summary()
        buf = io.StringIO()
        buf.write("=== DATASET SUMMARY ===\n")
        buf.write(f"Total Users,{stats['total_users']}\n")
        buf.write(f"Total Movies,{stats['total_movies']}\n")
        buf.write(f"Total Ratings,{stats['total_ratings']}\n")
        buf.write(f"Average Rating,{stats['average_rating']}\n")
        buf.write(f"Full Dataset Sparsity (%),{stats['full_sparsity_pct']}\n")
        buf.write(f"Training Sparsity (%),{stats['train_sparsity_pct']}\n")
        buf.write(f"Sparsity Formula,\"{stats['sparsity_formula']}\"\n\n")

        buf.write("=== RATING DISTRIBUTION ===\n")
        pd.DataFrame(stats["rating_distribution"]).to_csv(buf, index=False)
        buf.write("\n=== TOP 10 MOST-RATED MOVIES ===\n")
        pd.DataFrame(stats["most_rated_movies"]).to_csv(buf, index=False)
        return buf.getvalue()
