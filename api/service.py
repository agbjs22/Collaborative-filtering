"""Small in-memory service layer used by the FastAPI application."""
from __future__ import annotations

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
        ratings, self.movies = load_ratings(root / "data")
        indexed, self.user_map, self.item_map = reindex_ratings(ratings)
        self.inverse_items = {index: item for item, index in self.item_map.items()}
        train, _ = split_ratings(indexed)
        self.matrix, _ = build_matrix(train, len(self.user_map), len(self.item_map))
        self.models = {
            "user_cf": UserBasedCF("pearson").fit(self.matrix),
            "item_cf": ItemBasedCF("cosine").fit(self.matrix),
            "svd": SVDRecommender().fit(self.matrix),
            "popularity": PopularityRecommender().fit(self.matrix),
        }
        self.movie_by_id = self.movies.set_index("item_id").title.to_dict()

    def validate_user(self, user_id: int) -> int:
        if user_id not in self.user_map:
            raise ValueError("Unknown user ID. Choose a value from 1 to 943.")
        return self.user_map[user_id]

    def validate_model(self, model: str) -> Any:
        if model not in self.models:
            raise ValueError("Unknown model. Choose user_cf, item_cf, svd, or popularity.")
        return self.models[model]

    def movie(self, item_index: int, score: float | None = None, explanation: str | None = None) -> dict[str, Any]:
        item_id = self.inverse_items[item_index]
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
            "user_cf": "Similar viewers with a matching taste rated this highly.",
            "item_cf": "It resembles films you rated positively.",
            "svd": "It matches patterns in your viewing profile.",
            "popularity": "It is widely watched and well received.",
        }
        return [self.movie(i, scores[i], explanations[model_name]) for i in items]

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
        # Popularity is intentionally the cold-start fallback; item CF becomes useful after a few ratings.
        model = self.models["popularity"] if np.count_nonzero(profile) < 5 else ItemBasedCF("cosine").fit(np.vstack((self.matrix, profile)))
        user = 0 if model is self.models["popularity"] else self.matrix.shape[0]
        scores = model.score_all_items(user).astype(float)
        scores[profile > 0] = -np.inf
        items = np.argsort(scores)[::-1][:n]
        message = "Popular with the wider MovieLens community." if np.count_nonzero(profile) < 5 else "Similar to the films in your new profile."
        return [self.movie(i, scores[i], message) for i in items]

    def metrics(self) -> list[dict[str, Any]]:
        path = Path(__file__).resolve().parents[1] / "results" / "metrics.csv"
        return pd.read_csv(path).round(4).to_dict(orient="records") if path.exists() else []
