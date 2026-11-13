"""Popularity recommendation baseline and cold-start fallback."""
from __future__ import annotations
import numpy as np


class PopularityRecommender:
    """Rank items by rating count and then mean rating.

    This is a non-personalised baseline. It does not predict individual
    ratings, so RMSE/MAE are not applicable (reported as NaN).
    """
    def fit(self, matrix: np.ndarray) -> "PopularityRecommender":
        self.matrix = matrix
        mask = matrix > 0
        self.global_mean = float(matrix[mask].mean()) if mask.any() else 3.0
        self.counts = mask.sum(axis=0)
        self.item_means = np.where(
            self.counts > 0,
            matrix.sum(axis=0) / np.maximum(self.counts, 1),
            self.global_mean,
        )
        self.scores = self.counts + self.item_means / 10.0
        return self

    def predict(self, user: int, item: int) -> float:
        """Return the item mean as a rating prediction."""
        return float(np.clip(self.item_means[item], 1, 5))

    def score_all_items(self, user: int) -> np.ndarray:
        """Return popularity ranking scores for every item."""
        return self.scores.astype(float, copy=True)
