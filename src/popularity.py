"""Popularity recommendation baseline and cold-start fallback."""
from __future__ import annotations
import numpy as np


class PopularityRecommender:
    """Rank items by rating count and then mean rating."""
    def fit(self, matrix: np.ndarray) -> "PopularityRecommender":
        self.matrix = matrix
        mask = matrix > 0
        self.global_mean = float(matrix[mask].mean())
        self.counts = mask.sum(axis=0)
        self.means = np.divide(matrix.sum(axis=0), self.counts, out=np.full(matrix.shape[1], self.global_mean), where=self.counts > 0)
        self.scores = self.counts + self.means / 10.0
        return self

    def predict(self, user: int, item: int) -> float:
        """Return the item mean as a rating prediction."""
        return float(np.clip(self.means[item], 1, 5))

    def score_all_items(self, user: int) -> np.ndarray:
        """Return popularity ranking scores for every item."""
        return self.scores.astype(float, copy=True)
