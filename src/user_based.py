"""User-based collaborative filtering implemented with NumPy."""
from __future__ import annotations
import numpy as np
from .similarity import cosine_similarity, pearson_similarity


class UserBasedCF:
    """Mean-centred user-neighbourhood recommender."""
    def __init__(self, metric: str = "cosine", k: int = 40): self.metric, self.k = metric, k

    def fit(self, matrix: np.ndarray) -> "UserBasedCF":
        self.matrix, self.mask = matrix, matrix > 0
        counts = self.mask.sum(axis=1)
        self.global_mean = float(matrix[self.mask].mean())
        self.means = np.divide(matrix.sum(axis=1), counts, out=np.full(matrix.shape[0], self.global_mean), where=counts > 0)
        self.deviation = np.where(self.mask, matrix - self.means[:, None], 0.0)
        self.similarity = cosine_similarity(matrix) if self.metric == "cosine" else pearson_similarity(matrix)
        return self

    def predict(self, user: int, item: int) -> float:
        """Predict a user-item rating using the strongest eligible neighbours."""
        if user >= self.matrix.shape[0] or item >= self.matrix.shape[1]: return self.global_mean
        neighbours = np.flatnonzero(self.mask[:, item])
        if not len(neighbours): return float(self.means[user])
        order = neighbours[np.argsort(self.similarity[user, neighbours])[::-1][:self.k]]
        weights = self.similarity[user, order]
        denom = np.abs(weights).sum()
        value = self.means[user] if denom == 0 else self.means[user] + np.dot(weights, self.deviation[order, item]) / denom
        return float(np.clip(value, 1, 5))

    def score_all_items(self, user: int) -> np.ndarray:
        """Score all items efficiently with the top global user neighbours."""
        neighbours = np.argsort(self.similarity[user])[::-1][: max(self.k * 2, self.k)]
        weights = self.similarity[user, neighbours]
        numerator = weights @ self.deviation[neighbours]
        denominator = np.abs(weights) @ self.mask[neighbours]
        scores = self.means[user] + np.divide(numerator, denominator, out=np.zeros(self.matrix.shape[1]), where=denominator > 0)
        scores[denominator == 0] = self.means[user]
        return np.clip(scores, 1, 5)
