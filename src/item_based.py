"""Item-based collaborative filtering implemented with NumPy."""
from __future__ import annotations
import numpy as np
from .similarity import cosine_similarity, adjusted_cosine_similarity


class ItemBasedCF:
    """Mean-centred item-neighbourhood recommender."""
    def __init__(self, metric: str = "adjusted_cosine", k: int = 40): self.metric, self.k = metric, k

    def fit(self, matrix: np.ndarray) -> "ItemBasedCF":
        self.matrix, self.mask = matrix, matrix > 0
        counts = self.mask.sum(axis=0)
        self.global_mean = float(matrix[self.mask].mean())
        self.means = np.divide(matrix.sum(axis=0), counts, out=np.full(matrix.shape[1], self.global_mean), where=counts > 0)
        self.deviation = np.where(self.mask, matrix - self.means[None, :], 0.0)
        self.similarity = cosine_similarity(matrix.T) if self.metric == "cosine" else adjusted_cosine_similarity(matrix)
        return self

    def predict(self, user: int, item: int) -> float:
        """Predict a user-item rating using similar items the user rated."""
        if user >= self.matrix.shape[0] or item >= self.matrix.shape[1]: return self.global_mean
        neighbours = np.flatnonzero(self.mask[user])
        if not len(neighbours): return float(self.means[item])
        order = neighbours[np.argsort(self.similarity[item, neighbours])[::-1][:self.k]]
        weights = self.similarity[item, order]
        denom = np.abs(weights).sum()
        value = self.means[item] if denom == 0 else self.means[item] + np.dot(weights, self.deviation[user, order]) / denom
        return float(np.clip(value, 1, 5))

    def score_all_items(self, user: int) -> np.ndarray:
        """Score all candidate items for one user through matrix multiplication."""
        rated = self.mask[user]
        numerator = self.similarity @ self.deviation[user]
        denominator = np.abs(self.similarity) @ rated
        scores = self.means + np.divide(numerator, denominator, out=np.zeros(self.matrix.shape[1]), where=denominator > 0)
        scores[denominator == 0] = self.means[denominator == 0]
        return np.clip(scores, 1, 5)
