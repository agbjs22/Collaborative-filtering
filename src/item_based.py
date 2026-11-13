"""Item-based collaborative filtering implemented with NumPy."""
from __future__ import annotations
import numpy as np
from .similarity import cosine_sim, adjusted_cosine_sim


class ItemBasedCF:
    """Mean-centred item-neighbourhood recommender."""
    def __init__(self, metric: str = "adjusted_cosine", k: int = 40):
        self.metric, self.k = metric, k

    def fit(self, matrix: np.ndarray) -> "ItemBasedCF":
        self.matrix = matrix
        self.mask = matrix > 0
        self.global_mean = float(matrix[self.mask].mean()) if self.mask.any() else 3.0
        counts = self.mask.sum(axis=0)
        user_counts = self.mask.sum(axis=1)
        self.user_means = np.where(user_counts > 0, matrix.sum(axis=1) / np.maximum(user_counts, 1), self.global_mean)
        self.item_means = np.where(counts > 0, matrix.sum(axis=0) / np.maximum(counts, 1), self.global_mean)
        if self.metric == "cosine":
            self.similarity = cosine_sim(matrix.T)
        else:
            self.similarity = adjusted_cosine_sim(matrix, self.mask, self.user_means)
        return self

    def predict(self, user: int, item: int) -> float:
        """Predict a user-item rating using similar items the user rated."""
        if user >= self.matrix.shape[0] or item >= self.matrix.shape[1]:
            return self.global_mean
        items = np.array([item])
        return float(self._predict_items(user, items)[0])

    def _predict_items(self, u: int, items: np.ndarray) -> np.ndarray:
        """Rating of user u for each item, from the k most similar items u has rated."""
        S = self.similarity
        R = self.matrix
        mask = self.mask
        item_mean = self.item_means
        k = self.k

        rated = np.where(mask[u])[0]
        k_eff = min(k, len(rated))
        if k_eff < 1:
            return np.clip(item_mean[items], 1, 5)
        sims = S[np.ix_(items, rated)]
        top = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
        w = np.take_along_axis(sims, top, axis=1)
        nb = rated[top]
        dev = R[u, nb] - item_mean[nb]
        num, den = (w * dev).sum(axis=1), np.abs(w).sum(axis=1)
        adj = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return np.clip(item_mean[items] + adj, 1, 5)

    def score_all_items(self, user: int) -> np.ndarray:
        """Score all candidate items for one user."""
        return self._predict_items(user, np.arange(self.matrix.shape[1]))
