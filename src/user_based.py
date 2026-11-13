"""User-based collaborative filtering implemented with NumPy."""
from __future__ import annotations
import numpy as np
from .similarity import cosine_sim, pearson_sim


class UserBasedCF:
    """Mean-centred user-neighbourhood recommender."""
    def __init__(self, metric: str = "cosine", k: int = 40):
        self.metric, self.k = metric, k

    def fit(self, matrix: np.ndarray) -> "UserBasedCF":
        self.matrix = matrix
        self.mask = matrix > 0
        self.global_mean = float(matrix[self.mask].mean()) if self.mask.any() else 3.0
        counts = self.mask.sum(axis=1)
        self.means = np.where(counts > 0, matrix.sum(axis=1) / np.maximum(counts, 1), self.global_mean)
        if self.metric == "cosine":
            self.similarity = cosine_sim(matrix)
        else:
            self.similarity = pearson_sim(matrix, self.mask, self.means)
        return self

    def predict(self, user: int, item: int) -> float:
        """Predict a user-item rating using the strongest eligible neighbours."""
        if user >= self.matrix.shape[0] or item >= self.matrix.shape[1]:
            return self.global_mean
        items = np.array([item])
        return float(self._predict_items(user, items)[0])

    def _predict_items(self, u: int, items: np.ndarray) -> np.ndarray:
        """Rating of user u for each item, from the k most similar users who rated it."""
        S = self.similarity
        R = self.matrix
        mask = self.mask
        mean = self.means
        k = self.k

        sims = np.where(mask[:, items], S[u][:, None], -np.inf)    # raters only
        k_eff = min(k, sims.shape[0])
        if k_eff < 1:
            return np.full(len(items), mean[u])
        top = np.argpartition(-sims, k_eff - 1, axis=0)[:k_eff]
        w = np.take_along_axis(sims, top, axis=0)
        ok = np.isfinite(w)
        w = np.where(ok, w, 0.0)
        dev = (R[top, items[None, :]] - mean[top, None]) * ok
        num, den = (w * dev).sum(axis=0), np.abs(w).sum(axis=0)
        return np.clip(mean[u] + np.divide(num, den, out=np.zeros_like(num), where=den > 0), 1, 5)

    def score_all_items(self, user: int) -> np.ndarray:
        """Score all items for a user."""
        return self._predict_items(user, np.arange(self.matrix.shape[1]))
