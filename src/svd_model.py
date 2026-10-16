"""Matrix-factorisation recommender with a dependency-free NumPy implementation."""
from __future__ import annotations
import numpy as np


class SVDRecommender:
    """Biased latent-factor model trained with stochastic gradient descent."""
    def __init__(self, n_factors: int = 50, n_epochs: int = 30, lr: float = 0.005, reg: float = 0.02, seed: int = 42):
        self.n_factors, self.n_epochs, self.lr, self.reg, self.seed = n_factors, n_epochs, lr, reg, seed

    def fit(self, matrix: np.ndarray) -> "SVDRecommender":
        """Train user/item biases and latent vectors on observed ratings."""
        self.matrix, self.mask = matrix, matrix > 0
        users, items = np.nonzero(self.mask)
        ratings = matrix[users, items]
        self.global_mean = float(ratings.mean())
        rng = np.random.default_rng(self.seed)
        self.user_bias = np.zeros(matrix.shape[0]); self.item_bias = np.zeros(matrix.shape[1])
        self.user_factors = rng.normal(0, 0.1, (matrix.shape[0], self.n_factors))
        self.item_factors = rng.normal(0, 0.1, (matrix.shape[1], self.n_factors))
        order = np.arange(len(ratings))
        for _ in range(self.n_epochs):
            rng.shuffle(order)
            for index in order:
                u, i, rating = users[index], items[index], ratings[index]
                prediction = self.global_mean + self.user_bias[u] + self.item_bias[i] + self.user_factors[u] @ self.item_factors[i]
                error = rating - prediction
                self.user_bias[u] += self.lr * (error - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (error - self.reg * self.item_bias[i])
                pu = self.user_factors[u].copy()
                self.user_factors[u] += self.lr * (error * self.item_factors[i] - self.reg * self.user_factors[u])
                self.item_factors[i] += self.lr * (error * pu - self.reg * self.item_factors[i])
        return self

    def predict(self, user: int, item: int) -> float:
        """Predict a bounded rating for a known user and item."""
        if user >= len(self.user_bias) or item >= len(self.item_bias): return self.global_mean
        score = self.global_mean + self.user_bias[user] + self.item_bias[item] + self.user_factors[user] @ self.item_factors[item]
        return float(np.clip(score, 1, 5))

    def score_all_items(self, user: int) -> np.ndarray:
        """Return predicted ratings for every item for a user."""
        scores = self.global_mean + self.user_bias[user] + self.item_bias + self.item_factors @ self.user_factors[user]
        return np.clip(scores, 1, 5)
