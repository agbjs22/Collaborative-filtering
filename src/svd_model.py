"""SVD matrix factorisation recommender with Surprise and NumPy SGD fallback."""
from __future__ import annotations
import numpy as np


class SVDRecommender:
    """Biased latent-factor model trained with SGD (using Surprise if available, NumPy SGD otherwise)."""
    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 30,
        lr: float = 0.005,
        reg: float = 0.02,
        seed: int = 42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

    def fit(self, matrix: np.ndarray) -> "SVDRecommender":
        """Train user/item biases and latent vectors on observed ratings."""
        self.matrix = matrix
        self.mask = matrix > 0
        self.n_users, self.n_items = matrix.shape
        self.global_mean = float(matrix[self.mask].mean()) if self.mask.any() else 3.0

        try:
            import pandas as pd
            from surprise import Dataset, Reader, SVD

            users, items = np.nonzero(self.mask)
            ratings = matrix[users, items]
            df = pd.DataFrame({"user": users, "item": items, "rating": ratings})
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(df[["user", "item", "rating"]].astype(int), reader)
            trainset = data.build_full_trainset()

            svd = SVD(
                n_factors=self.n_factors,
                n_epochs=self.n_epochs,
                lr_all=self.lr,
                reg_all=self.reg,
                random_state=self.seed,
            )
            svd.fit(trainset)

            # Build full prediction matrix
            uu = np.array([trainset._raw2inner_id_users.get(u, -1) for u in range(self.n_users)])
            ii = np.array([trainset._raw2inner_id_items.get(i, -1) for i in range(self.n_items)])
            bu = np.where(uu >= 0, svd.bu[uu], 0.0)
            bi = np.where(ii >= 0, svd.bi[ii], 0.0)
            both = (uu[:, None] >= 0) & (ii[None, :] >= 0)
            pq = np.where(both, svd.pu[uu] @ svd.qi[ii].T, 0.0)
            self.est = np.clip(trainset.global_mean + bu[:, None] + bi[None, :] + pq, 1.0, 5.0)

        except ImportError:
            # Fallback to pure NumPy SGD
            users, items = np.nonzero(self.mask)
            ratings = matrix[users, items].astype(float)
            rng = np.random.default_rng(self.seed)

            user_bias = np.zeros(self.n_users, dtype=float)
            item_bias = np.zeros(self.n_items, dtype=float)
            user_factors = rng.normal(0.0, 0.1, (self.n_users, self.n_factors))
            item_factors = rng.normal(0.0, 0.1, (self.n_items, self.n_factors))

            order = np.arange(len(ratings))
            for _ in range(self.n_epochs):
                rng.shuffle(order)
                for idx in order:
                    u, i, r = users[idx], items[idx], ratings[idx]
                    pred = self.global_mean + user_bias[u] + item_bias[i] + float(user_factors[u] @ item_factors[i])
                    err = r - pred
                    user_bias[u] += self.lr * (err - self.reg * user_bias[u])
                    item_bias[i] += self.lr * (err - self.reg * item_bias[i])
                    pu = user_factors[u].copy()
                    user_factors[u] += self.lr * (err * item_factors[i] - self.reg * user_factors[u])
                    item_factors[i] += self.lr * (err * pu - self.reg * item_factors[i])

            self.est = np.clip(
                self.global_mean + user_bias[:, None] + item_bias[None, :] + user_factors @ item_factors.T,
                1.0,
                5.0,
            )

        return self

    def predict(self, user: int, item: int) -> float:
        """Predict a bounded rating for a known user and item."""
        if user >= self.n_users or item >= self.n_items:
            return self.global_mean
        return float(self.est[user, item])

    def score_all_items(self, user: int) -> np.ndarray:
        """Return predicted ratings for every item for a user."""
        if user >= self.n_users:
            return np.full(self.n_items, self.global_mean)
        return self.est[user].copy()
