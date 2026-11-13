"""SVD matrix factorisation recommender using the Surprise library."""
from __future__ import annotations
import numpy as np


class SVDRecommender:
    """Biased latent-factor model trained via Surprise's SVD implementation."""
    def __init__(self, n_factors: int = 50, n_epochs: int = 30,
                 lr: float = 0.005, reg: float = 0.02, seed: int = 42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

    def fit(self, matrix: np.ndarray) -> "SVDRecommender":
        """Train user/item biases and latent vectors on observed ratings."""
        import pandas as pd
        from surprise import Dataset, Reader, SVD

        self.matrix = matrix
        self.mask = matrix > 0
        self.n_users, self.n_items = matrix.shape
        self.global_mean = float(matrix[self.mask].mean()) if self.mask.any() else 3.0

        # Build Surprise training set from the rating matrix
        users, items = np.nonzero(self.mask)
        ratings = matrix[users, items]
        df = pd.DataFrame({"user": users, "item": items, "rating": ratings})
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[["user", "item", "rating"]].astype(int), reader)
        trainset = data.build_full_trainset()

        svd = SVD(n_factors=self.n_factors, n_epochs=self.n_epochs,
                  lr_all=self.lr, reg_all=self.reg, random_state=self.seed)
        svd.fit(trainset)

        # Build the full prediction matrix for fast scoring
        self._build_prediction_matrix(svd, trainset)
        return self

    def _build_prediction_matrix(self, model, trainset) -> None:
        """Convert Surprise internal arrays into a full n_users x n_items matrix."""
        uu = np.array([trainset._raw2inner_id_users.get(u, -1)
                       for u in range(self.n_users)])
        ii = np.array([trainset._raw2inner_id_items.get(i, -1)
                       for i in range(self.n_items)])
        bu = np.where(uu >= 0, model.bu[uu], 0.0)
        bi = np.where(ii >= 0, model.bi[ii], 0.0)
        both = (uu[:, None] >= 0) & (ii[None, :] >= 0)
        pq = np.where(both, model.pu[uu] @ model.qi[ii].T, 0.0)
        self.est = np.clip(trainset.global_mean + bu[:, None] + bi[None, :] + pq, 1, 5)

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
