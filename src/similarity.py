"""Vectorised similarity functions for ratings matrices."""
from __future__ import annotations
import numpy as np


def cosine_sim(M: np.ndarray) -> np.ndarray:
    """Cosine similarity over zero-filled rating vectors (row-wise)."""
    norm = np.linalg.norm(M, axis=1, keepdims=True)
    norm[norm == 0] = 1
    S = (M / norm) @ (M / norm).T
    np.fill_diagonal(S, 0)
    return np.clip(S, -1, 1)


# Keep the old name as an alias for backwards compatibility
cosine_similarity = cosine_sim


def pearson_sim(R: np.ndarray, mask: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Proper co-rated Pearson correlation with mask-aware denominator.

    Each pair's correlation is computed only over co-rated items, not
    the full rating vector.
    """
    C = (R - mean[:, None]) * mask              # mean-centred, 0 where unrated
    M = mask.astype(float)
    num = C @ C.T                               # sum over co-rated items
    den = np.sqrt((C ** 2 @ M.T) * (M @ (C ** 2).T))
    S = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    np.fill_diagonal(S, 0)
    return S


# Keep the old name as an alias
pearson_similarity = pearson_sim


def adjusted_cosine_sim(R: np.ndarray, mask: np.ndarray, user_mean: np.ndarray) -> np.ndarray:
    """Item-item cosine after subtracting each user's observed-rating mean."""
    C = (R - user_mean[:, None]) * mask
    return cosine_sim(C.T)                       # item-item similarity


# Keep the old name as an alias
adjusted_cosine_similarity = adjusted_cosine_sim
