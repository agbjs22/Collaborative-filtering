"""Vectorised similarity functions for ratings matrices."""
from __future__ import annotations
import numpy as np


def cosine_sim(M: np.ndarray) -> np.ndarray:
    """Cosine similarity over zero-filled rating vectors (row-wise)."""
    norm = np.linalg.norm(M, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    safe = M / norm
    S = safe @ safe.T
    np.fill_diagonal(S, 0.0)
    return np.clip(S, -1.0, 1.0)


cosine_similarity = cosine_sim


def pearson_sim(
    R: np.ndarray,
    mask: np.ndarray | None = None,
    mean: np.ndarray | None = None,
) -> np.ndarray:
    """Proper co-rated Pearson correlation with mask-aware denominator.

    Supports both full 3-arg signature (R, mask, mean) for high performance
    in inner loops, and 1-arg signature (R) for convenience and testing.
    """
    if mask is None:
        mask = R > 0
    if mean is None:
        counts = mask.sum(axis=1)
        mean = np.where(counts > 0, R.sum(axis=1) / np.maximum(counts, 1), 0.0)

    C = (R - mean[:, None]) * mask              # mean-centred, 0 where unrated
    M = mask.astype(float)
    num = C @ C.T                               # sum over co-rated items
    den = np.sqrt((C ** 2 @ M.T) * (M @ (C ** 2).T))
    S = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    np.fill_diagonal(S, 0.0)
    return np.clip(S, -1.0, 1.0)


pearson_similarity = pearson_sim


def adjusted_cosine_sim(
    R: np.ndarray,
    mask: np.ndarray | None = None,
    user_mean: np.ndarray | None = None,
) -> np.ndarray:
    """Item-item cosine after subtracting each user's observed-rating mean.

    Supports both 3-arg and 1-arg signatures.
    """
    if mask is None:
        mask = R > 0
    if user_mean is None:
        counts = mask.sum(axis=1)
        user_mean = np.where(counts > 0, R.sum(axis=1) / np.maximum(counts, 1), 0.0)

    C = (R - user_mean[:, None]) * mask
    return cosine_sim(C.T)                       # item-item similarity


adjusted_cosine_similarity = adjusted_cosine_sim
