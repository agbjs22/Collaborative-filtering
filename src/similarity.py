"""Vectorised similarity functions for ratings matrices."""
from __future__ import annotations
import numpy as np


def _cosine_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.divide(matrix, norms, out=np.zeros_like(matrix, dtype=float), where=norms > 0)
    sim = safe @ safe.T
    np.fill_diagonal(sim, 0.0)
    return np.clip(sim, -1, 1)


def cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity over zero-filled rating vectors."""
    return _cosine_rows(matrix.astype(float, copy=False))


def pearson_similarity(matrix: np.ndarray) -> np.ndarray:
    """Mean-centred row similarity, with unrated entries excluded from products."""
    observed = matrix > 0
    counts = observed.sum(axis=1, keepdims=True)
    means = np.divide(matrix.sum(axis=1, keepdims=True), counts, out=np.zeros((matrix.shape[0], 1)), where=counts > 0)
    centred = np.where(observed, matrix - means, 0.0)
    return _cosine_rows(centred)


def adjusted_cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """Item-item cosine after subtracting each user's observed-rating mean."""
    observed = matrix > 0
    counts = observed.sum(axis=1, keepdims=True)
    means = np.divide(matrix.sum(axis=1, keepdims=True), counts, out=np.zeros((matrix.shape[0], 1)), where=counts > 0)
    centred = np.where(observed, matrix - means, 0.0)
    return _cosine_rows(centred.T)
