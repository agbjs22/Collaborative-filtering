"""Rating and top-N evaluation utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd


def rating_metrics(model, test: pd.DataFrame) -> dict[str, float]:
    """Compute RMSE and MAE for rating predictions."""
    truth = test.rating.to_numpy(dtype=float)
    predicted = np.array([model.predict(int(u), int(i)) for u, i in zip(test.user_idx, test.item_idx)])
    error = predicted - truth
    return {"RMSE": float(np.sqrt(np.mean(error ** 2))), "MAE": float(np.mean(np.abs(error)))}


def ranking_metrics(model, train_matrix: np.ndarray, test: pd.DataFrame, k: int = 10, max_users: int | None = None) -> dict[str, float]:
    """Average Precision@K, Recall@K, and NDCG@K over users with relevant test items."""
    relevant = test.loc[test.rating >= 4].groupby("user_idx")["item_idx"].apply(set)
    users = list(relevant.index)
    if max_users is not None: users = users[:max_users]
    precision, recall, ndcg = [], [], []
    for user in users:
        scores = model.score_all_items(int(user))
        scores[train_matrix[int(user)] > 0] = -np.inf
        selected = np.argpartition(scores, -k)[-k:]
        selected = selected[np.argsort(scores[selected])[::-1]]
        hits = np.array([item in relevant[user] for item in selected], dtype=float)
        precision.append(hits.mean())
        recall.append(hits.sum() / len(relevant[user]))
        ideal = min(k, len(relevant[user]))
        dcg = np.sum(hits / np.log2(np.arange(2, k + 2)))
        idcg = np.sum(1 / np.log2(np.arange(2, ideal + 2)))
        ndcg.append(dcg / idcg if idcg else 0.0)
    return {f"Precision@{k}": float(np.mean(precision)), f"Recall@{k}": float(np.mean(recall)), f"NDCG@{k}": float(np.mean(ndcg))}


def evaluate_model(model, train_matrix: np.ndarray, test: pd.DataFrame, k: int = 10, max_ranking_users: int | None = None) -> dict[str, float]:
    """Run the full rating-prediction and ranking evaluation suite."""
    return rating_metrics(model, test) | ranking_metrics(model, train_matrix, test, k, max_ranking_users)
