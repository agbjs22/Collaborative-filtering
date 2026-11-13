"""Rating prediction and ranking evaluation utilities for recommendation models."""
from __future__ import annotations
import numpy as np
import pandas as pd


def evaluate_model(
    model,
    train_matrix: np.ndarray,
    test: pd.DataFrame,
    k: int = 10,
    max_ranking_users: int | None = None,
    rated_scores: bool = True,
    relevance_threshold: float = 4.0,
) -> dict[str, float]:
    """Run the full rating-prediction and top-K ranking evaluation suite.

    Parameters
    ----------
    model : fitted recommendation model
        Must implement score_all_items(user_idx) and predict(user_idx, item_idx).
    train_matrix : np.ndarray
        2D dense user-item matrix of training ratings (0 indicating unobserved).
    test : pd.DataFrame
        Held-out ratings containing user, item and rating columns.
    k : int, default=10
        Cutoff length for top-N recommendation list.
    max_ranking_users : int | None
        Optional limit on user count for faster evaluation runs.
    rated_scores : bool, default=True
        Whether the model outputs rating predictions (True) or ranking-only scores (False).
        For unpersonalized popularity models, set to False so RMSE/MAE evaluate to NaN.
    relevance_threshold : float, default=4.0
        Rating score threshold at or above which an item is considered relevant.

    Returns
    -------
    dict[str, float]
        Dictionary of RMSE, MAE, Precision@K, Recall@K, F1@K, NDCG@K.
    """
    user_col = "u" if "u" in test.columns else "user_idx"
    item_col = "i" if "i" in test.columns else "item_idx"
    rating_col = "rating"

    test_by_user = {u: g for u, g in test.groupby(user_col)}
    users = list(test_by_user.keys())
    if max_ranking_users is not None:
        users = users[:max_ranking_users]

    err = []
    p_list, rc_list, f1_list, ndcg_list = [], [], [], []

    for u in users:
        g = test_by_user[u]
        scores = model.score_all_items(int(u))

        # 1. Rating prediction evaluation (on observed test pairs)
        if rated_scores:
            predicted_ratings = scores[g[item_col].to_numpy(dtype=int)]
            err.extend(g[rating_col].to_numpy(dtype=float) - predicted_ratings)

        # 2. Ranking evaluation (filtering out already-rated items in training)
        relevant = set(g.loc[g[rating_col] >= relevance_threshold, item_col])
        if relevant:
            seen_mask = train_matrix[int(u)] > 0
            # Exclude training items from recommendation pool
            masked_scores = np.where(seen_mask, -np.inf, scores)
            ranked = np.argsort(-masked_scores, kind="stable")[:k]
            hits = np.array([1.0 if x in relevant else 0.0 for x in ranked])

            # Precision@K
            p_val = float(hits.sum() / k)
            # Recall@K
            r_val = float(hits.sum() / len(relevant))
            # F1@K
            if (p_val + r_val) > 0:
                f1_val = 2.0 * (p_val * r_val) / (p_val + r_val)
            else:
                f1_val = 0.0

            # NDCG@K
            dcg = float(np.sum(hits / np.log2(np.arange(2, k + 2))))
            ideal_hits = min(len(relevant), k)
            idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_hits + 2))))
            ndcg_val = dcg / idcg if idcg > 0 else 0.0

            p_list.append(p_val)
            rc_list.append(r_val)
            f1_list.append(f1_val)
            ndcg_list.append(ndcg_val)

    err_arr = np.array(err, dtype=float)
    rmse = float(np.sqrt(np.mean(err_arr ** 2))) if (rated_scores and len(err_arr) > 0) else np.nan
    mae = float(np.mean(np.abs(err_arr))) if (rated_scores and len(err_arr) > 0) else np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        f"Precision@{k}": float(np.mean(p_list)) if p_list else 0.0,
        f"Recall@{k}": float(np.mean(rc_list)) if rc_list else 0.0,
        f"F1@{k}": float(np.mean(f1_list)) if f1_list else 0.0,
        f"NDCG@{k}": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
    }


def rating_metrics(model, test: pd.DataFrame) -> dict[str, float]:
    """Compute RMSE and MAE directly from predict(u, i)."""
    user_col = "u" if "u" in test.columns else "user_idx"
    item_col = "i" if "i" in test.columns else "item_idx"
    truth = test.rating.to_numpy(dtype=float)
    predicted = np.array([model.predict(int(u), int(i)) for u, i in zip(test[user_col], test[item_col])])
    error = predicted - truth
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
    }


def ranking_metrics(
    model,
    train_matrix: np.ndarray,
    test: pd.DataFrame,
    k: int = 10,
    max_users: int | None = None,
    relevance_threshold: float = 4.0,
) -> dict[str, float]:
    """Average Precision@K, Recall@K, F1@K, and NDCG@K over users with relevant test items."""
    res = evaluate_model(
        model=model,
        train_matrix=train_matrix,
        test=test,
        k=k,
        max_ranking_users=max_users,
        rated_scores=False,
        relevance_threshold=relevance_threshold,
    )
    return {
        f"Precision@{k}": res[f"Precision@{k}"],
        f"Recall@{k}": res[f"Recall@{k}"],
        f"F1@{k}": res[f"F1@{k}"],
        f"NDCG@{k}": res[f"NDCG@{k}"],
    }
