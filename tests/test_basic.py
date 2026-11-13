"""Unit tests for collaborative filtering algorithms, similarities, and evaluation metrics."""
import numpy as np
import pandas as pd
import pytest

from src.data import split_ratings, build_matrix
from src.similarity import (
    cosine_sim,
    cosine_similarity,
    pearson_sim,
    pearson_similarity,
    adjusted_cosine_sim,
    adjusted_cosine_similarity,
)
from src.user_based import UserBasedCF
from src.item_based import ItemBasedCF
from src.popularity import PopularityRecommender
from src.svd_model import SVDRecommender
from src.evaluate import rating_metrics, ranking_metrics, evaluate_model


@pytest.fixture
def sample_matrix():
    return np.array([
        [5, 4, 0, 1],
        [4, 0, 3, 1],
        [0, 2, 5, 4],
        [1, 1, 0, 5],
    ], dtype=float)


def test_similarity_matrices_are_valid(sample_matrix):
    """Test symmetry, diagonal zeroing, and value bounds [-1, 1] for all similarity metrics."""
    sim_funcs = [cosine_similarity, pearson_similarity, adjusted_cosine_similarity]
    for func in sim_funcs:
        S = func(sample_matrix)
        assert np.allclose(S, S.T, atol=1e-7), f"{func.__name__} is not symmetric"
        assert np.allclose(np.diag(S), 0.0), f"{func.__name__} diagonal is not 0"
        assert np.all(S >= -1.0 - 1e-7) and np.all(S <= 1.0 + 1e-7), f"{func.__name__} has values outside [-1, 1]"


def test_similarity_handles_edge_cases():
    """Test zero vectors and all-zero matrix do not produce NaN or Inf."""
    all_zeros = np.zeros((3, 3), dtype=float)
    for func in [cosine_similarity, pearson_similarity, adjusted_cosine_similarity]:
        S = func(all_zeros)
        assert not np.isnan(S).any()
        assert not np.isinf(S).any()
        assert np.all(S == 0.0)

    # Matrix with one completely unrated user
    sparse_mat = np.array([
        [0, 0, 0],
        [5, 4, 3],
        [1, 2, 0]
    ], dtype=float)
    for func in [cosine_similarity, pearson_similarity, adjusted_cosine_similarity]:
        S = func(sparse_mat)
        assert not np.isnan(S).any()
        assert not np.isinf(S).any()


def test_predictions_are_bounded(sample_matrix):
    """Ensure rating predictions stay within the legitimate [1, 5] rating scale."""
    models = [
        UserBasedCF(metric="cosine", k=2).fit(sample_matrix),
        UserBasedCF(metric="pearson", k=2).fit(sample_matrix),
        ItemBasedCF(metric="cosine", k=2).fit(sample_matrix),
        ItemBasedCF(metric="adjusted_cosine", k=2).fit(sample_matrix),
        SVDRecommender(n_factors=2, n_epochs=5, seed=42).fit(sample_matrix),
        PopularityRecommender().fit(sample_matrix),
    ]
    for model in models:
        for u in range(sample_matrix.shape[0]):
            for i in range(sample_matrix.shape[1]):
                pred = model.predict(u, i)
                assert 1.0 <= pred <= 5.0, f"{model} predicted {pred} out of [1, 5] for ({u}, {i})"


def test_split_has_no_pair_overlap():
    """Verify train and test splits are strictly disjoint without data leakage."""
    ratings = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "item_id": list(range(10)),
        "rating": [1] * 10,
        "user_idx": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        "item_idx": list(range(10)),
    })
    train, test = split_ratings(ratings, test_size=0.2, seed=42)
    assert len(train) == 8 and len(test) == 2
    train_pairs = set(zip(train.user_id, train.item_id))
    test_pairs = set(zip(test.user_id, test.item_id))
    assert not (train_pairs & test_pairs), "Data leakage: overlapping user-item pairs between train and test"


def test_perfect_predictor_has_zero_rmse():
    """Verify RMSE equals zero for perfect predictions."""
    class Perfect:
        def predict(self, user, item):
            return float(item + 1)
    test = pd.DataFrame({"user_idx": [0, 0], "item_idx": [0, 1], "rating": [1.0, 2.0]})
    metrics = rating_metrics(Perfect(), test)
    assert np.isclose(metrics["RMSE"], 0.0)
    assert np.isclose(metrics["MAE"], 0.0)


def test_ranking_metrics_are_bounded():
    """Verify Precision@K, Recall@K, F1@K and NDCG@K are within [0, 1]."""
    train = np.array([[5, 0, 0], [0, 4, 0]], dtype=float)
    test = pd.DataFrame({"user_idx": [0, 1], "item_idx": [1, 2], "rating": [5.0, 4.0]})
    class DummyModel:
        def score_all_items(self, user):
            return np.array([0.1, 0.9, 0.8])
    metrics = ranking_metrics(DummyModel(), train, test, k=2)
    for name, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{name} = {value} is out of bounds [0, 1]"
    assert "F1@2" in metrics


def test_no_already_rated_items_in_recommendations():
    """Ensure already-rated items from train_matrix are excluded from recommendations."""
    train = np.array([[5, 0, 5], [0, 5, 0]], dtype=float)
    test = pd.DataFrame({"user_idx": [0], "item_idx": [1], "rating": [5.0]})
    class Model:
        def score_all_items(self, user):
            return np.array([10.0, 1.0, 9.0])  # Items 0 and 2 have highest raw scores
    # Since items 0 and 2 are in train[0], top recommendation must be item 1
    eval_res = evaluate_model(Model(), train, test, k=1, rated_scores=False)
    # If item 1 was selected (hit), Precision@1 should be 1.0
    assert eval_res["Precision@1"] == 1.0
