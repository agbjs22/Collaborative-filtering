import numpy as np
import pandas as pd

from src.data import split_ratings, build_matrix
from src.similarity import cosine_similarity, pearson_similarity, adjusted_cosine_similarity
from src.user_based import UserBasedCF
from src.item_based import ItemBasedCF
from src.evaluate import rating_metrics, ranking_metrics


def matrix():
    return np.array([[5, 4, 0, 1], [4, 0, 3, 1], [0, 2, 5, 4], [1, 1, 0, 5]], dtype=float)


def test_similarity_matrices_are_valid():
    for similarity in (cosine_similarity(matrix()), pearson_similarity(matrix()), adjusted_cosine_similarity(matrix())):
        assert np.allclose(similarity, similarity.T)
        assert np.allclose(np.diag(similarity), 0)
        assert np.all(np.abs(similarity) <= 1 + 1e-8)


def test_predictions_are_bounded():
    for model in (UserBasedCF().fit(matrix()), ItemBasedCF().fit(matrix())):
        assert 1 <= model.predict(0, 2) <= 5


def test_split_has_no_pair_overlap():
    ratings = pd.DataFrame({"user_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5], "item_id": range(10), "rating": [1] * 10, "user_idx": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4], "item_idx": range(10)})
    train, test = split_ratings(ratings)
    assert len(train) == 8 and len(test) == 2
    assert not set(zip(train.user_id, train.item_id)) & set(zip(test.user_id, test.item_id))


def test_perfect_predictor_has_zero_rmse():
    class Perfect:
        def predict(self, user, item): return float(item + 1)
    test = pd.DataFrame({"user_idx": [0, 0], "item_idx": [0, 1], "rating": [1, 2]})
    assert rating_metrics(Perfect(), test)["RMSE"] == 0


def test_ranking_metrics_are_bounded():
    train = np.array([[5, 0, 0], [0, 4, 0]], dtype=float)
    test = pd.DataFrame({"user_idx": [0, 1], "item_idx": [1, 2], "rating": [5, 4]})
    class Model:
        def score_all_items(self, user): return np.array([0.1, 0.9, 0.8])
    metrics = ranking_metrics(Model(), train, test, k=2)
    assert all(0 <= value <= 1 for value in metrics.values())
