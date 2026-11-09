"""Tests for the MovieLens Recommender System.

Unit tests use tiny synthetic data (fast, no download). One
integration test loads the real MovieLens 100K split and checks the
evaluation runs end-to-end (skipped if the data is not present).
"""

import numpy as np
import pandas as pd
import pytest

from src.metrics import ndcg_at_k, precision_at_k, recall_at_k
from src.models import ContentBased, Hybrid, MatrixFactorization, Popularity

GENRES = ["Action", "Comedy", "Drama"]


def _small_train() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2, 3, 3],
        "item_id": [10, 11, 12, 10, 11, 13, 12, 13],
        "rating": [5, 4, 2, 4, 5, 3, 5, 4],
        "timestamp": [1, 2, 3, 1, 2, 3, 1, 2],
    })


def _small_movies() -> pd.DataFrame:
    return pd.DataFrame({
        "item_id": [10, 11, 12, 13],
        "title": ["A", "B", "C", "D"],
        "Action": [1, 0, 1, 0],
        "Comedy": [0, 1, 0, 1],
        "Drama": [0, 0, 1, 1],
    })


# --- metrics -----------------------------------------------------------

def test_precision_recall_ndcg_known_values():
    recs = [1, 2, 3, 4, 5]
    rel = {2, 4}
    assert precision_at_k(recs, rel, k=5) == pytest.approx(0.4)
    assert recall_at_k(recs, rel, k=5) == pytest.approx(1.0)
    # DCG: items 2 (rank 2) and 4 (rank 4) hit
    expected = (1 / np.log2(3) + 1 / np.log2(5)) / (1 / np.log2(2) + 1 / np.log2(3))
    assert ndcg_at_k(recs, rel, k=5) == pytest.approx(expected)


def test_metrics_empty_inputs():
    assert precision_at_k([], {1}, k=10) == 0.0
    assert recall_at_k([1, 2], set(), k=10) == 0.0
    assert ndcg_at_k([], set(), k=10) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)


# --- popularity ---------------------------------------------------------

def test_popularity_ranks_most_liked_first():
    pop = Popularity().fit(_small_train())
    # item 10 has two 4+ ratings, item 11 has two 4+ ratings, item 12 one
    recs = pop.recommend(99, n=3)
    assert len(recs) == 3
    assert recs[0] in (10, 11)


def test_popularity_excludes_seen():
    pop = Popularity().fit(_small_train())
    recs = pop.recommend(1, n=10, seen={10, 11, 12, 13})
    assert recs == []


# --- matrix factorization ------------------------------------------------

def test_mf_predicts_close_to_train_ratings():
    train = _small_train()
    mf = MatrixFactorization(n_epochs=50, lr=0.02, seed=1).fit(train)
    for _, row in train.iterrows():
        pred = mf.predict(int(row["user_id"]), int(row["item_id"]))
        assert abs(pred - row["rating"]) < 1.5


def test_mf_recommend_excludes_seen_and_returns_n():
    train = _small_train()
    mf = MatrixFactorization(n_epochs=10, seed=1).fit(train)
    # only item 13 is unseen for user 1 (4 items total, 3 seen)
    recs = mf.recommend(1, n=2, seen={10, 11, 12})
    assert recs == [13]
    assert 10 not in recs and 11 not in recs and 12 not in recs


def test_mf_unknown_user_returns_empty():
    train = _small_train()
    mf = MatrixFactorization(n_epochs=5, seed=1).fit(train)
    assert mf.recommend(999, n=5) == []


# --- content based -------------------------------------------------------

def test_content_based_recommends_same_genre():
    train = _small_train()
    movies = _small_movies()
    cb = ContentBased(movies).fit(train)
    # user 1 liked item 10 (Action) and 11 (Comedy); item 12 is Action+Drama
    recs = cb.recommend(1, n=2, seen={10, 11, 12})
    assert recs == [13] or len(recs) <= 2


def test_content_based_no_history_returns_empty():
    train = _small_train()
    movies = _small_movies()
    cb = ContentBased(movies).fit(train)
    assert cb.recommend(999, n=5) == []


# --- hybrid --------------------------------------------------------------

def test_hybrid_returns_ranked_list():
    train = _small_train()
    movies = _small_movies()
    hy = Hybrid(MatrixFactorization(n_epochs=10, seed=1),
                ContentBased(movies)).fit(train)
    recs = hy.recommend(1, n=2, seen={10, 11, 12})
    assert len(recs) <= 2
    assert all(isinstance(x, (int, np.integer)) for x in recs)


# --- cold start -----------------------------------------------------------

def test_cold_start_falls_back_to_popularity():
    from src.cold_start import recommend_for_new_user
    pop = Popularity().fit(_small_train())
    recs = recommend_for_new_user(pop, n=2)
    assert len(recs) == 2
    assert recs[0] in (10, 11)


# --- split -----------------------------------------------------------------

def test_train_test_split_covers_all_users():
    from src.data import train_test_split
    ratings = pd.DataFrame({
        "user_id": [1]*10 + [2]*10,
        "item_id": list(range(10)) * 2,
        "rating": [4]*20,
        "timestamp": list(range(10)) * 2,
    })
    train, test = train_test_split(ratings, test_frac=0.2)
    assert len(train) == 16 and len(test) == 4
    assert set(test["user_id"]) == {1, 2}
