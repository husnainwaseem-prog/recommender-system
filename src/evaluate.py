"""Evaluate every model on the train/test split.

For each test user, the model recommends 10 movies it has not seen in
train; relevance = test items the user rated >= 4. Metrics:
Precision@10, Recall@10, NDCG@10 (see metrics.py). Unknown users in
test fall back to popularity (cold start) so every model is scored on
the same users.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .cold_start import recommend_for_new_user
from .data import load_movies, load_ratings, train_test_split
from .metrics import mean_metrics
from .models import ContentBased, Hybrid, MatrixFactorization, Popularity

K = 10
RELEVANT_RATING = 4


def evaluate_all(train: pd.DataFrame, test: pd.DataFrame, movies: pd.DataFrame,
                 k: int = K, seed: int = 42) -> dict:
    """Train all four models and score them. Returns the full report dict."""
    # relevant test items per user (rating >= 4)
    relevant: dict[int, set[int]] = {}
    for user, group in test[test["rating"] >= RELEVANT_RATING].groupby("user_id"):
        relevant[int(user)] = set(group["item_id"].tolist())
    users = sorted(relevant.keys())

    # seen items in train (to exclude from recommendations)
    seen: dict[int, set[int]] = {}
    for user, group in train.groupby("user_id"):
        seen[int(user)] = set(group["item_id"].tolist())

    pop = Popularity().fit(train)
    mf = MatrixFactorization(seed=seed).fit(train)
    cb = ContentBased(movies).fit(train)
    hybrid = Hybrid(MatrixFactorization(seed=seed), ContentBased(movies)).fit(train)

    models = {"popularity": pop, "matrix_factorization": mf,
              "content_based": cb, "hybrid": hybrid}

    results: dict[str, dict] = {}
    for name, model in models.items():
        recs: dict[int, list[int]] = {}
        for user in users:
            user_seen = seen.get(user, set())
            r = model.recommend(user, n=k, seen=user_seen)
            if not r:  # cold start inside test -> popularity fallback
                r = recommend_for_new_user(pop, n=k)
                r = [i for i in r if i not in user_seen][:k]
            recs[user] = r
        results[name] = mean_metrics(recs, relevant, k=k)

    return {
        "k": k,
        "relevant_rating": RELEVANT_RATING,
        "n_test_users": len(users),
        "n_train_ratings": int(len(train)),
        "n_test_ratings": int(len(test)),
        "models": results,
    }


def main(data_dir: str = "data/ml-100k", out_json: str = "results/evaluation.json") -> dict:
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    train, test = train_test_split(ratings)
    print(f"Train: {len(train)} ratings | Test: {len(test)} ratings | "
          f"Users: {ratings['user_id'].nunique()} | Movies: {ratings['item_id'].nunique()}")
    report = evaluate_all(train, test, movies)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2))
    print(f"\nRanking metrics @ K={report['k']} (relevant = rating >= 4):")
    for name, m in report["models"].items():
        print(f"  {name:22s} P@10 {m['precision@10']:.4f}  "
              f"R@10 {m['recall@10']:.4f}  NDCG@10 {m['ndcg@10']:.4f}")
    print(f"\nFull report: {out_json}")
    return report
