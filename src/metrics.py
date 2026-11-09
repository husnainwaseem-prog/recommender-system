"""Ranking metrics: Precision@K, Recall@K, NDCG@K.

A test item counts as "relevant" when the user rated it >= 4 in the
test set. All metrics are averaged over users who have at least one
relevant test item.
"""

from __future__ import annotations

import numpy as np


def precision_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    rec = recommended[:k]
    if not rec:
        return 0.0
    return len(set(rec) & relevant) / len(rec)


def recall_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def dcg_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal))
    if idcg == 0:
        return 0.0
    return dcg_at_k(recommended, relevant, k) / idcg


def evaluate_user(recommended: list[int], relevant: set[int], k: int = 10) -> dict[str, float]:
    return {
        f"precision@{k}": precision_at_k(recommended, relevant, k),
        f"recall@{k}": recall_at_k(recommended, relevant, k),
        f"ndcg@{k}": ndcg_at_k(recommended, relevant, k),
    }


def mean_metrics(all_recs: dict[int, list[int]], all_relevant: dict[int, set[int]],
                 k: int = 10) -> dict[str, float]:
    """Average each metric over users with >= 1 relevant test item."""
    sums = {f"precision@{k}": 0.0, f"recall@{k}": 0.0, f"ndcg@{k}": 0.0}
    n = 0
    for user, relevant in all_relevant.items():
        if not relevant:
            continue
        recs = all_recs.get(user, [])
        for key, val in evaluate_user(recs, relevant, k).items():
            sums[key] += val
        n += 1
    return {key: round(val / n, 4) for key, val in sums.items()} if n else sums
