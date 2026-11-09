"""Recommender models. One interface, four models:

- Popularity: most-liked items overall (the baseline).
- MatrixFactorization: collaborative filtering — latent factors learned
  with SGD from scratch (numpy). The main CF model.
- ContentBased: genre similarity — recommends movies like the ones the
  user already liked, from the 19 MovieLens genre flags.
- Hybrid: weighted blend of MF + content-based scores.

Every model implements fit(train_df) and recommend(user_id, n).
A user with no history gets the popularity list (cold start, see
cold_start.py for the full explanation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class Popularity:
    """Baseline: rank by number of 4+ star ratings, tie-break by mean rating."""

    name = "popularity"

    def fit(self, train: pd.DataFrame) -> "Popularity":
        liked = train[train["rating"] >= 4]
        counts = liked.groupby("item_id").size()
        means = train.groupby("item_id")["rating"].mean()
        self.ranking_ = (
            pd.DataFrame({"count": counts, "mean": means})
            .fillna(0)
            .sort_values(["count", "mean"], ascending=False)
            .index.tolist()
        )
        self.all_items_ = sorted(train["item_id"].unique().tolist())
        return self

    def recommend(self, user_id: int, n: int = 10, seen: set[int] | None = None) -> list[int]:
        seen = seen or set()
        return [i for i in self.ranking_ if i not in seen][:n]


class MatrixFactorization:
    """Collaborative filtering via latent factors, SGD from scratch.

    Predicts rating = global_mean + user_bias + item_bias
                      + user_factors . item_factors.
    """

    name = "matrix_factorization"

    def __init__(self, n_factors: int = 20, lr: float = 0.005, reg: float = 0.02,
                 n_epochs: int = 30, seed: int = 42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> "MatrixFactorization":
        rng = np.random.RandomState(self.seed)
        users = sorted(train["user_id"].unique())
        items = sorted(train["item_id"].unique())
        self.user_idx_ = {u: i for i, u in enumerate(users)}
        self.item_idx_ = {m: i for i, m in enumerate(items)}
        self.users_ = users
        self.items_ = items
        n_u, n_i = len(users), len(items)

        self.global_mean_ = float(train["rating"].mean())
        self.user_bias_ = np.zeros(n_u)
        self.item_bias_ = np.zeros(n_i)
        self.user_factors_ = rng.normal(0, 0.1, (n_u, self.n_factors))
        self.item_factors_ = rng.normal(0, 0.1, (n_i, self.n_factors))

        u = train["user_id"].map(self.user_idx_).to_numpy()
        i = train["item_id"].map(self.item_idx_).to_numpy()
        r = train["rating"].to_numpy(dtype=float)

        for _ in range(self.n_epochs):
            order = rng.permutation(len(r))
            for idx in order:
                uu, ii, rr = u[idx], i[idx], r[idx]
                pred = (self.global_mean_ + self.user_bias_[uu] + self.item_bias_[ii]
                        + self.user_factors_[uu] @ self.item_factors_[ii])
                err = rr - pred
                self.user_bias_[uu] += self.lr * (err - self.reg * self.user_bias_[uu])
                self.item_bias_[ii] += self.lr * (err - self.reg * self.item_bias_[ii])
                uf = self.user_factors_[uu].copy()
                self.user_factors_[uu] += self.lr * (err * self.item_factors_[ii] - self.reg * uf)
                self.item_factors_[ii] += self.lr * (err * uf - self.reg * self.item_factors_[ii])
        self.train_ = train
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_idx_ or item_id not in self.item_idx_:
            return self.global_mean_
        uu, ii = self.user_idx_[user_id], self.item_idx_[item_id]
        return float(self.global_mean_ + self.user_bias_[uu] + self.item_bias_[ii]
                     + self.user_factors_[uu] @ self.item_factors_[ii])

    def recommend(self, user_id: int, n: int = 10, seen: set[int] | None = None) -> list[int]:
        seen = seen or set()
        if user_id not in self.user_idx_:
            return []  # unknown user -> caller falls back to popularity
        uu = self.user_idx_[user_id]
        scores = (self.global_mean_ + self.user_bias_[uu] + self.item_bias_
                  + self.item_factors_ @ self.user_factors_[uu])
        order = np.argsort(-scores)
        return [self.items_[j] for j in order if self.items_[j] not in seen][:n]


class ContentBased:
    """Genre similarity: for each user, average the genre vectors of the
    movies they liked (rating >= 4), weighted by (rating - 3); rank
    unseen movies by cosine similarity to that profile."""

    name = "content_based"

    def __init__(self, movies: pd.DataFrame):
        self.movies = movies

    def fit(self, train: pd.DataFrame) -> "ContentBased":
        # genre columns = every column except the id and title
        self.genre_cols_ = [c for c in self.movies.columns if c not in ("item_id", "title")]
        self.genre_matrix_ = self.movies.set_index("item_id")[self.genre_cols_].to_numpy(dtype=float)
        self.movie_ids_ = self.movies["item_id"].tolist()
        self.pos_ = {m: i for i, m in enumerate(self.movie_ids_)}
        self.train_ = train
        return self

    def _user_profile(self, user_id: int) -> np.ndarray | None:
        liked = self.train_[(self.train_["user_id"] == user_id) & (self.train_["rating"] >= 4)]
        if liked.empty:
            return None
        vecs, weights = [], []
        for _, row in liked.iterrows():
            if row["item_id"] in self.pos_:
                vecs.append(self.genre_matrix_[self.pos_[row["item_id"]]])
                weights.append(row["rating"] - 3.0)
        if not vecs:
            return None
        profile = np.average(np.array(vecs), axis=0, weights=np.array(weights))
        return profile

    def recommend(self, user_id: int, n: int = 10, seen: set[int] | None = None) -> list[int]:
        seen = seen or set()
        profile = self._user_profile(user_id)
        if profile is None or np.linalg.norm(profile) == 0:
            return []
        sims = cosine_similarity(profile.reshape(1, -1), self.genre_matrix_)[0]
        order = np.argsort(-sims)
        return [self.movie_ids_[j] for j in order if self.movie_ids_[j] not in seen][:n]

    def item_scores(self, user_id: int) -> dict[int, float]:
        """Cosine similarity of every movie to the user's profile."""
        profile = self._user_profile(user_id)
        if profile is None or np.linalg.norm(profile) == 0:
            return {}
        sims = cosine_similarity(profile.reshape(1, -1), self.genre_matrix_)[0]
        return {m: float(s) for m, s in zip(self.movie_ids_, sims)}


class Hybrid:
    """Weighted blend of MF (collaborative) and content-based scores.

    Both score lists are min-max normalised per user, then combined:
    score = alpha * mf + (1 - alpha) * content. Alpha 0.7 favours the
    (usually stronger) collaborative signal while letting genres break
    ties and cover thin histories.
    """

    name = "hybrid"

    def __init__(self, mf: MatrixFactorization, cb: ContentBased, alpha: float = 0.7):
        self.mf = mf
        self.cb = cb
        self.alpha = alpha

    def fit(self, train: pd.DataFrame) -> "Hybrid":
        self.mf.fit(train)
        self.cb.fit(train)
        return self

    def recommend(self, user_id: int, n: int = 10, seen: set[int] | None = None) -> list[int]:
        seen = seen or set()
        if user_id not in self.mf.user_idx_:
            return []
        uu = self.mf.user_idx_[user_id]
        mf_scores = (self.mf.global_mean_ + self.mf.user_bias_[uu] + self.mf.item_bias_
                     + self.mf.item_factors_ @ self.mf.user_factors_[uu])
        mf_norm = _minmax(mf_scores)
        cb_scores = self.cb.item_scores(user_id)
        cb_arr = np.array([cb_scores.get(m, 0.0) for m in self.mf.items_])
        cb_norm = _minmax(cb_arr)
        combined = self.alpha * mf_norm + (1 - self.alpha) * cb_norm
        order = np.argsort(-combined)
        return [self.mf.items_[j] for j in order if self.mf.items_[j] not in seen][:n]


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
