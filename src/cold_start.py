"""Cold start: what to do for users (and items) with no history.

New user (no ratings at all):
  The collaborative and content models have nothing to score from —
  MF has no latent factors for the user, content has no profile.
  Fall back to the popularity baseline: the most-liked movies overall.
  This is the standard, honest answer: until we know anything about
  a user, the best guess is what most people liked. As soon as the
  user rates even one movie, the content model can use its genres,
  and after a few ratings MF takes over.

  If the product asks the user for a few favourite genres or movies
  on signup (onboarding), those seed the content profile immediately
  — see `recommend_for_new_user` below.

New item (no ratings yet):
  MF cannot rank it (no factors, no bias). Content-based can: the
  movie has genres, so it enters the candidate pool for users whose
  profile matches those genres. The hybrid therefore surfaces new
  items to the right users, while pure MF never would.

Implementation: `recommend_for_new_user` and `recommend` wrappers
that route unknown users to popularity.
"""

from __future__ import annotations

import pandas as pd

from .models import ContentBased, Popularity


def recommend_for_new_user(popularity: Popularity, n: int = 10) -> list[int]:
    """Top-n most-liked movies for a user with zero history."""
    return popularity.recommend(user_id=-1, n=n, seen=set())


def recommend_with_genre_seed(cb: ContentBased, liked_genres: list[str],
                              n: int = 10, seen: set[int] | None = None) -> list[int]:
    """Onboarding path: the user picked favourite genres at signup.
    Build a profile from those genre flags and rank unseen movies."""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    profile = np.array([1.0 if g in liked_genres else 0.0 for g in cb.genre_cols_])
    if profile.sum() == 0:
        return []
    sims = cosine_similarity(profile.reshape(1, -1), cb.genre_matrix_)[0]
    order = np.argsort(-sims)
    seen = seen or set()
    return [cb.movie_ids_[j] for j in order if cb.movie_ids_[j] not in seen][:n]
