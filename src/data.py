"""Load MovieLens 100K and split it into train / test.

Split: per-user holdout. Each user's ratings are sorted by timestamp
and the most recent 20% go to test, the rest to train. This mirrors
real use (predict what a user will like next) and guarantees every
test user also appears in train — so the split measures ranking
quality, not cold start. Cold start is handled separately (see
cold_start.py).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ml-100k"


def load_ratings(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """All 100,000 ratings: user_id, item_id, rating, timestamp."""
    path = Path(data_dir) / "u.data"
    return pd.read_csv(
        path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
    )


def load_movies(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Movies with title and 19 binary genre flags."""
    path = Path(data_dir) / "u.item"
    cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
    df = pd.read_csv(path, sep="|", names=cols, encoding="latin-1", usecols=["item_id", "title"] + GENRES)
    return df


def train_test_split(ratings: pd.DataFrame, test_frac: float = 0.2,
                     seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user holdout: each user's latest `test_frac` ratings -> test."""
    rng = np.random.RandomState(seed)
    train_parts, test_parts = [], []
    for _, group in ratings.groupby("user_id"):
        group = group.sort_values("timestamp")
        n_test = max(1, int(round(len(group) * test_frac)))
        test_parts.append(group.iloc[-n_test:])
        train_parts.append(group.iloc[:-n_test])
    train = pd.concat(train_parts).sample(frac=1, random_state=rng).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    return train, test


def ensure_dataset(data_dir: str | Path = "data") -> Path:
    """Download + extract MovieLens 100K if not already present."""
    data_dir = Path(data_dir)
    target = data_dir / "ml-100k" / "u.data"
    if target.exists():
        return target.parent
    import urllib.request
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = data_dir / "ml-100k.zip"
    print(f"Downloading MovieLens 100K from {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)
    print(f"Extracted to {target.parent}")
    return target.parent
