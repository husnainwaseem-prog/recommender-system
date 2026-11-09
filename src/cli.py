"""CLI for the MovieLens Recommender System.

Usage:
    python -m src.cli download [--data-dir data]
    python -m src.cli evaluate [--data-dir data/ml-100k] [--out results/evaluation.json]
    python -m src.cli recommend --user 196 [--model hybrid] [--n 10]
    python -m src.cli new-user [--n 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cold_start import recommend_for_new_user  # noqa: E402
from src.data import ensure_dataset, load_movies, load_ratings, train_test_split  # noqa: E402
from src.evaluate import evaluate_all  # noqa: E402
from src.models import ContentBased, Hybrid, MatrixFactorization, Popularity  # noqa: E402


def _load(data_dir: str):
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    train, _ = train_test_split(ratings)
    return ratings, movies, train


def main() -> None:
    p = argparse.ArgumentParser(description="MovieLens Recommender System")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="Download MovieLens 100K")
    d.add_argument("--data-dir", default="data")

    e = sub.add_parser("evaluate", help="Train all models and score them")
    e.add_argument("--data-dir", default="data/ml-100k")
    e.add_argument("--out", default="results/evaluation.json")

    r = sub.add_parser("recommend", help="Recommend movies for a user")
    r.add_argument("--user", type=int, required=True)
    r.add_argument("--model", default="hybrid",
                   choices=["popularity", "matrix_factorization", "content_based", "hybrid"])
    r.add_argument("--n", type=int, default=10)
    r.add_argument("--data-dir", default="data/ml-100k")

    n = sub.add_parser("new-user", help="Recommendations for a user with no history")
    n.add_argument("--n", type=int, default=10)
    n.add_argument("--data-dir", default="data/ml-100k")

    args = p.parse_args()

    if args.cmd == "download":
        ensure_dataset(args.data_dir)
    elif args.cmd == "evaluate":
        from src.evaluate import main as eval_main
        eval_main(data_dir=args.data_dir, out_json=args.out)
    elif args.cmd == "recommend":
        ratings, movies, train = _load(args.data_dir)
        titles = movies.set_index("item_id")["title"].to_dict()
        seen = set(train[train["user_id"] == args.user]["item_id"])
        if args.model == "popularity":
            model = Popularity().fit(train)
        elif args.model == "matrix_factorization":
            model = MatrixFactorization().fit(train)
        elif args.model == "content_based":
            model = ContentBased(movies).fit(train)
        else:
            model = Hybrid(MatrixFactorization(), ContentBased(movies)).fit(train)
        recs = model.recommend(args.user, n=args.n, seen=seen)
        if not recs:
            recs = recommend_for_new_user(Popularity().fit(train), n=args.n)
            print(f"(user {args.user} has no history — showing popularity fallback)")
        print(f"Top {args.n} for user {args.user} [{args.model}]:")
        for i, item in enumerate(recs, 1):
            print(f"  {i:2d}. {titles.get(item, item)}")
    elif args.cmd == "new-user":
        _, _, train = _load(args.data_dir)
        movies = load_movies(args.data_dir)
        titles = movies.set_index("item_id")["title"].to_dict()
        recs = recommend_for_new_user(Popularity().fit(train), n=args.n)
        print(f"Top {args.n} for a brand-new user (popularity fallback):")
        for i, item in enumerate(recs, 1):
            print(f"  {i:2d}. {titles.get(item, item)}")


if __name__ == "__main__":
    main()
