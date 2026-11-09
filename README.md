# MovieLens Recommender System

A movie recommender built on the public **MovieLens 100K** dataset
(100,000 ratings, 943 users, 1,682 movies). Four models, one
evaluation: a popularity baseline, a collaborative-filtering model
(matrix factorization), a content-based model (genre similarity),
and a hybrid of the two — scored with ranking metrics
(Precision@K, Recall@K, NDCG@K) on a proper per-user train/test split.

## Results (K = 10, relevant = rating ≥ 4)

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularity (baseline) | 0.0791 | 0.0770 | 0.1026 |
| Matrix factorization (CF) | 0.0484 | 0.0439 | 0.0591 |
| Content-based (genres) | 0.0184 | 0.0167 | 0.0226 |
| Hybrid (MF + content) | 0.0485 | 0.0442 | 0.0629 |

Popularity wins the offline ranking metrics — this is the well-known
**popularity bias** of offline evaluation (see REPORT.md §4): the test
set only contains movies the user actually watched, and popular movies
dominate that. The CF model still shows real signal: it ranks a user's
relevant test items above random unseen items 68.8% of the time
(50% = chance), and its recommendations are sensible and personalized
(e.g. a drama lover gets acclaimed dramas, not just the global hits).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt

# download the dataset (MovieLens 100K, ~5 MB)
python -m src.cli download

# train all four models and evaluate
python -m src.cli evaluate

# recommend for a user
python -m src.cli recommend --user 196 --model hybrid --n 10

# recommendations for a brand-new user (cold start)
python -m src.cli new-user --n 10
```

## Project structure

```
recommender-system/
├── src/
│   ├── data.py          # load MovieLens 100K, per-user train/test split
│   ├── models.py        # Popularity, MatrixFactorization, ContentBased, Hybrid
│   ├── metrics.py       # Precision@K, Recall@K, NDCG@K
│   ├── evaluate.py      # train + score all models
│   ├── cold_start.py    # new-user / new-item handling
│   └── cli.py           # download / evaluate / recommend / new-user
├── data/ml-100k/        # MovieLens 100K (downloaded by the CLI)
├── results/evaluation.json
├── tests/test_recommender.py   # 13 tests
├── README.md
└── REPORT.md            # design, results, cold start, limitations
```

## Models

- **Popularity** — rank by number of 4+ star ratings, tie-break by mean
  rating. The baseline every other model must beat.
- **Matrix factorization (CF)** — latent factors (20 dims) learned with
  SGD from scratch in numpy: rating ≈ global mean + user bias +
  item bias + user factors · item factors. Test RMSE 0.98.
- **Content-based** — each user gets a genre profile (average of the
  genres of movies they rated ≥ 4, weighted by rating − 3); unseen
  movies are ranked by cosine similarity to that profile.
- **Hybrid** — min-max normalised MF score blended with content score
  (α = 0.7): `0.7 × MF + 0.3 × content`.

## Cold start

- **New user, no history** → popularity fallback (the most-liked
  movies overall). If the user picks favourite genres at signup, the
  content model builds a profile from those immediately.
- **New item, no ratings** → MF cannot rank it, but content-based can
  (it has genres), so the hybrid still surfaces it to the right users.

See `REPORT.md` for the full write-up.
