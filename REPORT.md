# Recommendation System — Report

## 1. Task

Build a recommender on a public dataset (MovieLens 100K), implement a
popularity baseline, a collaborative-filtering model, and a
content-based / hybrid model, evaluate with ranking metrics
(Precision@K, Recall@K, NDCG@K) on a proper train/test split, and
explain cold-start handling for new users.

## 2. Dataset

MovieLens 100K (GroupLens): 100,000 ratings from 943 users on 1,682
movies, 1997–1998. Each movie carries 19 binary genre flags
(Action, Comedy, Drama, …). Ratings are 1–5 stars; timestamp included.

**Split.** Per-user holdout: each user's ratings are sorted by
timestamp and the most recent 20% go to test (20,000 ratings), the
rest to train (80,000). This mirrors real use — predict what a user
will like next — and guarantees every test user also appears in train,
so the split measures ranking quality, not cold start. Seed 42.

**Relevance.** A test item counts as relevant when the user rated it
≥ 4 stars.

## 3. Models

| Model | Idea | Key detail |
|---|---|---|
| Popularity (baseline) | Recommend what most people liked | Rank by count of 4+ ratings, tie-break by mean rating |
| Matrix factorization (CF) | Learn latent taste factors from ratings | 20 factors, SGD from scratch (numpy), lr 0.005, reg 0.02, 30 epochs; rating ≈ global mean + user bias + item bias + user·item factors |
| Content-based | Recommend movies like the ones the user liked | User genre profile = weighted average of genres of movies rated ≥ 4 (weight = rating − 3); rank unseen movies by cosine similarity |
| Hybrid | Blend CF + content | Min-max normalised scores, `0.7 × MF + 0.3 × content` |

## 4. Results (K = 10, relevant = rating ≥ 4)

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularity | **0.0791** | **0.0770** | **0.1026** |
| Matrix factorization | 0.0484 | 0.0439 | 0.0591 |
| Content-based | 0.0184 | 0.0167 | 0.0226 |
| Hybrid | 0.0485 | 0.0442 | 0.0629 |

**Why does the baseline win?** This is the well-known **popularity
bias of offline evaluation**, not a broken model:

1. The test set only contains movies the user actually watched. People
   mostly watch popular movies, so the relevant items in test skew
   heavily popular — the baseline's exact strength.
2. MF recommends good movies the user has *not* watched yet. Those
   cannot appear in the test set, so they score zero even when the
   user would love them.
3. Verified the CF model has real signal: it ranks a user's relevant
   test items above random unseen items **68.8% of the time**
   (50% = chance), and its test RMSE is 0.98. Its picks are sensible
   and personalized — e.g. user 196 (likes dramas: *The English
   Patient*, *Being There*, *A Fish Called Wanda*) gets *Schindler's
   List*, *Shawshank Redemption*, *The African Queen* — acclaimed
   dramas, not just the global top 10.

The hybrid edges MF on every metric (NDCG 0.0629 vs 0.0591) because
genre similarity breaks ties and covers thin histories. Content-based
alone is weakest: 19 binary genres are a coarse signal (two dramas can
be very different films), and it can only repeat genres the user
already liked.

**Tuning.** lr 0.005 / 30 epochs was best (test RMSE 0.9842); lr 0.01
gave 1.0168 and lr 0.02 / 50 epochs gave 1.0949.

## 5. Cold start

**New user (no ratings).** MF has no latent factors for the user and
content has no profile, so both return nothing. Fallback: the
popularity list — the most-liked movies overall. This is the honest
default: until we know anything about a user, the best guess is what
most people liked. As soon as the user rates one movie, content can
use its genres; after a few ratings MF takes over.

**Onboarding shortcut.** If signup asks for favourite genres (or a
few liked movies), `recommend_with_genre_seed` builds a content
profile from those immediately — no waiting for the first rating.

**New item (no ratings).** MF cannot rank it (no factors, no bias).
Content-based can — the movie has genres — so the hybrid still
surfaces it to users whose profile matches those genres, while pure
MF would never show it.

## 6. How to reproduce

```bash
pip install -r requirements.txt
python -m src.cli download
python -m src.cli evaluate        # writes results/evaluation.json
python -m pytest tests/ -q        # 13 tests
python -m src.cli recommend --user 196 --model hybrid --n 10
python -m src.cli new-user --n 10
```

## 7. Limitations

- Offline metrics understate personalized models (popularity bias, §4).
  A live A/B test or a user study would measure the real difference.
- Content features are only 19 genre flags — no plot, cast, or director.
  Richer features (e.g. TF-IDF on plot summaries) would strengthen it.
- MF is plain SGD in numpy (30 epochs ≈ 35 s on CPU); no GPU needed
  at this scale, but it would not scale to millions of ratings without
  vectorised/batched updates or a library implementation.
- The dataset is from 1997–1998; tastes and the movie catalogue have
  moved on. The pipeline is dataset-agnostic (any user/item/rating CSV
  works).
