# Commerce Recommendation & Ranking System

An implicit-feedback product recommender built with a **leakage-safe evaluation protocol**, an honest **baseline-vs-personalization** benchmark, and an **end-to-end serving layer** (FastAPI + a searchable web demo). Validated on **two independent datasets**.

## Problem

Can a personalized recommender produce more relevant top-10 rankings than a non-personalized **popularity baseline** — while keeping useful catalog coverage — on real, timestamped behavior?

The signal here isn't just calling a library; it's the **defensible choices**: temporal validation, leakage prevention, honest baselines, ranking metrics, cold-start handling, and shipping the model behind an API.

## Approach

- **Data (2 datasets):**
  - [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) — 2.7M timestamped events (view / cart / transaction).
  - [Amazon Reviews 2023 — CDs & Vinyl](https://amazon-reviews-2023.github.io/) — 4.8M ratings (explicit 1–5 → implicit positives at ≥4 stars).
- **Leakage-safe temporal split:** one global time cutoff — train on the past, evaluate on the future. Train-only item eligibility; already-seen items excluded from both recommendations and ground truth.
- **Models:** global popularity **baseline** vs. **ALS** matrix factorization (`implicit`).
- **Metrics:** Recall@10, NDCG@10, Hit Rate@10, catalog coverage — evaluated separately for **warm** (returning) users and **cold-start** users (popularity fallback).

## Results — Popularity baseline vs. ALS (warm users, leakage-safe temporal split)

**RetailRocket** (n = 2,174 warm users):

| Metric | Popularity | ALS | Lift |
|---|---:|---:|:--:|
| Recall@10 | 0.0077 | 0.0316 | **4.1×** |
| Hit Rate@10 | 0.0216 | 0.0727 | **3.4×** |
| NDCG@10 | 0.0047 | 0.0238 | **5.1×** |
| Coverage (distinct items recommended) | ~31 | ~1,346 | **43×** |

**Amazon CDs & Vinyl** (n = 11,548 warm users):

| Metric | Popularity | ALS | Lift |
|---|---:|---:|:--:|
| Recall@10 | 0.0021 | 0.0075 | **3.6×** |
| Hit Rate@10 | 0.0060 | 0.0230 | **3.9×** |
| NDCG@10 | 0.0016 | 0.0052 | **3.3×** |
| Coverage (distinct items recommended) | ~16 | ~1,592 | **~100×** |

**Takeaways.** ALS beats popularity across the board on *both* datasets by a consistent ~3–5× on ranking metrics and ~40–100× on coverage — popularity shows everyone the same handful of bestsellers, while ALS actually personalizes. Absolute numbers are modest by design: the split is strictly leakage-safe and the data is sparse, so these reflect *honest* offline performance, not inflated metrics.

## Serving layer — from model to system

The trained model is served behind a **FastAPI** app, with a lightweight web demo:

- `GET /recommend/{user_id}` — personalized top-K for a user (warm → ALS, unknown → popularity fallback).
- `GET /search?q=` — find albums/products by title or artist (readable input).
- `GET /similar/{id}` — item-to-item "customers who liked this also liked…" (ALS item vectors).
- `GET /app` — a searchable, image-rich demo UI (browse by cover art).

Offline training (`scripts/train.py`) saves model artifacts; the API loads them at startup and serves in milliseconds — the standard offline-train / online-serve split.

## Directory structure

```
commerce-recommender/
├── src/
│   ├── data.py        # loaders (RetailRocket events + Amazon ratings), filter, sparse matrix
│   ├── split.py       # leakage-safe temporal split, eligibility, seen-item sets
│   ├── baseline.py    # popularity recommender
│   ├── model.py       # ALS matrix factorization + recommendation
│   └── evaluate.py    # Recall@10 / NDCG@10 / Hit Rate@10 / coverage
├── scripts/
│   ├── train.py       # offline: train ALS, save artifacts
│   └── build_meta.py  # build item metadata lookup (title / artist / image) for the demo
├── serve/app.py       # FastAPI: /recommend, /search, /similar, /app
├── web/index.html     # searchable album-cover demo UI
├── artifacts/         # saved model + maps + metadata (gitignored)
└── data/              # datasets (gitignored — downloaded separately)
```

## Running

```bash
pip install -r requirements.txt
python -m scripts.train          # train + save artifacts (offline)
python -m scripts.build_meta     # build the metadata lookup for the demo
uvicorn serve.app:app --reload   # serve the API + demo at http://127.0.0.1:8000/app
```

## Next

- [ ] Dockerize training + serving for one-command reproducibility.
- [ ] Metric + split correctness tests (`tests/`).
- [ ] Request-latency logging / basic monitoring.

---

*Datasets: RetailRocket (Kaggle, CC BY-NC-SA 4.0) and Amazon Reviews 2023 (McAuley Lab, UCSD). Modeling with the [`implicit`](https://github.com/benfred/implicit) library.*
