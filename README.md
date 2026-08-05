# Commerce Recommendation & Ranking System

> 🚧 **In progress.** An implicit-feedback product recommender on real e-commerce behavior, built with a leakage-safe evaluation protocol and an honest baseline-vs-personalization comparison.

## Problem

Can a personalized recommender produce more relevant top-10 product rankings than a non-personalized **popularity baseline** — while keeping useful catalog coverage — on real, timestamped commerce behavior?

The signal here isn't just calling an algorithm library; it's the **defensible choices**: temporal validation, leakage prevention, baselines, ranking metrics, and cold-start handling.

## Approach

- **Data:** [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) — 2.7M timestamped events (views, add-to-cart, transactions).
- **Implicit feedback:** views / carts / purchases as graded positive signals (a missing pair means *"not observed"*, never *"disliked"*).
- **Leakage-safe temporal split:** train only on past behavior; evaluate on future held-out interactions.
- **Models:** global popularity **baseline** vs. **ALS** matrix factorization (`implicit`).
- **Metrics:** Recall@10, NDCG@10, Hit Rate@10, catalog coverage — plus cold-start / sparse-history slices.

## Directory structure

```
commerce-recommender/
├── README.md
├── requirements.txt
├── src/
│   ├── data.py         # load, weight, filter, build sparse matrix
│   ├── split.py        # leakage-safe temporal split + eligibility
│   ├── baselines.py    # popularity recommender
│   ├── models.py       # ALS (and optional BPR)
│   └── evaluate.py     # ranking metrics + coverage
├── scripts/
│   └── train.py        # reproducible experiment entry point
├── tests/              # metric + split correctness tests
├── artifacts/          # results.csv, comparison outputs
└── data/               # RetailRocket (gitignored — download separately)
```

## Getting the data

```bash
pip install kaggle          # requires a Kaggle API token at ~/.kaggle/kaggle.json
kaggle datasets download -d retailrocket/ecommerce-dataset -p data --unzip
```

## Running (once complete)

```bash
pip install -r requirements.txt
python scripts/train.py
```

## Status

- [x] Data loading + implicit-feedback weighting
- [ ] Leakage-safe temporal split
- [ ] Popularity baseline
- [ ] ALS model
- [ ] Ranking-metric evaluation harness
- [ ] Cold-start / sparsity analysis
- [ ] Results table + write-up

## Results

*Measured results and the baseline-vs-ALS comparison will be added here once the evaluation harness is complete. Numbers will reflect only what has actually been run.*

---

*Dataset: RetailRocket (via Kaggle, CC BY-NC-SA 4.0). Modeling with the [`implicit`](https://github.com/benfred/implicit) library.*
