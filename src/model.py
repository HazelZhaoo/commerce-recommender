from implicit.als import AlternatingLeastSquares


def train_als(matrix, factors=64, regularization=0.05, iterations=15, seed=42):
    """Train ALS matrix factorization on the confidence matrix.

    The library does the alternating least-squares math; we choose the knobs.
    """
    model = AlternatingLeastSquares(
        factors=factors,               # size of each user/item vector (the "k" latent factors)
        regularization=regularization, # the ridge penalty -> stops vectors blowing up
        iterations=iterations,         # how many alternating passes (user<->item)
        random_state=seed,             # reproducibility (same result every run)
    )
    model.fit(matrix)                  # learns user + item vectors from your matrix
    return model

def als_recommendations(model, matrix, users, k=10):
    """Generate {user: top-k item indices} from the trained ALS model.

    filter_already_liked_items=True does the seen-item exclusion for us,
    using each user's training row -> same Guard 3 as the baseline.
    """
    user_list = list(users)
    ids, _ = model.recommend(
        user_list, matrix[user_list], N=k,
        filter_already_liked_items=True,
    )
    return {u: list(items) for u, items in zip(user_list, ids)}

if __name__ == "__main__":
    from .data import load_events
    from .split import prepare
    from .baseline import popularity_ranking, recommend_popular
    from .evaluate import evaluate

    df = load_events("data/events.csv")
    data = prepare(df)
    n_items = data["matrix"].shape[1]
    warm_users = list(data["warm_truth"])

    # --- ALS ---
    model = train_als(data["matrix"])
    als_recs = als_recommendations(model, data["matrix"], warm_users, k=10)

    # --- Popularity baseline (same warm users, fair comparison) ---
    ranking = popularity_ranking(data["matrix"])
    pop_recs = {u: recommend_popular(ranking, data["seen"].get(u, set()), 10)
                for u in warm_users}

    print("\n=== WARM users: Popularity vs ALS ===")
    print("Popularity:", evaluate(pop_recs, data["warm_truth"], n_items, k=10))
    print("ALS       :", evaluate(als_recs, data["warm_truth"], n_items, k=10))
