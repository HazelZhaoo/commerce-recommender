from .data import build_index, build_user_item_matrix, filter_interactions
import pandas as pd

def temporal_split(df , test_frac = 0.2):
    """Cut events at a time quantile: train = past, test = future.

    One global cutoff (not per-user) mirrors how a recommender is really
    deployed -- trained today, asked to rank tomorrow -- so future events
    can never leak into training.
    """
    cutoff = df["timestamp"].quantile(1.0 - test_frac)
    train = df[df["timestamp"] < cutoff].copy()
    test = df[df["timestamp"] >= cutoff].copy()
    return train , test , cutoff

def build_eligibility(test , user_map , item_map):
    """Held-out ground-truth items per user, under train-only eligibility.

    Test interactions on items never seen in training are DROPPED -- no model
    has a vector for them (item cold-start), so scoring them is unfair noise.
    Warm users exist in the training index; cold users don't (popularity fallback later).
    """
    test = test[test["item"].isin(item_map)].copy()
    test["item_idx"] = test["item"].map(item_map)
    # Vectorized: one grouped aggregation instead of a Python loop over 160k+ groups.
    # -> a plain dict {raw_user_id: {item_idx, ...}}
    truth = test.groupby("user", sort=False)["item_idx"].agg(set).to_dict()
    # Then a cheap split into warm (in the train index) vs cold (unseen) users.
    warm_truth = {user_map[u]: items for u, items in truth.items() if u in user_map}
    cold_truth = {u: items for u, items in truth.items() if u not in user_map}
    return warm_truth, cold_truth

def seen_items_by_user(train, user_map , item_map):
    """Items each training user already touched -- recorded now so we can EXCLUDE
    them at recommend time (don't take credit for repeating history)."""
    d = train[train["user"].isin(user_map) & train["item"].isin(item_map)].copy()
    d["item_idx"] = d["item"].map(item_map)
    # Same vectorized trick: group -> set in one call, then remap user id -> row index.
    seen = d.groupby("user", sort=False)["item_idx"].agg(set).to_dict()
    return {user_map[u]: items for u, items in seen.items()}

def prepare(df, test_frac=0.2, min_user=5, min_item=5):
    """End to end: temporal split -> train-only filter -> index -> matrix + answer key."""
    train_raw, test_raw, cutoff = temporal_split(df, test_frac=test_frac)
    train = filter_interactions(train_raw, min_user=min_user, min_item=min_item)
    user_map, item_map = build_index(train)
    matrix = build_user_item_matrix(train, user_map, item_map)
    warm_truth, cold_truth = build_eligibility(test_raw, user_map, item_map)
    seen = seen_items_by_user(train, user_map, item_map)
     # A warm user's already-seen items are excluded at recommend time, so they can
    # never be recommended -> drop them from the truth set too (same principle as
    # cold-start eligibility: don't grade the model on what it structurally can't do).
    warm_truth = {u: items - seen.get(u, set()) for u, items in warm_truth.items()}
    warm_truth = {u: items for u, items in warm_truth.items() if items}  # drop now-empty
    return {
        "cutoff": cutoff, "user_map": user_map, "item_map": item_map,
        "matrix": matrix, "warm_truth": warm_truth,
        "cold_truth": cold_truth, "seen": seen,
    }

if __name__ == "__main__":
    from .data import load_events
    print("loading events...")
    df = load_events("data/events.csv")
    data = prepare(df)
    n_users, n_items = data["matrix"].shape
    print(f"cutoff time     : {pd.to_datetime(data['cutoff'], unit='ms')}")
    print(f"train matrix    : {n_users:,} users x {n_items:,} items")
    print(f"interactions    : {data['matrix'].nnz:,} nonzero cells")
    print(f"warm test users : {len(data['warm_truth']):,}")
    print(f"cold test users : {len(data['cold_truth']):,}")
