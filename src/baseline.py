import numpy as np

def popularity_ranking(matrix):
    """Rank items by how many DISTINCT users interacted with them (train popularity).

    A count of distinct users -- not total interactions -- so a few power users
    spamming one item can't fake broad appeal. Returns item indices, most popular first.
    """
    # matrix is CSR (users x items). (matrix > 0) turns any interaction into True/1,
    # then summing down each column counts how many users touched that item.
    item_pop = np.asarray((matrix >0).sum(axis = 0)).ravel()
    ranking = np.argsort(-item_pop)
    return ranking

def recommend_popular(ranking , seen_items , k = 10):
    """Top-k most-popular items for one user, skipping ones they've already seen.

    `ranking` is the most->least popular item list from popularity_ranking().
    `seen_items` is that user's already-touched set (empty for cold users).
    """
    recs = []
    for item in ranking:
        if item not in seen_items:
            recs.append(item)
            if len(recs) >= k:
                break
    return recs

if __name__ == "__main__":
    from .data import load_events
    from .split import prepare

    print("loading + preparing data...")
    df = load_events("data/events.csv")
    data = prepare(df)
    ranking = popularity_ranking(data["matrix"])

    # ranking holds matrix indices; translate back to real RetailRocket item ids to read them
    idx_to_item = {idx: item for item, idx in data["item_map"].items()}
    print("\nTop 10 most popular items:")
    for rank, idx in enumerate(ranking[:10], start=1):
        print(f"  {rank:2d}. item {idx_to_item[idx]}")

    # now recommend for one real warm user, excluding what they've already seen
    sample_user = next(iter(data["warm_truth"]))     # grab any warm user's row index
    seen = data["seen"].get(sample_user, set())
    recs = recommend_popular(ranking, seen, k=10)
    print(f"\nSample warm user (row {sample_user}) has already seen {len(seen)} items.")
    print("Their top-10 popularity recommendations:", recs)