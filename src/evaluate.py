import numpy as np

def recall_at_k(recommended, relevant, k=10):
    if not relevant:
        return None
    top_k = recommended[:k]
    hits = 0                      
    for item in relevant:
        if item in top_k:
            hits += 1
    return hits / len(relevant)


def hit_rate_at_k(recommended , relevant , k = 10):
    if not relevant:
        return None
    top_k = recommended[:k]
    for item in relevant:
        if item in top_k:
            return 1.0

    return 0.0

def ndcg_at_k(recommended , relevant , k = 10):
    if not relevant:
        return None
    top_k = recommended[:k]
    dcg = 0.0
    for rank , item in enumerate(top_k):
        if item in relevant:
            dcg += 1.0/np.log2(rank + 2)
    
    idcg = 0.0
    ideal_hits = min(len(relevant) , k)
    for rank in range(ideal_hits):
        idcg += 1.0/np.log2(rank + 2)
    
    return dcg/idcg

def evaluate(rec , truth , n_items , k = 10):
    """Average ranking metrics over all users, given precomputed top-k recs.
    recommendations: {user: ranked list of item indices}  (their top-k)
    truth:           {user: set of true future item indices}  (the answer key)
    n_items:         catalog size (for coverage)
    """
    recalls , hits , ndcgs = [] , [] , []
    recommended_times = set()
    for user , relevant in truth.items():
        if not relevant:
            continue
            
        recs = rec.get(user , [])
        recalls.append(recall_at_k(recs , relevant , k))
        hits.append(hit_rate_at_k(recs , relevant , k))
        ndcgs.append(ndcg_at_k(recs , relevant , k))
        recommended_times.update(recs[:k])
    return {
        f"recall@{k}":   float(np.mean(recalls)),
        f"hit_rate@{k}": float(np.mean(hits)),
        f"ndcg@{k}":     float(np.mean(ndcgs)),
        "coverage":      len(recommended_times) / n_items,
        "n_users":       len(recalls),
    }

if __name__ == "__main__":
    from .data import load_events
    from .split import prepare
    from .baseline import popularity_ranking, recommend_popular   # note: baseline (singular)

    print("loading + preparing data...")
    df = load_events("data/events.csv")
    data = prepare(df)

    ranking = popularity_ranking(data["matrix"])
    n_items = data["matrix"].shape[1]

    # WARM users: popularity recs, excluding what they've already seen
    warm_recs = {u: recommend_popular(ranking, data["seen"].get(u, set()), 10)
                 for u in data["warm_truth"]}
    print("\n=== WARM users (returning) ===")
    print(evaluate(warm_recs, data["warm_truth"], n_items, k=10))

    # COLD users: popularity fallback (no history to exclude)
    cold_recs = {u: recommend_popular(ranking, set(), 10)
                 for u in data["cold_truth"]}
    print("\n=== COLD users (fallback) ===")
    print(evaluate(cold_recs, data["cold_truth"], n_items, k=10))