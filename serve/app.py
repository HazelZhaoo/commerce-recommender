import pickle
import numpy as np
import scipy.sparse as sp
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from implicit.cpu.als import AlternatingLeastSquares

app = FastAPI(title="Commerce Recommender")

# ---- PART 1: load artifacts ONCE at startup (offline work, done ahead of time) ----
model = AlternatingLeastSquares.load("artifacts/als_model.npz")
matrix = sp.load_npz("artifacts/matrix.npz")
popularity = np.load("artifacts/popularity.npy")
with open("artifacts/maps.pkl", "rb") as f:
    maps = pickle.load(f)
user_map = maps["user_map"]
item_map = maps["item_map"]
idx_to_item = {idx: item for item, idx in item_map.items()}   # flip: index -> real item id
with open("artifacts/meta.pkl", "rb") as f:                   # asin -> {title, artist, image}
    meta = pickle.load(f)


# ---- PART 2: the endpoints (online work, per request) ----
@app.get("/")
def health():
    """A quick 'is it alive?' check."""
    return {"status": "ok", "users": len(user_map), "items": len(item_map)}


@app.get("/app", response_class=HTMLResponse)
def frontend():
    """Serve the album-cover demo UI (same origin as the API, so no CORS)."""
    with open("web/index.html") as f:
        return f.read()


@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = 10):
    """Recommend top-k items for a user. Warm -> ALS, cold -> popularity fallback."""
    if user_id in user_map:                          # WARM user -> personalized ALS
        row = user_map[user_id]
        ids, _ = model.recommend(row, matrix[row], N=k, filter_already_liked_items=True)
        strategy = "als"
    else:                                            # COLD user -> popularity fallback
        ids = popularity[:k]
        strategy = "popularity"
    items = []                                       # indices -> rich product cards
    for i in ids:
        asin = idx_to_item[i]
        info = meta.get(asin, {})
        items.append({
            "asin": asin,
            "title": info.get("title"),
            "artist": info.get("artist"),
            "image": info.get("image"),
        })
    return {"user_id": user_id, "strategy": strategy, "recommendations": items}


def _card(asin):
    info = meta.get(asin, {})
    return {"asin": asin, "title": info.get("title"), "artist": info.get("artist"), "image": info.get("image")}


@app.get("/search")
def search(q: str, k: int = 12):
    """Find albums whose title or artist contains the query text (readable input)."""
    ql = q.lower().strip()
    hits = []
    for asin, info in meta.items():
        title = (info.get("title") or "").lower()
        artist = (info.get("artist") or "").lower()
        if ql in title or ql in artist:
            hits.append(_card(asin))
            if len(hits) >= k:
                break
    return {"query": q, "results": hits}


@app.get("/similar/{asin}")
def similar(asin: str, k: int = 12):
    """Item-to-item: given an album, return the most similar albums (ALS item vectors)."""
    if asin not in item_map:
        return {"seed": None, "results": []}
    row = item_map[asin]
    ids, _ = model.similar_items(row, N=k + 1)         # +1 because the album itself is included
    results = [_card(idx_to_item[i]) for i in ids if idx_to_item[i] != asin][:k]
    return {"seed": _card(asin), "results": results}
