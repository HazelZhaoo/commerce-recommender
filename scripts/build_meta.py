"""Build a lightweight metadata lookup {asin: {title, artist, image}} for the
items the model actually knows about. Reads Table 2 (product details) once and
keeps only the trained items, so the API can turn asins into names + cover art.
"""
import gzip
import json
import pickle

META_PATH = "data/amazon/meta_CDs_and_Vinyl.jsonl.gz"


def main():
    with open("artifacts/maps.pkl", "rb") as f:
        item_map = pickle.load(f)["item_map"]
    wanted = set(item_map)                       # only items the model was trained on

    meta = {}
    with gzip.open(META_PATH, "rt") as f:
        for line in f:
            d = json.loads(line)
            asin = d.get("parent_asin")
            if asin not in wanted:
                continue
            imgs = d.get("images") or []
            image = None
            if imgs:
                image = imgs[0].get("large") or imgs[0].get("thumb")
            meta[asin] = {
                "title": d.get("title"),
                "artist": (d.get("store") or "").split("Format:")[0].strip(),
                "image": image,
            }
    with open("artifacts/meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"saved metadata for {len(meta):,} items -> artifacts/meta.pkl")


if __name__ == "__main__":
    main()
