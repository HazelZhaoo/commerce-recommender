import pickle
import numpy as np
import scipy.sparse as sp

from src.data import load_amazon
from src.split import prepare
from src.model import train_als
from src.baseline import popularity_ranking

def main():
    print("loading+ preparing...")
    df = load_amazon("data/amazon/CD.csv")
    data = prepare(df)

    print("trainig ALS...")
    model = train_als(data["matrix"])
    print("saving artifacts...")
    model.save("artifacts/als_model.npz")                        # the learned vectors
    sp.save_npz("artifacts/matrix.npz", data["matrix"])          # for seen-item exclusion
    np.save("artifacts/popularity.npy", popularity_ranking(data["matrix"]))  # cold fallback
    with open("artifacts/maps.pkl", "wb") as f:                  # id <-> index translation
        pickle.dump({"user_map": data["user_map"], "item_map": data["item_map"]}, f)
    print("done -> artifacts/")

if __name__ == "__main__":
    main()