"""Data loading, implicit-feedback weighting, filtering, and matrix building.

RetailRocket events: (timestamp, visitorid, event, itemid, transactionid).
We treat views/carts/purchases as *positive* implicit feedback of increasing
strength — a missing (user, item) pair means "not observed", never "disliked".
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Event weights are hypotheses, not universal truth (see README). Recorded here
# so the whole experiment is reproducible and easy to sweep.
EVENT_WEIGHTS = {"view": 1.0, "addtocart": 2.0, "transaction": 3.0}


def load_events(path, event_weights=EVENT_WEIGHTS):
    df = pd.read_csv(path, usecols=["timestamp", "visitorid", "event", "itemid"])
    df = df[df["event"].isin(event_weights)].copy()
    df["weight"] = df["event"].map(event_weights).astype("float32")
    df = df.rename(columns={"visitorid": "user", "itemid": "item"})
    return df[["timestamp", "user", "item", "event", "weight"]]


def filter_interactions(df, min_user=5, min_item=5, iterations=2):
    """Keep users and items with at least `min_*` interactions.

    Iterated a couple of times because dropping sparse items can push a user
    below the threshold and vice versa (approximate k-core).
    """
    for _ in range(iterations):
        item_counts = df["item"].value_counts()
        df = df[df["item"].isin(item_counts[item_counts >= min_item].index)]
        user_counts = df["user"].value_counts()
        df = df[df["user"].isin(user_counts[user_counts >= min_user].index)]
    return df.copy()


def build_index(df):
    """Map raw user/item ids -> contiguous matrix indices. Built on TRAIN only,
    so the model can never represent an item it didn't see in training."""
    users = df["user"].unique()
    items = df["item"].unique()
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {it: i for i, it in enumerate(items)}
    return user_map, item_map


def build_user_item_matrix(df, user_map, item_map):
    """CSR (users x items) with confidence = summed interaction weight."""
    d = df[df["user"].isin(user_map) & df["item"].isin(item_map)]
    g = d.groupby(["user", "item"], sort=False)["weight"].sum().reset_index()
    rows = g["user"].map(user_map).to_numpy()
    cols = g["item"].map(item_map).to_numpy()
    vals = g["weight"].to_numpy(dtype="float32")
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(len(user_map), len(item_map)))
    return mat
