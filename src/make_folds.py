"""Assign 5 cross-validation folds by story, over train + val only.

Every pair from one story lands in the same fold, so a model is never tested
on a story it trained on. Stories are shuffled with a fixed seed, then dealt
into folds so each fold gets roughly the same number of pairs. The test split
is left out entirely.

Usage: python -m src.make_folds
"""

import numpy as np
import pandas as pd

IN_FILE = "data/processed/pairs_significant.csv"
OUT_FILE = "data/processed/pairs_cv.csv"


def assign_folds(pairs: pd.DataFrame, k: int = 5, seed: int = 42) -> pd.DataFrame:
    """Return the non-test pairs with a `fold` column (0 to k-1), grouped by story."""
    dev = pairs[pairs["split"] != "test"].copy()
    sizes = dev.groupby("story_group").size().sort_index()
    rng = np.random.default_rng(seed)
    order = sizes.index[rng.permutation(len(sizes))]
    # Greedy balancing: put each story in the fold that currently has the fewest pairs.
    load, fold_of = np.zeros(k), {}
    for story in order:
        f = int(np.argmin(load))
        fold_of[story] = f
        load[f] += sizes[story]
    dev["fold"] = dev["story_group"].map(fold_of)
    return dev.reset_index(drop=True)


if __name__ == "__main__":
    pairs = pd.read_csv(IN_FILE)
    cv = assign_folds(pairs)
    cv.to_csv(OUT_FILE, index=False)
    print(f"Saved {OUT_FILE}: {len(cv)} pairs, {cv['story_group'].nunique()} stories")
    print(cv.groupby("fold").agg(pairs=("label", "size"), stories=("story_group", "nunique")).to_string())
