"""Is model A really better than model B, or is the gap just noise?

Compares two prediction files on the same pairs with two tests:

1. McNemar's test: looks only at pairs where the models disagree, and asks
   whether one model is right in those cases more often than chance.
2. Paired bootstrap over story groups: resamples whole stories with
   replacement (pairs from one story are not independent) and gives a 95%
   confidence interval for the accuracy difference.

Usage:
    python -m src.significance data/processed/deberta_val_predictions.csv \
        data/processed/embedding_val_predictions.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest

KEY = ["test_id", "headline_a", "headline_b"]


def align(file_a: str, file_b: str) -> pd.DataFrame:
    """Join two prediction files on the pair, checking they cover the same pairs."""
    a = pd.read_csv(file_a)[KEY + ["story_group", "label", "correct"]]
    b = pd.read_csv(file_b)[KEY + ["correct"]]
    merged = a.merge(b, on=KEY, suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError(f"Files cover different pairs: {len(a)}, {len(b)}, matched {len(merged)}")
    return merged


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on the pairs where exactly one model is right."""
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n > 0 else 1.0
    return {"only_a_right": only_a, "only_b_right": only_b, "p_value": float(p)}


def bootstrap_difference(df: pd.DataFrame, n_boot: int = 5000, seed: int = 0) -> dict:
    """95% CI for accuracy(A) minus accuracy(B), resampling story groups."""
    df = df.assign(ca=df["correct_a"].astype(int), cb=df["correct_b"].astype(int))
    groups = df.groupby("story_group").agg(ca=("ca", "sum"), cb=("cb", "sum"), n=("ca", "size"))
    sums = groups.to_numpy()
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s = sums[rng.integers(0, len(sums), len(sums))].sum(axis=0)
        diffs[i] = (s[0] - s[1]) / s[2]
    observed = (df["ca"].sum() - df["cb"].sum()) / len(df)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"difference": float(observed), "ci_low": float(low), "ci_high": float(high)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_a", help="Prediction CSV of the first model")
    parser.add_argument("model_b", help="Prediction CSV of the second model")
    args = parser.parse_args()

    df = align(args.model_a, args.model_b)
    acc_a, acc_b = df["correct_a"].mean(), df["correct_b"].mean()
    m = mcnemar(df["correct_a"].to_numpy(bool), df["correct_b"].to_numpy(bool))
    b = bootstrap_difference(df)

    print(f"Pairs compared: {len(df)}")
    print(f"Accuracy A: {acc_a:.3f}   ({args.model_a})")
    print(f"Accuracy B: {acc_b:.3f}   ({args.model_b})")
    print(f"\nMcNemar: A right and B wrong on {m['only_a_right']} pairs, "
          f"B right and A wrong on {m['only_b_right']}; p = {m['p_value']:.4f}")
    print(f"Paired bootstrap (by story): A minus B = {b['difference']:+.3f}, "
          f"95% CI [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]")
    verdict = "significant" if m["p_value"] < 0.05 and (b["ci_low"] > 0 or b["ci_high"] < 0) else "not significant"
    print(f"\nVerdict at the 5% level: {verdict}")


if __name__ == "__main__":
    main()
