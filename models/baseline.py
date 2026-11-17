"""Baseline 1: predict that the more negative headline gets more clicks.

This turns the finding of Robertson et al. (2023) into a prediction rule.
"""

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def negativity_score(headline: str) -> float:
    """Share of the headline's sentiment that VADER rates as negative (0 to 1)."""
    return analyzer.polarity_scores(headline)["neg"]


def predict_pair(headline_a: str, headline_b: str,
                 rng: np.random.Generator | None = None) -> int:
    """Return 1 if headline_a is predicted to win, 0 if headline_b is.

    If both headlines are equally negative the rule has no opinion, so we
    flip a coin (seeded, for reproducibility). Without an rng, ties return 0.
    """
    score_a, score_b = negativity_score(headline_a), negativity_score(headline_b)
    if score_a == score_b:
        return int(rng.integers(0, 2)) if rng is not None else 0
    return int(score_a > score_b)


def evaluate(pairs: pd.DataFrame, seed: int = 42) -> dict:
    """Accuracy overall (ties broken randomly) and on non-tied pairs only."""
    rng = np.random.default_rng(seed)
    score_a = pairs["headline_a"].map(negativity_score)
    score_b = pairs["headline_b"].map(negativity_score)
    tied = score_a == score_b
    preds = np.array([predict_pair(a, b, rng)
                      for a, b in zip(pairs["headline_a"], pairs["headline_b"])])
    correct = preds == pairs["label"].to_numpy()
    return {
        "n_pairs": len(pairs),
        "n_ties": int(tied.sum()),
        "accuracy_all": correct.mean(),
        "accuracy_non_tied": correct[~tied.to_numpy()].mean(),
    }


if __name__ == "__main__":
    pairs = pd.read_csv("data/processed/pairs_significant.csv")
    val = pairs[pairs["split"] == "val"]
    results = evaluate(val)
    print(f"Pairs:                     {results['n_pairs']}")
    print(f"Ties (equal negativity):   {results['n_ties']} "
          f"({results['n_ties'] / results['n_pairs']:.0%})")
    print(f"Accuracy, all pairs:       {results['accuracy_all']:.3f}")
    print(f"Accuracy, non-tied pairs:  {results['accuracy_non_tied']:.3f}")