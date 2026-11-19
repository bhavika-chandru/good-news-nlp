"""5-fold cross-validation by story for the laptop models.

Why: the validation split has only a couple of hundred stories, too few to
show that a 3 to 4 point gap between models is real. Here train and val are
combined (the test split is never touched) and divided into 5 folds by
story group. Each model is trained 5 times, each time predicting the fold it
did not see, so every pair gets an "out-of-fold" prediction. That is about
6 times as much evaluation data as validation alone.

Hyperparameters are fixed in advance (the values chosen earlier on the
validation split), so no tuning happens on the folds being scored.

Usage:
    python -m src.make_folds        (once; writes data/processed/pairs_cv.csv)
    python -m src.cross_validate    (runs all laptop models)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from models import baseline, baseline_tfidf

CV_FILE = "data/processed/pairs_cv.csv"
OUT = "data/processed/cv_predictions_{}.csv"
TFIDF_C = 1.0        # chosen on the validation split
EMBEDDING_C = 0.01   # chosen on the validation split


def save(df: pd.DataFrame, probs: np.ndarray, name: str) -> float:
    """Write one model's out-of-fold predictions and return its accuracy."""
    out = df[["test_id", "story_group", "fold", "headline_a", "headline_b", "label"]].copy()
    out["prob_a_wins"] = probs
    out["prediction"] = (probs > 0.5).astype(int)
    out["correct"] = out["prediction"] == out["label"]
    out.to_csv(OUT.format(name), index=False)
    return out["correct"].mean()


def negativity_probs(df: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """The rule needs no training. Ties get a seeded coin flip, as in models/baseline.py."""
    rng = np.random.default_rng(seed)
    preds = [baseline.predict_pair(a, b, rng) for a, b in zip(df["headline_a"], df["headline_b"])]
    return np.array(preds, dtype=float)


def tfidf_probs(df: pd.DataFrame) -> np.ndarray:
    """Out-of-fold TF-IDF predictions; the vocabulary is refit on each training part."""
    probs = np.zeros(len(df))
    for k in sorted(df["fold"].unique()):
        train, held = df[df["fold"] != k], df["fold"] == k
        vec = baseline_tfidf.fit_vectorizer(train)
        model = baseline_tfidf.train_model(baseline_tfidf.pair_features(train, vec), train["label"], TFIDF_C)
        probs[held.to_numpy()] = model.predict_proba(baseline_tfidf.pair_features(df[held], vec))[:, 1]
    return probs


def embedding_probs(df: pd.DataFrame) -> np.ndarray:
    """Out-of-fold predictions for frozen embeddings + logistic regression."""
    from models import embedding_model  # imported here: needs sentence-transformers
    lookup = embedding_model.load_or_compute_embeddings(df)
    X = embedding_model.pair_features(df, lookup)
    probs = np.zeros(len(df))
    for k in sorted(df["fold"].unique()):
        tr, held = (df["fold"] != k).to_numpy(), (df["fold"] == k).to_numpy()
        model = LogisticRegression(C=EMBEDDING_C, fit_intercept=False, max_iter=5000, random_state=42)
        probs[held] = model.fit(X[tr], df["label"][tr]).predict_proba(X[held])[:, 1]
    return probs


if __name__ == "__main__":
    df = pd.read_csv(CV_FILE)
    print(f"Pairs: {len(df)} in {df['story_group'].nunique()} stories, 5 folds\n")
    for name, fn in [("negativity", negativity_probs), ("tfidf", tfidf_probs),
                     ("embedding", embedding_probs)]:
        acc = save(df, fn(df), name)
        per_fold = pd.read_csv(OUT.format(name)).groupby("fold")["correct"].mean()
        print(f"{name:<11} accuracy {acc:.3f}   per fold: {', '.join(f'{v:.3f}' for v in per_fold)}")
    print("\nSaved data/processed/cv_predictions_<model>.csv")
