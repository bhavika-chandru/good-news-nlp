"""Baseline 2: TF-IDF features + logistic regression on the headline pair.

Each headline becomes a TF-IDF vector. The model sees the DIFFERENCE between
the two vectors (a minus b), so it learns which words push a headline to win.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def fit_vectorizer(train: pd.DataFrame) -> TfidfVectorizer:
    """Learn the vocabulary from training headlines only (no peeking at val/test)."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    vectorizer.fit(pd.concat([train["headline_a"], train["headline_b"]]))
    return vectorizer


def pair_features(pairs: pd.DataFrame, vectorizer: TfidfVectorizer):
    """Feature for a pair = TF-IDF(headline_a) minus TF-IDF(headline_b)."""
    return vectorizer.transform(pairs["headline_a"]) - vectorizer.transform(pairs["headline_b"])


def train_model(X, y, C: float, seed: int = 42) -> LogisticRegression:
    """Logistic regression without an intercept: swapping a and b should flip the prediction."""
    model = LogisticRegression(C=C, fit_intercept=False, max_iter=2000, random_state=seed)
    return model.fit(X, y)


if __name__ == "__main__":
    pairs = pd.read_csv("data/processed/pairs_significant.csv")
    train = pairs[pairs["split"] == "train"]
    val = pairs[pairs["split"] == "val"]

    vectorizer = fit_vectorizer(train)
    X_train, y_train = pair_features(train, vectorizer), train["label"]
    X_val, y_val = pair_features(val, vectorizer), val["label"]
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # Tune the regularisation strength C on the validation set.
    best_C, best_acc = None, 0.0
    for C in [0.01, 0.1, 0.3, 1, 3, 10]:
        acc = (train_model(X_train, y_train, C).predict(X_val) == y_val).mean()
        print(f"C = {C:<5} validation accuracy = {acc:.3f}")
        if acc > best_acc:
            best_C, best_acc = C, acc
    print(f"\nBest: C = {best_C}, validation accuracy = {best_acc:.3f}")

    # Which words push a headline to win or lose?
    model = train_model(X_train, y_train, best_C)
    words = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(model.coef_[0])
    print("\nWords linked to WINNING: ", ", ".join(words[order[-15:]][::-1]))
    print("Words linked to LOSING:  ", ", ".join(words[order[:15]]))
