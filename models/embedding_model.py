"""Model A: frozen sentence embeddings + logistic regression on the headline pair.

A pretrained sentence encoder (bge-small, 33M parameters) turns each headline
into a 384-number vector that captures its meaning, not just its words. As in
the TF-IDF baseline, the classifier sees the difference between the two
vectors (a minus b) and learns which direction in "meaning space" wins clicks.

The encoder itself is not trained (it stays frozen), so this runs on a laptop CPU.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

MODEL_NAME = "BAAI/bge-small-en-v1.5"
CACHE_DIR = Path("data/processed/embeddings")


def embed_headlines(headlines: list[str], encoder: SentenceTransformer) -> dict[str, np.ndarray]:
    """Encode each unique headline once and return a {headline: vector} lookup."""
    unique = sorted(set(headlines))
    vectors = encoder.encode(unique, batch_size=64, show_progress_bar=True,
                             normalize_embeddings=True)
    return dict(zip(unique, vectors))


def load_or_compute_embeddings(pairs: pd.DataFrame) -> dict[str, np.ndarray]:
    """Embeddings take a few minutes on CPU, so we save them and reuse them next time."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "bge_small.npz"
    headlines = list(pairs["headline_a"]) + list(pairs["headline_b"])
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        lookup = dict(zip(data["headlines"], data["vectors"]))
        if set(headlines) <= set(lookup):
            return lookup
    encoder = SentenceTransformer(MODEL_NAME)
    lookup = embed_headlines(headlines, encoder)
    np.savez(cache, headlines=np.array(list(lookup), dtype=object),
             vectors=np.stack(list(lookup.values())))
    return lookup


def pair_features(pairs: pd.DataFrame, lookup: dict[str, np.ndarray]) -> np.ndarray:
    """Feature for a pair = embedding(headline_a) minus embedding(headline_b)."""
    a = np.stack([lookup[h] for h in pairs["headline_a"]])
    b = np.stack([lookup[h] for h in pairs["headline_b"]])
    return a - b


if __name__ == "__main__":
    pairs = pd.read_csv("data/processed/pairs_significant.csv")
    lookup = load_or_compute_embeddings(pairs)
    train = pairs[pairs["split"] == "train"]
    val = pairs[pairs["split"] == "val"]
    X_train, y_train = pair_features(train, lookup), train["label"]
    X_val, y_val = pair_features(val, lookup), val["label"]

    best_C, best_acc = None, 0.0
    for C in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1]:
        model = LogisticRegression(C=C, fit_intercept=False, max_iter=5000, random_state=42)
        acc = (model.fit(X_train, y_train).predict(X_val) == y_val).mean()
        print(f"C = {C:<5} validation accuracy = {acc:.3f}")
        if acc > best_acc:
            best_C, best_acc = C, acc
    print(f"\nBest: C = {best_C}, validation accuracy = {best_acc:.3f}")

    # Save the best model's validation predictions for significance tests.
    model = LogisticRegression(C=best_C, fit_intercept=False, max_iter=5000, random_state=42)
    model.fit(X_train, y_train)
    out = val[["test_id", "story_group", "headline_a", "headline_b", "label"]].copy()
    out["prob_a_wins"] = model.predict_proba(X_val)[:, 1]
    out["prediction"] = (out["prob_a_wins"] > 0.5).astype(int)
    out["correct"] = out["prediction"] == out["label"]
    out.to_csv("data/processed/embedding_val_predictions.csv", index=False)
    print("Saved data/processed/embedding_val_predictions.csv")
