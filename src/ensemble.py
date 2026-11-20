"""Average the predicted probabilities of two or more models.

If models make different mistakes, averaging their confidence can beat each
of them alone. Works on any prediction files that cover the same pairs.

Usage:
    python -m src.ensemble data/processed/cv_predictions_deberta.csv \
        data/processed/cv_predictions_llm_lora.csv \
        --out data/processed/cv_predictions_ensemble.csv
"""

import argparse

import pandas as pd

KEY = ["test_id", "headline_a", "headline_b"]


def average_predictions(files: list[str]) -> pd.DataFrame:
    """Return the first file's pairs with prob_a_wins replaced by the mean over all files."""
    base = pd.read_csv(files[0])
    probs = base[KEY + ["prob_a_wins"]].rename(columns={"prob_a_wins": "p0"})
    for i, f in enumerate(files[1:], start=1):
        other = pd.read_csv(f)[KEY + ["prob_a_wins"]].rename(columns={"prob_a_wins": f"p{i}"})
        probs = probs.merge(other, on=KEY, validate="one_to_one")
    if len(probs) != len(base):
        raise ValueError("Prediction files do not cover the same pairs")
    out = base.drop(columns=["prob_a_wins", "prediction", "correct"]).merge(
        probs[KEY].assign(prob_a_wins=probs[[f"p{i}" for i in range(len(files))]].mean(axis=1)),
        on=KEY)
    out["prediction"] = (out["prob_a_wins"] > 0.5).astype(int)
    out["correct"] = out["prediction"] == out["label"]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="Prediction CSVs to average (2 or more)")
    parser.add_argument("--out", required=True, help="Where to save the ensemble predictions")
    args = parser.parse_args()
    for f in args.files:
        print(f"{pd.read_csv(f)['correct'].mean():.3f}  {f}")
    ens = average_predictions(args.files)
    ens.to_csv(args.out, index=False)
    print(f"{ens['correct'].mean():.3f}  ensemble (average of probabilities)  saved to {args.out}")
