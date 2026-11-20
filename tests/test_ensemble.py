"""Unit test for probability averaging."""

import pandas as pd

from src.ensemble import average_predictions


def test_average_of_two_models(tmp_path):
    base = {"test_id": ["t1", "t2"], "story_group": ["g1", "g2"], "headline_a": ["a", "c"],
            "headline_b": ["b", "d"], "label": [1, 0]}
    f1, f2 = tmp_path / "m1.csv", tmp_path / "m2.csv"
    pd.DataFrame({**base, "prob_a_wins": [0.9, 0.6], "prediction": [1, 1], "correct": [True, False]}).to_csv(f1, index=False)
    # Second file in a different row order: rows must be matched by pair, not position.
    pd.DataFrame({**{k: v[::-1] for k, v in base.items()}, "prob_a_wins": [0.2, 0.3],
                  "prediction": [0, 0], "correct": [True, False]}).to_csv(f2, index=False)
    ens = average_predictions([str(f1), str(f2)]).set_index("test_id")
    assert ens.loc["t1", "prob_a_wins"] == 0.6    # (0.9 + 0.3) / 2
    assert ens.loc["t2", "prob_a_wins"] == 0.4    # (0.6 + 0.2) / 2
    assert ens["correct"].all()
