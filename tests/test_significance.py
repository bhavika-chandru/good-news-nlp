"""Unit tests for the significance tests."""

import numpy as np
import pandas as pd

from src.significance import bootstrap_difference, mcnemar


def test_mcnemar_identical_models_not_significant():
    c = np.array([True, False, True, True])
    assert mcnemar(c, c)["p_value"] == 1.0


def test_mcnemar_clear_winner_is_significant():
    a = np.array([True] * 40)
    b = np.array([False] * 30 + [True] * 10)
    result = mcnemar(a, b)
    assert result["only_a_right"] == 30 and result["only_b_right"] == 0
    assert result["p_value"] < 0.001


def test_bootstrap_ci_contains_observed_difference():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "story_group": [f"g{i // 2}" for i in range(400)],
        "correct_a": rng.random(400) < 0.75,
        "correct_b": rng.random(400) < 0.65,
    })
    result = bootstrap_difference(df, n_boot=500)
    assert result["ci_low"] <= result["difference"] <= result["ci_high"]
