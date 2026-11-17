"""Unit tests for the preprocessing pipeline, on a tiny hand-made dataset."""

import pandas as pd
import pytest

from src.preprocessing import (build_pairs, clean_packages, keep_same_image_tests,
                               split_by_test, two_proportion_pvalue)


@pytest.fixture
def packages() -> pd.DataFrame:
    """Three tests: one clean, one with two images, one flagged as a problem."""
    return pd.DataFrame({
        "clickability_test_id": ["t1", "t1", "t1", "t2", "t2", "t3", "t3"],
        "headline": ["A", "B", "B", "C", "D", "E", "F"],
        "impressions": [5000, 5000, 3000, 4000, 4000, 4000, 4000],
        "clicks": [100, 50, 20, 40, 41, 30, 60],
        "eyecatcher_id": ["img1", "img1", "img1", "img2", "img3", "img4", "img4"],
        "problem": [0, 0, 0, 0, 0, 1, 1],
    })


def test_clean_drops_problem_tests_and_duplicates(packages):
    clean = clean_packages(packages, min_impressions=1000)
    assert "t3" not in set(clean["clickability_test_id"])
    # Duplicate headline "B" in t1 is reduced to one row (the larger one).
    t1 = clean[clean["clickability_test_id"] == "t1"]
    assert len(t1) == 2
    assert t1.loc[t1["headline"] == "B", "impressions"].item() == 5000


def test_same_image_filter(packages):
    clean = clean_packages(packages, min_impressions=1000)
    kept = keep_same_image_tests(clean)
    assert set(kept["clickability_test_id"]) == {"t1"}


def test_pvalue_behaviour():
    assert two_proportion_pvalue(100, 5000, 100, 5000) == pytest.approx(1.0)
    assert two_proportion_pvalue(100, 5000, 50, 5000) < 0.001


def test_pairs_label_points_to_higher_ctr(packages):
    clean = keep_same_image_tests(clean_packages(packages, min_impressions=1000))
    pairs = build_pairs(clean)
    assert len(pairs) == 1
    row = pairs.iloc[0]
    winner = row["headline_a"] if row["label"] == 1 else row["headline_b"]
    assert winner == "A"
    assert row["significant"]


def test_split_keeps_tests_together():
    pairs = pd.DataFrame({"test_id": [f"t{i // 3}" for i in range(60)]})
    split = split_by_test(pairs, seed=1)
    assert (split.groupby("test_id")["split"].nunique() == 1).all()
    assert set(split["split"]) == {"train", "val", "test"}
