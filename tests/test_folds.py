"""Unit tests for cross-validation folds."""

import pandas as pd

from src.make_folds import assign_folds


def make_pairs() -> pd.DataFrame:
    rows = []
    for g in range(50):
        for j in range(1 + g % 4):
            rows.append({"story_group": f"s{g:02d}", "label": j % 2,
                         "split": "test" if g >= 45 else "train"})
    return pd.DataFrame(rows)


def test_stories_never_split_across_folds():
    cv = assign_folds(make_pairs())
    assert (cv.groupby("story_group")["fold"].nunique() == 1).all()


def test_test_split_excluded_and_folds_balanced():
    cv = assign_folds(make_pairs())
    assert "test" not in set(cv["split"])
    sizes = cv.groupby("fold").size()
    assert sorted(sizes.index) == [0, 1, 2, 3, 4]
    assert sizes.max() - sizes.min() <= 4
