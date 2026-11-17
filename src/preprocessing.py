"""Turn the raw Upworthy archive into clean headline pairs.

Each Upworthy A/B test showed several "packages" (headline + image) for the
same story to randomly assigned visitors. We keep tests where only the
headline changed, then compare every two headlines inside a test. A pair is
labelled with the headline that got the higher click-through rate (CTR),
and kept only if that difference is statistically significant.

Run as a script for a quick data check:
    python -m src.preprocessing --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import html
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

REQUIRED_COLUMNS = ["clickability_test_id", "headline", "impressions", "clicks"]

# Archive authors: tests created in this window may have broken randomisation.
PROBLEM_WINDOW = (pd.Timestamp("2013-06-25"), pd.Timestamp("2014-01-10 23:59:59"))


def strip_html(text: object) -> str:
    """Remove HTML tags and collapse whitespace in the `lede` field."""
    if not isinstance(text, str):
        return ""
    no_tags = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(re.sub(r"\s+", " ", no_tags)).strip()


def load_config(path: str | Path) -> dict:
    """Read the YAML config file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_packages(path: str | Path) -> pd.DataFrame:
    """Load an Upworthy packages CSV and check the columns we rely on exist."""
    df = pd.read_csv(path, low_memory=False)
    # Some releases include an unnamed index column; drop it if present.
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def clean_packages(df: pd.DataFrame, min_impressions: int = 1000,
                   drop_problem_tests: bool = True) -> pd.DataFrame:
    """Remove unusable packages.

    Drops rows with no headline, packages below `min_impressions`, exact
    duplicate headlines within a test (keeping the one with most impressions),
    and, if asked, tests the archive authors flag as having broken
    randomisation (`problem` == 1).
    """
    out = df.copy()
    out["headline"] = out["headline"].astype(str).str.strip()
    out = out[out["headline"].ne("") & out["headline"].ne("nan")]
    if drop_problem_tests:
        if "problem" in out.columns:
            out = out[out["problem"].fillna(0).astype(int) == 0]
        else:
            # The 03.12.2020 release has no `problem` column, so we drop the
            # flagged window by date, at the test level (a test is dropped if
            # any of its packages was created inside the window).
            created = pd.to_datetime(out["created_at"], errors="coerce", format="mixed")
            in_window = created.between(*PROBLEM_WINDOW)
            bad_tests = set(out.loc[in_window, "clickability_test_id"])
            out = out[~out["clickability_test_id"].isin(bad_tests)]
    if "lede" in out.columns:
        out["lede"] = out["lede"].map(strip_html)
    out = out[out["impressions"] >= min_impressions]
    out = out[out["clicks"] <= out["impressions"]]
    # Stable sort with a tie-break so the result is identical on every machine.
    out = (out.sort_values(["clickability_test_id", "impressions", "headline"],
                           ascending=[True, False, True], kind="mergesort")
              .drop_duplicates(["clickability_test_id", "headline"]))
    out["ctr"] = out["clicks"] / out["impressions"]
    return out.reset_index(drop=True)


def keep_same_image_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only tests where every package used one image (`eyecatcher_id`).

    That way a CTR difference inside a test can be attributed to the headline.
    If the column is missing we return the data unchanged.
    """
    if "eyecatcher_id" not in df.columns:
        return df
    n_images = df.groupby("clickability_test_id")["eyecatcher_id"].transform("nunique")
    return df[n_images == 1].reset_index(drop=True)


def two_proportion_pvalue(clicks_a: int, n_a: int, clicks_b: int, n_b: int) -> float:
    """Two-sided p-value for H0: CTR_a == CTR_b, using a pooled z-test."""
    p_pool = (clicks_a + clicks_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0
    z = (clicks_a / n_a - clicks_b / n_b) / se
    return float(2 * norm.sf(abs(z)))


def build_pairs(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Compare every two headlines within each test.

    Returns one row per pair with both headlines, their CTRs, the p-value,
    `label` (1 if headline_a won, 0 if headline_b won) and `significant`.
    Which headline is called "a" is randomised so the label is balanced.
    """
    rng = np.random.default_rng(0)
    rows = []
    # `excerpt` is mostly a site-wide tagline, so only `lede` is kept as story text.
    text_cols = [c for c in ("lede",) if c in df.columns]
    for test_id, group in df.groupby("clickability_test_id", sort=True):
        # Fixed order inside each test, so the random a/b swap is reproducible.
        records = group.sort_values("headline", kind="mergesort").to_dict("records")
        for a, b in combinations(records, 2):
            if rng.random() < 0.5:
                a, b = b, a
            p = two_proportion_pvalue(a["clicks"], a["impressions"],
                                      b["clicks"], b["impressions"])
            row = {
                "test_id": test_id,
                "headline_a": a["headline"], "headline_b": b["headline"],
                "ctr_a": a["ctr"], "ctr_b": b["ctr"],
                "impressions_a": a["impressions"], "impressions_b": b["impressions"],
                "p_value": p,
                "label": int(a["ctr"] > b["ctr"]),
                "significant": p < alpha,
                "created_at": a.get("created_at"),
            }
            for c in text_cols:
                row[c] = a.get(c)
            rows.append(row)
    return pd.DataFrame(rows)


def story_groups(pairs: pd.DataFrame) -> dict:
    """Map each test to a story group.

    Upworthy sometimes re-tested the same story in a new test. Tests that
    share any headline or the same story text (`lede`) are linked, and each
    connected set of tests counts as one story (union-find).
    """
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for test_id in pairs["test_id"].unique():
        find(("test", test_id))
    for _, row in pairs.iterrows():
        node = ("test", row["test_id"])
        union(node, ("headline", row["headline_a"].lower()))
        union(node, ("headline", row["headline_b"].lower()))
        lede = row.get("lede")
        if isinstance(lede, str) and len(lede) > 30:
            union(node, ("lede", lede.lower()))
    # Name each group by its smallest test ID, which does not depend on the
    # order in which links were made.
    root_of = {t: find(("test", t)) for t in pairs["test_id"].unique()}
    name: dict = {}
    for t, root in root_of.items():
        name[root] = min(name.get(root, t), t)
    return {t: name[root] for t, root in root_of.items()}


def split_by_test(pairs: pd.DataFrame, train: float = 0.7, val: float = 0.15,
                  seed: int = 42) -> pd.DataFrame:
    """Assign each story group (not each pair or test) to train/val/test.

    Splitting by story means no headline or story text appears in more than
    one split, so validation and test scores are not inflated by memorisation.
    """
    groups = story_groups(pairs)
    keys = sorted(set(groups.values()))
    rng = np.random.default_rng(seed)
    keys = [keys[i] for i in rng.permutation(len(keys))]
    n_train = int(len(keys) * train)
    n_val = int(len(keys) * val)
    split = {k: "train" for k in keys[:n_train]}
    split.update({k: "val" for k in keys[n_train:n_train + n_val]})
    split.update({k: "test" for k in keys[n_train + n_val:]})
    out = pairs.copy()
    out["story_group"] = out["test_id"].map(groups)
    out["split"] = out["test_id"].map(lambda t: split[groups[t]])
    return out


def summarise(raw: pd.DataFrame, clean: pd.DataFrame, same_img: pd.DataFrame,
              pairs: pd.DataFrame) -> str:
    """Human-readable report of how much data survives each step."""
    sig = pairs[pairs["significant"]]
    lines = [
        f"Raw packages:              {len(raw):>8,}  in {raw['clickability_test_id'].nunique():,} tests",
        f"After cleaning:            {len(clean):>8,}  in {clean['clickability_test_id'].nunique():,} tests",
        f"Same-image tests only:     {len(same_img):>8,}  in {same_img['clickability_test_id'].nunique():,} tests",
        f"All headline pairs:        {len(pairs):>8,}",
        f"Significant pairs:         {len(sig):>8,}  in {sig['test_id'].nunique():,} tests",
        f"Median CTR:                {clean['ctr'].median():.4f}",
        f"Median packages per test:  {same_img.groupby('clickability_test_id').size().median():.0f}",
    ]
    if "split" in sig.columns:
        counts = sig["split"].value_counts().to_dict()
        lines.append(f"Significant pairs by split: {counts}")
    return "\n".join(lines)


def main() -> None:
    """Run the full preprocessing pipeline and print the data check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", default=None, help="Override the raw CSV path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    f_cfg = cfg["filtering"]
    raw = load_packages(args.input or cfg["data"]["raw_exploratory"])
    clean = clean_packages(raw, f_cfg["min_impressions"], f_cfg["drop_problem_tests"])
    same_img = keep_same_image_tests(clean) if f_cfg["same_image_only"] else clean
    pairs = build_pairs(same_img, f_cfg["alpha"])
    pairs = split_by_test(pairs, cfg["split"]["train"], cfg["split"]["val"], cfg["seed"])

    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_dir / "pairs_all.csv", index=False)
    pairs[pairs["significant"]].to_csv(out_dir / "pairs_significant.csv", index=False)
    print(summarise(raw, clean, same_img, pairs))


if __name__ == "__main__":
    main()
