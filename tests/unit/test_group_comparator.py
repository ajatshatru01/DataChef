"""
Unit tests for GroupComparator.
"""

from __future__ import annotations

import polars as pl

from datachef.analytics.group_comparator import GroupComparator


def test_group_comparator_basic():
    df = pl.DataFrame({
        "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "metric": [10.0, 12.0, 11.0, 13.0, 25.0, 27.0, 26.0, 28.0],
    })

    comparator = GroupComparator()
    result = comparator.compare(df, numeric_col="metric", group_col="group")

    assert "error" not in result
    assert "group_stats" in result
    assert len(result["group_stats"]) == 2

    stats_a = next(g for g in result["group_stats"] if g["group"] == "A")
    stats_b = next(g for g in result["group_stats"] if g["group"] == "B")

    assert stats_a["mean"] < stats_b["mean"]
    assert "effect_sizes" in result
    assert len(result["effect_sizes"]) >= 1

    comparison = result["effect_sizes"][0]
    assert comparison["cohens_d"] is not None
    # Effect size should be large
    assert abs(comparison["cohens_d"]) > 2.0


def test_group_comparator_invalid_column():
    df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    comparator = GroupComparator()
    result = comparator.compare(df, numeric_col="non_existent", group_col="y")
    assert "error" in result
