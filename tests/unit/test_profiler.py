"""
Unit tests for DatasetProfiler.
"""

from __future__ import annotations

import polars as pl
import pytest

from datachef.analytics.profiler import DatasetProfiler


def test_profiler_basic_stats():
    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "category": ["A", "B", "A", "B", "C"],
        "metric": [10.0, 20.0, 15.0, 30.0, 25.0],
    })

    profiler = DatasetProfiler()
    profile = profiler.profile(df, source_path="test.csv")

    assert profile.row_count == 5
    assert profile.column_count == 3
    assert profile.duplicate_row_count == 0

    col_names = profile.column_names()
    assert "id" in col_names
    assert "category" in col_names
    assert "metric" in col_names

    metric_col = profile.get_column("metric")
    assert metric_col is not None
    assert metric_col.inferred_role == "numeric"
    assert metric_col.stats is not None
    assert metric_col.stats.mean == 20.0
    assert metric_col.stats.min == 10.0
    assert metric_col.stats.max == 30.0


def test_profiler_null_detection():
    df = pl.DataFrame({
        "feature_a": [1.0, None, 3.0, None, 5.0],
        "feature_b": ["x", "y", "z", "w", "v"],
    })

    profiler = DatasetProfiler()
    profile = profiler.profile(df, source_path="test_nulls.csv")

    col_a = profile.get_column("feature_a")
    assert col_a is not None
    assert col_a.null_count == 2
    assert pytest.approx(col_a.null_pct, 0.01) == 40.0


def test_profiler_correlation_matrix():
    df = pl.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [2.0, 4.0, 6.0, 8.0, 10.0],
        "c": [5.0, 4.0, 3.0, 2.0, 1.0],
    })

    profiler = DatasetProfiler()
    profile = profiler.profile(df, source_path="test_corr.csv")

    assert profile.correlation_matrix is not None
    assert pytest.approx(profile.correlation_matrix["a"]["b"], 0.01) == 1.0
    assert pytest.approx(profile.correlation_matrix["a"]["c"], 0.01) == -1.0
