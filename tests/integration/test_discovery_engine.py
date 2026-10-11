"""
Integration test for DiscoveryEngine.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from datachef.analytics.discovery_engine import DiscoveryEngine
from datachef.analytics.profiler import DatasetProfiler


def test_discovery_engine_candidates():
    # Load sample mixed fixture
    fixture_path = Path(__file__).parent.parent / "fixtures" / "sample_mixed.csv"
    assert fixture_path.exists()

    df = pl.read_csv(fixture_path)
    profiler = DatasetProfiler()
    profile = profiler.profile(df, source_path=str(fixture_path))

    engine = DiscoveryEngine()
    candidates = engine.run(profile)

    assert len(candidates) > 0
    # Should identify missing values (in 'age') or correlations or group diffs
    check_types = {c.check_type for c in candidates}
    assert "missingness" in check_types or "cardinality" in check_types

    # Ensure all candidates have valid scores between 0 and 1
    for c in candidates:
        assert 0.0 <= c.heuristic_score <= 1.0
        assert len(c.description) > 0
