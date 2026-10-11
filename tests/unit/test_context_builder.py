"""
Unit tests for ContextBuilder.
"""

from __future__ import annotations

import polars as pl

from datachef.analytics.profiler import DatasetProfiler
from datachef.context.context_builder import ContextBuilder
from datachef.context.models import InvestigationState, ToolCallRecord


def test_context_builder_schema_block():
    df = pl.DataFrame({
        "customer_id": [1, 2, 3],
        "spend": [100.5, 200.0, 300.2],
        "category": ["A", "B", "A"],
    })
    profiler = DatasetProfiler()
    profile = profiler.profile(df, source_path="test_context.csv")

    builder = ContextBuilder(profile)
    schema_block = builder.build_schema_block()

    assert "customer_id" in schema_block
    assert "spend" in schema_block
    assert "category" in schema_block
    assert "3 rows" in schema_block


def test_context_builder_turn_context_no_raw_rows():
    df = pl.DataFrame({
        "val_1": [1, 2, 3],
        "val_2": ["x", "y", "z"],
    })
    profiler = DatasetProfiler()
    profile = profiler.profile(df, source_path="test_no_rows.csv")

    state = InvestigationState(dataset_profile=profile)
    state.add_tool_record(
        ToolCallRecord(
            turn_index=1,
            tool_name="get_column_stats",
            arguments={"columns": ["val_1"]},
            result_summary='{"mean": 2.0, "min": 1.0, "max": 3.0}',
            validation_passed=True,
        )
    )

    builder = ContextBuilder(profile)
    summary = builder.build_profile_summary()
    assert "val_1" in summary
    assert "val_2" in summary
