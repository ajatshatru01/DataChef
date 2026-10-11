"""
Unit tests for ToolValidator.
"""

from __future__ import annotations

from datachef.tools.registry import ToolRegistry, ToolValidator


def test_validator_valid_call():
    registry = ToolRegistry()
    columns = ["age", "income", "gender"]
    validator = ToolValidator(registry, columns)

    err = validator.validate_and_format_error(
        "get_column_stats",
        {"columns": ["age", "income"]},
    )
    assert err is None


def test_validator_unknown_tool():
    registry = ToolRegistry()
    columns = ["age", "income"]
    validator = ToolValidator(registry, columns)

    err = validator.validate_and_format_error(
        "non_existent_tool",
        {"columns": ["age"]},
    )
    assert err is not None
    assert "does not exist" in err


def test_validator_missing_column():
    registry = ToolRegistry()
    columns = ["age", "income"]
    validator = ToolValidator(registry, columns)

    err = validator.validate_and_format_error(
        "compare_groups",
        {"numeric_col": "salary", "group_col": "age"},
    )
    assert err is not None
    assert "salary" in err
