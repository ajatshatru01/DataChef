"""Additional schema-boundary tests for model-generated tool arguments."""

from datachef.tools.registry import ToolRegistry, ToolValidator


def _validator() -> ToolValidator:
    return ToolValidator(ToolRegistry(), ["sales", "profit", "region"])


def test_validator_rejects_wrong_types_and_unknown_parameters() -> None:
    errors = _validator().validate(
        "detect_outliers",
        {"column": "sales", "max_results": "many", "surprise": True},
    )

    assert any("must be of type integer" in error for error in errors)
    assert any("Unexpected parameter" in error for error in errors)


def test_validator_enforces_ranges_and_minimum_items() -> None:
    assert _validator().validate("get_correlation", {"columns": ["sales"]})
    assert _validator().validate(
        "get_top_correlations",
        {"columns": ["sales", "profit"], "threshold": 1.5},
    )
    assert _validator().validate(
        "get_column_stats",
        {"columns": ["sales"], "percentiles": [-1, 50, 101]},
    )


def test_duplicate_summary_is_registered_and_bounded() -> None:
    registry = ToolRegistry()
    validator = ToolValidator(registry, ["sales"])

    assert registry.has_tool("get_duplicate_summary")
    assert validator.validate("get_duplicate_summary", {"max_results": 20}) == []
    assert validator.validate("get_duplicate_summary", {"max_results": 0})
