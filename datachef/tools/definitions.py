"""
Tool definitions — JSON schemas and handler bindings for every registered tool.

Each tool entry maps a name to:
  • A JSON schema (sent to the model as the function definition).
  • A handler callable (receives validated kwargs + the Polars DataFrame).

Adding a new tool requires only adding an entry here and implementing
the handler (or delegating to an existing analytics module).
"""

from __future__ import annotations

from typing import Any

import polars as pl

from datachef.analytics.anomaly_detector import AnomalyDetector
from datachef.analytics.correlation_engine import CorrelationEngine
from datachef.analytics.group_comparator import GroupComparator
from datachef.analytics.stat_engine import StatEngine
from datachef.analytics.trend_analyzer import TrendAnalyzer

# Module-level singletons (stateless)
_stat = StatEngine()
_comparator = GroupComparator()
_anomaly = AnomalyDetector()
_correlation = CorrelationEngine()
_trend = TrendAnalyzer()


# ---------------------------------------------------------------------------
# Handler functions
# All handlers have the signature: (df: pl.DataFrame, **kwargs) -> dict
# ---------------------------------------------------------------------------


def _handle_get_column_stats(df: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:
    columns = kwargs.get("columns", [])
    if isinstance(columns, str):
        columns = [columns]
    percentiles = kwargs.get("percentiles", [25, 50, 75])
    return _stat.describe(df, columns, percentiles=percentiles)


def _handle_compare_groups(df: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:
    return _comparator.compare(
        df,
        numeric_col=kwargs["numeric_col"],
        group_col=kwargs["group_col"],
        run_kruskal=kwargs.get("run_kruskal", True),
        top_groups=kwargs.get("top_groups", 10),
    )


def _handle_get_correlation(df: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:
    return _correlation.pairwise(
        df,
        columns=kwargs["columns"],
        method=kwargs.get("method", "pearson"),
    )


def _handle_detect_outliers(df: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:
    return _anomaly.detect(
        df,
        column=kwargs["column"],
        method=kwargs.get("method", "both"),
        max_results=kwargs.get("max_results", 20),
    )


def _handle_get_value_counts(df: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:
    return _stat.value_counts(
        df,
        column=kwargs["column"],
        top_n=kwargs.get("top_n", 20),
        include_nulls=kwargs.get("include_nulls", True),
    )


def _handle_analyze_trend(df: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:
    return _trend.monotonic_trend(
        df,
        value_col=kwargs["value_col"],
        time_col=kwargs.get("time_col"),
        window=kwargs.get("window", 7),
    )


def _handle_get_missing_summary(df: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:
    return _stat.missing_summary(df)


def _handle_get_duplicate_summary(df: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:
    return _stat.duplicate_summary(df, max_results=kwargs.get("max_results", 20))


def _handle_get_group_distribution(df: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:
    return _stat.group_distribution(
        df,
        numeric_col=kwargs["numeric_col"],
        group_col=kwargs["group_col"],
        top_groups=kwargs.get("top_groups", 10),
    )


def _handle_get_top_correlations(df: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:
    return _correlation.top_n(
        df,
        columns=kwargs["columns"],
        n=kwargs.get("n", 10),
        threshold=kwargs.get("threshold", 0.3),
        method=kwargs.get("method", "pearson"),
    )


def _handle_run_cross_tabulation(df: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:
    return _stat.cross_tab(
        df,
        col_a=kwargs["col_a"],
        col_b=kwargs["col_b"],
        top_a=kwargs.get("top_a", 8),
        top_b=kwargs.get("top_b", 8),
    )


# ---------------------------------------------------------------------------
# Tool registry definition
# Format: name → {"schema": {...}, "handler": callable}
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "get_column_stats": {
        "handler": _handle_get_column_stats,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_column_stats",
                "description": (
                    "Return descriptive statistics (mean, median, std, min, max, percentiles) "
                    "for one or more columns. For categorical columns returns value counts instead."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "uniqueItems": True,
                            "description": "List of column names to describe.",
                        },
                        "percentiles": {
                            "type": "array",
                            "items": {"type": "number", "minimum": 0, "maximum": 100},
                            "minItems": 1,
                            "maxItems": 20,
                            "description": "Percentile values to compute (0–100). Default: [25, 50, 75].",
                            "default": [25, 50, 75],
                        },
                    },
                    "required": ["columns"],
                },
            },
        },
    },

    "compare_groups": {
        "handler": _handle_compare_groups,
        "schema": {
            "type": "function",
            "function": {
                "name": "compare_groups",
                "description": (
                    "Compare a numeric column across groups defined by a categorical column. "
                    "Returns group means, medians, standard deviations, Cohen's d effect sizes, "
                    "and an optional Kruskal–Wallis significance test."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "numeric_col": {
                            "type": "string",
                            "description": "The numeric column to compare.",
                        },
                        "group_col": {
                            "type": "string",
                            "description": "The categorical column defining the groups.",
                        },
                        "run_kruskal": {
                            "type": "boolean",
                            "description": "Whether to run the Kruskal–Wallis test. Default: true.",
                            "default": True,
                        },
                        "top_groups": {
                            "type": "integer",
                            "minimum": 2,
                            "maximum": 100,
                            "description": "Maximum number of groups to return. Default: 10.",
                            "default": 10,
                        },
                    },
                    "required": ["numeric_col", "group_col"],
                },
            },
        },
    },

    "get_correlation": {
        "handler": _handle_get_correlation,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_correlation",
                "description": (
                    "Compute a pairwise correlation matrix for a set of numeric columns. "
                    "Supports Pearson (linear) and Spearman (monotonic) methods."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "uniqueItems": True,
                            "description": "List of numeric column names (minimum 2).",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["pearson", "spearman"],
                            "description": "Correlation method. Default: pearson.",
                            "default": "pearson",
                        },
                    },
                    "required": ["columns"],
                },
            },
        },
    },

    "detect_outliers": {
        "handler": _handle_detect_outliers,
        "schema": {
            "type": "function",
            "function": {
                "name": "detect_outliers",
                "description": (
                    "Identify candidate outlier rows in a numeric column using IQR fences "
                    "and/or the modified Z-score method. Returns outlier row indices and values."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "The numeric column to analyse.",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["iqr", "zscore", "both"],
                            "description": "Detection method. Default: both.",
                            "default": "both",
                        },
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "description": "Maximum number of outlier rows to return. Default: 20.",
                            "default": 20,
                        },
                    },
                    "required": ["column"],
                },
            },
        },
    },

    "get_value_counts": {
        "handler": _handle_get_value_counts,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_value_counts",
                "description": (
                    "Return a frequency table (value counts and percentages) for a categorical column."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "The categorical column to count.",
                        },
                        "top_n": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "description": "Maximum number of values to return. Default: 20.",
                            "default": 20,
                        },
                        "include_nulls": {
                            "type": "boolean",
                            "description": "Whether to include null count in the result. Default: true.",
                            "default": True,
                        },
                    },
                    "required": ["column"],
                },
            },
        },
    },

    "analyze_trend": {
        "handler": _handle_analyze_trend,
        "schema": {
            "type": "function",
            "function": {
                "name": "analyze_trend",
                "description": (
                    "Detect monotonic trend in a numeric column, optionally indexed by a datetime column. "
                    "Returns Spearman rho, p-value, trend direction, and period-over-period comparison."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value_col": {
                            "type": "string",
                            "description": "The numeric column to analyse for trend.",
                        },
                        "time_col": {
                            "type": "string",
                            "description": "Optional datetime/sortable column to order the data by. "
                                           "If omitted, row order is used.",
                        },
                        "window": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10000,
                            "description": "Rolling mean window size. Default: 7.",
                            "default": 7,
                        },
                    },
                    "required": ["value_col"],
                },
            },
        },
    },

    "get_missing_summary": {
        "handler": _handle_get_missing_summary,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_missing_summary",
                "description": (
                    "Return missing value counts and percentages for every column in the dataset. "
                    "Also reports the number of rows with any missing value."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
    },

    "get_duplicate_summary": {
        "handler": _handle_get_duplicate_summary,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_duplicate_summary",
                "description": (
                    "Return duplicate-row counts and bounded duplicate-group hashes. "
                    "No raw row values are returned."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "description": "Maximum duplicate groups to return. Default: 20.",
                            "default": 20,
                        }
                    },
                    "required": [],
                },
            },
        },
    },

    "get_group_distribution": {
        "handler": _handle_get_group_distribution,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_group_distribution",
                "description": (
                    "Return distribution statistics (mean, median, std, min, max) for a numeric "
                    "column broken down by a categorical group column."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "numeric_col": {
                            "type": "string",
                            "description": "The numeric column to summarize.",
                        },
                        "group_col": {
                            "type": "string",
                            "description": "The categorical column to group by.",
                        },
                        "top_groups": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "description": "Maximum number of groups to return. Default: 10.",
                            "default": 10,
                        },
                    },
                    "required": ["numeric_col", "group_col"],
                },
            },
        },
    },

    "get_top_correlations": {
        "handler": _handle_get_top_correlations,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_top_correlations",
                "description": (
                    "Return the N strongest pairwise correlations (above a threshold) "
                    "among a set of numeric columns, sorted by absolute correlation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "uniqueItems": True,
                            "description": "List of numeric column names to consider.",
                        },
                        "n": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "description": "Maximum number of pairs to return. Default: 10.",
                            "default": 10,
                        },
                        "threshold": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Minimum absolute correlation to include (0–1). Default: 0.3.",
                            "default": 0.3,
                        },
                        "method": {
                            "type": "string",
                            "enum": ["pearson", "spearman"],
                            "description": "Correlation method. Default: pearson.",
                            "default": "pearson",
                        },
                    },
                    "required": ["columns"],
                },
            },
        },
    },

    "run_cross_tabulation": {
        "handler": _handle_run_cross_tabulation,
        "schema": {
            "type": "function",
            "function": {
                "name": "run_cross_tabulation",
                "description": (
                    "Compute a frequency cross-tabulation of two categorical columns. "
                    "Useful for understanding the joint distribution of two categorical variables."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "col_a": {
                            "type": "string",
                            "description": "First categorical column (rows of the table).",
                        },
                        "col_b": {
                            "type": "string",
                            "description": "Second categorical column (columns of the table).",
                        },
                        "top_a": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "description": "Limit row axis to top N most frequent values. Default: 8.",
                            "default": 8,
                        },
                        "top_b": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "description": "Limit column axis to top N most frequent values. Default: 8.",
                            "default": 8,
                        },
                    },
                    "required": ["col_a", "col_b"],
                },
            },
        },
    },
}
