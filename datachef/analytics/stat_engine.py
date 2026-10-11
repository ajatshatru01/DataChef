"""
StatEngine — descriptive statistics, aggregations, value counts, cross-tabulations.

All methods accept a Polars DataFrame and return serializable dicts / lists
suitable for inclusion in a ToolResult.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


class StatEngine:
    """Local statistical computation engine."""

    # ------------------------------------------------------------------
    # Descriptive statistics
    # ------------------------------------------------------------------

    def describe(
        self,
        df: pl.DataFrame,
        columns: list[str],
        percentiles: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Return descriptive statistics for one or more columns.

        Parameters
        ----------
        df:
            Source DataFrame.
        columns:
            Column names to describe.
        percentiles:
            Optional list of percentiles (0–100) to include.
        """
        percentiles = percentiles or [25, 50, 75]
        result: dict[str, Any] = {}

        for col_name in columns:
            series = df[col_name].drop_nulls()
            dtype = df[col_name].dtype
            n_total = len(df[col_name])
            n_null = df[col_name].null_count()

            col_result: dict[str, Any] = {
                "dtype": str(dtype),
                "n_total": n_total,
                "n_null": n_null,
                "n_valid": n_total - n_null,
            }

            if dtype in (
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64,
            ):
                arr = series.cast(pl.Float64).to_numpy()
                arr = arr[np.isfinite(arr)]
                if len(arr) > 0:
                    col_result.update({
                        "mean": round(float(np.mean(arr)), 6),
                        "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 6),
                        "min": round(float(np.min(arr)), 6),
                        "max": round(float(np.max(arr)), 6),
                    })
                    for p in percentiles:
                        col_result[f"p{int(p)}"] = round(float(np.percentile(arr, p)), 6)
            else:
                col_result["n_unique"] = series.n_unique()
                top = series.value_counts(sort=True).head(5)
                col_result["top_values"] = [
                    {"value": str(r[col_name]), "count": r["count"]}
                    for r in top.iter_rows(named=True)
                ]

            result[col_name] = col_result

        return result

    # ------------------------------------------------------------------
    # Value counts
    # ------------------------------------------------------------------

    def value_counts(
        self,
        df: pl.DataFrame,
        column: str,
        top_n: int = 20,
        include_nulls: bool = True,
    ) -> dict[str, Any]:
        """
        Return frequency table for a categorical column.

        Parameters
        ----------
        df:
            Source DataFrame.
        column:
            Column name.
        top_n:
            Maximum number of values to return.
        include_nulls:
            Whether to count null values as a separate category.
        """
        n_total = len(df)
        null_count = df[column].null_count()

        vc = (
            df[column]
            .drop_nulls()
            .value_counts(sort=True)
            .head(top_n)
        )

        rows = []
        for row in vc.iter_rows(named=True):
            value = next(v for k, v in row.items() if k != "count")
            count = row["count"]
            rows.append({
                "value": str(value),
                "count": count,
                "pct": round(count / max(n_total, 1) * 100, 2),
            })

        result: dict[str, Any] = {
            "column": column,
            "n_total": n_total,
            "n_unique": df[column].n_unique(),
            "n_null": null_count,
            "values": rows,
        }
        if include_nulls and null_count > 0:
            result["null_pct"] = round(null_count / max(n_total, 1) * 100, 2)
        return result

    # ------------------------------------------------------------------
    # Missing value summary
    # ------------------------------------------------------------------

    def missing_summary(self, df: pl.DataFrame) -> dict[str, Any]:
        """
        Return missing value counts and percentages for all columns.
        Columns with no missing values are included for completeness.
        """
        n_rows = len(df)
        columns_info = []
        for col in df.columns:
            null_count = df[col].null_count()
            columns_info.append({
                "column": col,
                "null_count": null_count,
                "null_pct": round(null_count / max(n_rows, 1) * 100, 2),
                "has_missing": null_count > 0,
            })

        columns_info.sort(key=lambda x: -x["null_count"])
        n_complete_cols = sum(1 for c in columns_info if not c["has_missing"])
        n_rows_with_any_null = int(df.filter(
            pl.any_horizontal(pl.col(c).is_null() for c in df.columns)
        ).shape[0])

        return {
            "n_rows": n_rows,
            "n_columns": len(df.columns),
            "n_complete_columns": n_complete_cols,
            "n_columns_with_missing": len(df.columns) - n_complete_cols,
            "n_rows_with_any_null": n_rows_with_any_null,
            "columns": columns_info,
        }

    def duplicate_summary(
        self, df: pl.DataFrame, max_results: int = 20
    ) -> dict[str, Any]:
        """Return duplicate-row counts and bounded duplicate examples as hashes/counts."""
        n_rows = len(df)
        grouped = (
            df.group_by(df.columns)
            .len(name="duplicate_count")
            .filter(pl.col("duplicate_count") > 1)
            .sort("duplicate_count", descending=True)
        )
        duplicate_row_count = int(grouped["duplicate_count"].sum() - grouped.height)
        groups = []
        if grouped.height:
            # Do not return raw rows. A stable row hash is sufficient to identify
            # and reproduce each duplicate group locally.
            row_hashes = grouped.drop("duplicate_count").hash_rows().to_list()
            counts = grouped["duplicate_count"].to_list()
            groups = [
                {"row_hash": str(row_hash), "count": int(count)}
                for row_hash, count in zip(row_hashes[:max_results], counts[:max_results], strict=True)
            ]
        return {
            "n_rows": n_rows,
            "duplicate_row_count": duplicate_row_count,
            "duplicate_pct": round(duplicate_row_count / max(n_rows, 1) * 100, 2),
            "duplicate_group_count": grouped.height,
            "duplicate_groups": groups,
            "groups_truncated": grouped.height > max_results,
        }

    # ------------------------------------------------------------------
    # Grouped aggregation
    # ------------------------------------------------------------------

    def group_distribution(
        self,
        df: pl.DataFrame,
        numeric_col: str,
        group_col: str,
        top_groups: int = 10,
    ) -> dict[str, Any]:
        """
        Distribution statistics for a numeric column broken down by a group.

        Parameters
        ----------
        df:
            Source DataFrame.
        numeric_col:
            The numeric column to analyse.
        group_col:
            The categorical column to group by.
        top_groups:
            Maximum number of groups to return (by sample size, descending).
        """
        sub = df.select([group_col, numeric_col]).drop_nulls()
        if sub.is_empty():
            return {"error": "No non-null rows found for the specified columns."}

        agg = (
            sub.group_by(group_col)
            .agg([
                pl.col(numeric_col).count().alias("n"),
                pl.col(numeric_col).mean().alias("mean"),
                pl.col(numeric_col).median().alias("median"),
                pl.col(numeric_col).std().alias("std"),
                pl.col(numeric_col).min().alias("min"),
                pl.col(numeric_col).max().alias("max"),
            ])
            .sort("n", descending=True)
            .head(top_groups)
        )

        groups = []
        for row in agg.iter_rows(named=True):
            groups.append({
                "group": str(row[group_col]),
                "n": row["n"],
                "mean": round(row["mean"], 4) if row["mean"] is not None else None,
                "median": round(row["median"], 4) if row["median"] is not None else None,
                "std": round(row["std"], 4) if row["std"] is not None else None,
                "min": round(row["min"], 4) if row["min"] is not None else None,
                "max": round(row["max"], 4) if row["max"] is not None else None,
            })

        return {
            "numeric_col": numeric_col,
            "group_col": group_col,
            "n_groups_total": sub[group_col].n_unique(),
            "n_groups_shown": len(groups),
            "groups": groups,
        }

    # ------------------------------------------------------------------
    # Cross-tabulation
    # ------------------------------------------------------------------

    def cross_tab(
        self,
        df: pl.DataFrame,
        col_a: str,
        col_b: str,
        top_a: int = 8,
        top_b: int = 8,
    ) -> dict[str, Any]:
        """
        Frequency cross-tabulation of two categorical columns.

        Parameters
        ----------
        df, col_a, col_b:
            Source data and column names.
        top_a, top_b:
            Limit each axis to its most frequent values.
        """
        sub = df.select([col_a, col_b]).drop_nulls()
        if sub.is_empty():
            return {"error": "No non-null rows found for the specified columns."}

        # Identify top values for each column
        top_vals_a = (
            sub[col_a].value_counts(sort=True).head(top_a)[col_a].to_list()
        )
        top_vals_b = (
            sub[col_b].value_counts(sort=True).head(top_b)[col_b].to_list()
        )

        sub_filtered = sub.filter(
            pl.col(col_a).is_in(top_vals_a) & pl.col(col_b).is_in(top_vals_b)
        )

        crosstab = (
            sub_filtered
            .group_by([col_a, col_b])
            .agg(pl.len().alias("count"))
            .sort([col_a, col_b])
        )

        rows = [
            {"row_val": str(r[col_a]), "col_val": str(r[col_b]), "count": r["count"]}
            for r in crosstab.iter_rows(named=True)
        ]

        return {
            "col_a": col_a,
            "col_b": col_b,
            "n_rows_analysed": len(sub_filtered),
            "row_values": [str(v) for v in top_vals_a],
            "col_values": [str(v) for v in top_vals_b],
            "cells": rows,
        }
