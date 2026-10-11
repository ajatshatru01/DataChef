"""
CorrelationEngine — pairwise Pearson and Spearman correlations.

Provides both full pairwise computation and a ranked top-N view of the
strongest associations above a configurable threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy import stats as scipy_stats


class CorrelationEngine:
    """Computes pairwise correlations between numeric columns."""

    def pairwise(
        self,
        df: pl.DataFrame,
        columns: list[str],
        method: str = "pearson",
    ) -> dict[str, Any]:
        """
        Compute a pairwise correlation matrix for the specified columns.

        Parameters
        ----------
        df:
            Source DataFrame.
        columns:
            Numeric column names to include. Must be 2 or more.
        method:
            "pearson" or "spearman".
        """
        if len(columns) < 2:
            return {"error": "At least 2 columns are required for correlation."}

        sub = df.select(columns).drop_nulls()
        if sub.shape[0] < 5:
            return {
                "error": "Not enough complete rows to compute correlations "
                "(minimum 5 non-null rows required across all selected columns)."
            }

        arrs = {col: sub[col].cast(pl.Float64).to_numpy() for col in columns}
        finite_mask = np.logical_and.reduce([np.isfinite(arr) for arr in arrs.values()])
        if int(np.sum(finite_mask)) < 5:
            return {
                "error": "Not enough finite complete rows to compute correlations "
                "(minimum 5 required)."
            }
        arrs = {col: arr[finite_mask] for col, arr in arrs.items()}
        matrix: dict[str, dict[str, float]] = {}

        for col_a in columns:
            matrix[col_a] = {}
            for col_b in columns:
                if col_a == col_b:
                    matrix[col_a][col_b] = 1.0
                    continue
                try:
                    if method == "spearman":
                        r, _ = scipy_stats.spearmanr(arrs[col_a], arrs[col_b])
                    else:
                        r, _ = scipy_stats.pearsonr(arrs[col_a], arrs[col_b])
                    matrix[col_a][col_b] = round(float(r), 4) if np.isfinite(r) else None
                except Exception:
                    matrix[col_a][col_b] = None

        return {
            "method": method,
            "n_rows_analysed": sub.shape[0],
            "columns": columns,
            "matrix": matrix,
        }

    def top_n(
        self,
        df: pl.DataFrame,
        columns: list[str],
        n: int = 10,
        threshold: float = 0.3,
        method: str = "pearson",
    ) -> dict[str, Any]:
        """
        Return the N strongest correlations above *threshold*.

        Parameters
        ----------
        df:
            Source DataFrame.
        columns:
            Numeric column names to consider.
        n:
            Maximum number of pairs to return.
        threshold:
            Minimum absolute correlation to include.
        method:
            "pearson" or "spearman".
        """
        pairwise_result = self.pairwise(df, columns, method=method)
        if "error" in pairwise_result:
            return pairwise_result

        matrix = pairwise_result["matrix"]
        pairs: list[dict[str, Any]] = []
        seen: set[frozenset] = set()

        for col_a in columns:
            for col_b in columns:
                if col_a == col_b:
                    continue
                pair_key = frozenset([col_a, col_b])
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                r = matrix[col_a].get(col_b)
                if r is not None and abs(r) >= threshold:
                    pairs.append({
                        "col_a": col_a,
                        "col_b": col_b,
                        "correlation": r,
                        "abs_correlation": abs(r),
                        "direction": "positive" if r > 0 else "negative",
                        "strength": _correlation_strength(abs(r)),
                    })

        pairs.sort(key=lambda x: -x["abs_correlation"])
        pairs = pairs[:n]

        return {
            "method": method,
            "threshold": threshold,
            "n_pairs_found": len(pairs),
            "pairs": pairs,
            "note": (
                "Correlation measures linear (Pearson) or monotonic (Spearman) association. "
                "Correlation does not imply causation."
            ),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _correlation_strength(abs_r: float) -> str:
    if abs_r < 0.1:
        return "negligible"
    if abs_r < 0.3:
        return "weak"
    if abs_r < 0.5:
        return "moderate"
    if abs_r < 0.7:
        return "strong"
    return "very strong"
