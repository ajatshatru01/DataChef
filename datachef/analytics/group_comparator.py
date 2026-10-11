"""
GroupComparator — compares a numeric metric across categorical groups.

Computes group-level statistics, Cohen's d effect size, and an optional
Kruskal–Wallis non-parametric significance test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy import stats as scipy_stats


class GroupComparator:
    """Compare a numeric column across groups defined by a categorical column."""

    def compare(
        self,
        df: pl.DataFrame,
        numeric_col: str,
        group_col: str,
        run_kruskal: bool = True,
        top_groups: int = 10,
    ) -> dict[str, Any]:
        """
        Compare *numeric_col* across groups in *group_col*.

        Returns group-level stats, effect sizes between the largest group
        and each other group (Cohen's d), and optionally a Kruskal–Wallis
        test result.

        Parameters
        ----------
        df:
            Source DataFrame.
        numeric_col:
            The numeric column to compare.
        group_col:
            The categorical grouping column.
        run_kruskal:
            Whether to run Kruskal–Wallis H-test across all groups.
        top_groups:
            Limit output to this many groups (by sample size, descending).
        """
        if group_col not in df.columns:
            return {"error": f"Group column '{group_col}' not found in dataset."}
        if numeric_col not in df.columns:
            return {"error": f"Numeric column '{numeric_col}' not found in dataset."}

        sub = df.select([group_col, numeric_col]).drop_nulls()
        if sub.is_empty():
            return {"error": "No non-null rows for the specified columns."}

        # Build per-group arrays
        group_arrays: dict[str, np.ndarray] = {}
        for group_val in sub[group_col].unique().to_list():
            arr = (
                sub.filter(pl.col(group_col) == group_val)[numeric_col]
                .cast(pl.Float64)
                .to_numpy()
            )
            arr = arr[np.isfinite(arr)]
            if len(arr):
                group_arrays[str(group_val)] = arr

        if not group_arrays:
            return {"error": "No finite numeric values were found for the specified columns."}

        # Sort by sample size descending; limit to top_groups
        sorted_groups = sorted(group_arrays.items(), key=lambda x: -len(x[1]))[:top_groups]

        # Group-level statistics
        group_stats: list[dict[str, Any]] = []
        for name, arr in sorted_groups:
            stat: dict[str, Any] = {
                "group": name,
                "n": len(arr),
                "mean": round(float(np.mean(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
            }
            group_stats.append(stat)

        # Effect size: Cohen's d between every pair (only first 5 groups to keep output bounded)
        effect_sizes: list[dict[str, Any]] = []
        pairs = sorted_groups[:5]
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                name_i, arr_i = pairs[i]
                name_j, arr_j = pairs[j]
                d = _cohens_d(arr_i, arr_j)
                effect_sizes.append({
                    "group_a": name_i,
                    "group_b": name_j,
                    "cohens_d": round(d, 4) if d is not None else None,
                    "magnitude": _d_magnitude(d),
                })

        # Kruskal–Wallis test
        kruskal_result: dict[str, Any] | None = None
        if run_kruskal and len(sorted_groups) >= 2:
            arrays = [arr for _, arr in sorted_groups if len(arr) >= 2]
            if len(arrays) >= 2:
                try:
                    stat_h, p_value = scipy_stats.kruskal(*arrays)
                    kruskal_result = {
                        "test": "Kruskal–Wallis H",
                        "H_statistic": round(float(stat_h), 4),
                        "p_value": round(float(p_value), 6),
                        "significant_at_0_05": p_value < 0.05,
                        "note": (
                            "The Kruskal–Wallis test indicates whether at least one group "
                            "median differs from the others. A significant result does not "
                            "identify which groups differ."
                        ),
                    }
                except Exception as exc:
                    kruskal_result = {"error": str(exc)}

        return {
            "numeric_col": numeric_col,
            "group_col": group_col,
            "n_groups_total": len(group_arrays),
            "n_groups_shown": len(group_stats),
            "group_stats": group_stats,
            "effect_sizes": effect_sizes,
            "kruskal_wallis": kruskal_result,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cohens_d(arr_a: np.ndarray, arr_b: np.ndarray) -> float | None:
    """Compute Cohen's d between two samples."""
    n_a, n_b = len(arr_a), len(arr_b)
    if n_a < 2 or n_b < 2:
        return None
    mean_diff = np.mean(arr_a) - np.mean(arr_b)
    pooled_std = np.sqrt(
        ((n_a - 1) * np.var(arr_a, ddof=1) + (n_b - 1) * np.var(arr_b, ddof=1))
        / (n_a + n_b - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float(mean_diff / pooled_std)


def _d_magnitude(d: float | None) -> str:
    """Classify Cohen's d by conventional thresholds."""
    if d is None or not np.isfinite(d):
        return "unknown"
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"
