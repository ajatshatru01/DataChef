"""
TrendAnalyzer — detects temporal trends in time-indexed numeric columns.

Uses Spearman rank correlation as a non-parametric monotonic trend measure.
Also computes period-over-period changes and rolling window statistics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy import stats as scipy_stats


class TrendAnalyzer:
    """Analyses time-dependent trends in numeric columns."""

    def monotonic_trend(
        self,
        df: pl.DataFrame,
        value_col: str,
        time_col: str | None = None,
        window: int = 7,
    ) -> dict[str, Any]:
        """
        Detect monotonic trend in *value_col*, optionally ordered by *time_col*.

        Parameters
        ----------
        df:
            Source DataFrame.
        value_col:
            The numeric column to analyse for trend.
        time_col:
            Optional datetime or sortable column to order the data by.
            If None, the DataFrame row order is used.
        window:
            Rolling window size for smoothed trend line.
        """
        if time_col is not None:
            sub = df.select([time_col, value_col]).drop_nulls().sort(time_col)
            values = sub[value_col].cast(pl.Float64).to_numpy()
            time_vals = sub[time_col].to_list()
        else:
            sub = df.select([value_col]).drop_nulls()
            values = sub[value_col].cast(pl.Float64).to_numpy()
            time_vals = list(range(len(values)))

        finite_mask = np.isfinite(values)
        values = values[finite_mask]
        time_vals = [value for value, keep in zip(time_vals, finite_mask, strict=True) if keep]

        if len(values) < 4:
            return {"error": "Not enough data points for trend analysis (minimum 4 required)."}

        # Spearman rank correlation with index as time proxy
        time_index = np.arange(len(values), dtype=float)
        rho, p_value = scipy_stats.spearmanr(time_index, values)
        if not np.isfinite(rho) or not np.isfinite(p_value):
            return {"error": "Trend is undefined because the finite values have no variation."}

        # Rolling mean
        rolling_means: list[float | None] = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            rolling_means.append(round(float(np.mean(values[start : i + 1])), 4))

        # Period-over-period change (overall first vs last third)
        n = len(values)
        third = max(1, n // 3)
        first_third_mean = float(np.mean(values[:third]))
        last_third_mean = float(np.mean(values[-third:]))
        pct_change = (
            (last_third_mean - first_third_mean) / abs(first_third_mean) * 100
            if first_third_mean != 0
            else None
        )

        # Sample of data for charting (downsample to max 200 points)
        if len(values) > 200:
            idx = np.round(np.linspace(0, len(values) - 1, 200)).astype(int)
            chart_values = [round(float(values[i]), 4) for i in idx]
            chart_times = [str(time_vals[i]) for i in idx]
            chart_rolling = [rolling_means[i] for i in idx]
        else:
            chart_values = [round(float(v), 4) for v in values]
            chart_times = [str(t) for t in time_vals]
            chart_rolling = rolling_means

        return {
            "value_col": value_col,
            "time_col": time_col,
            "n_observations": len(values),
            "trend": {
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(p_value), 6),
                "direction": _trend_direction(rho),
                "strength": _trend_strength(abs(rho)),
                "significant_at_0_05": p_value < 0.05,
            },
            "period_comparison": {
                "first_third_mean": round(first_third_mean, 4),
                "last_third_mean": round(last_third_mean, 4),
                "pct_change": round(pct_change, 2) if pct_change is not None else None,
            },
            "rolling_window": window,
            "chart_data": {
                "times": chart_times,
                "values": chart_values,
                "rolling_mean": chart_rolling,
            },
            "note": (
                "Spearman rho measures the monotonic association between time and value. "
                "A significant rho indicates a consistent directional trend, "
                "not necessarily a linear one."
            ),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trend_direction(rho: float) -> str:
    if abs(rho) < 0.05:
        return "flat"
    return "upward" if rho > 0 else "downward"


def _trend_strength(abs_rho: float) -> str:
    if abs_rho < 0.2:
        return "negligible"
    if abs_rho < 0.4:
        return "weak"
    if abs_rho < 0.6:
        return "moderate"
    if abs_rho < 0.8:
        return "strong"
    return "very strong"
