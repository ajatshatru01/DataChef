"""
AnomalyDetector — identifies candidate outliers in numeric columns.

Uses two complementary methods:
  • IQR fence (Tukey's method): flags values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
  • Modified Z-score (Iglewicz & Hoaglin): robust to non-Gaussian distributions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


class AnomalyDetector:
    """Detects candidate outliers in numeric columns."""

    def detect(
        self,
        df: pl.DataFrame,
        column: str,
        method: str = "both",
        max_results: int = 20,
    ) -> dict[str, Any]:
        """
        Identify candidate outlier rows in *column*.

        Parameters
        ----------
        df:
            Source DataFrame.
        column:
            Name of the numeric column to analyse.
        method:
            One of "iqr", "zscore", or "both".
        max_results:
            Maximum number of outlier rows to return.
        """
        series = df[column]
        arr = series.cast(pl.Float64).to_numpy()
        valid_mask = np.isfinite(arr)
        valid_arr = arr[valid_mask]
        original_indices = np.where(valid_mask)[0]

        if len(valid_arr) < 4:
            return {"error": "Not enough non-null values to detect outliers (minimum 4 required)."}

        results: dict[str, Any] = {
            "column": column,
            "n_analysed": int(len(valid_arr)),
            "methods_used": [],
        }

        iqr_outlier_set: set[int] = set()
        zscore_outlier_set: set[int] = set()

        if method in ("iqr", "both"):
            iqr_flags, iqr_bounds = self._iqr_method(valid_arr)
            iqr_idx = original_indices[iqr_flags].tolist()
            iqr_outlier_set = set(iqr_idx)
            results["iqr_method"] = {
                "lower_fence": round(float(iqr_bounds[0]), 4),
                "upper_fence": round(float(iqr_bounds[1]), 4),
                "n_outliers": len(iqr_idx),
            }
            results["methods_used"].append("iqr")

        if method in ("zscore", "both"):
            zscore_flags = self._modified_zscore_method(valid_arr)
            zscore_idx = original_indices[zscore_flags].tolist()
            zscore_outlier_set = set(zscore_idx)
            results["modified_zscore_method"] = {
                "threshold": 3.5,
                "n_outliers": len(zscore_idx),
            }
            results["methods_used"].append("modified_zscore")

        # Union of outliers across methods
        if method == "iqr":
            all_outlier_idx = sorted(iqr_outlier_set)
        elif method == "zscore":
            all_outlier_idx = sorted(zscore_outlier_set)
        else:
            all_outlier_idx = sorted(iqr_outlier_set | zscore_outlier_set)

        results["n_outliers_total"] = len(all_outlier_idx)
        results["outlier_pct"] = round(len(all_outlier_idx) / max(len(valid_arr), 1) * 100, 2)

        # Sample of outlier rows
        sample_idx = all_outlier_idx[:max_results]
        outlier_rows: list[dict[str, Any]] = []
        for idx in sample_idx:
            row_val = arr[idx]
            detected_by = []
            if idx in iqr_outlier_set:
                detected_by.append("iqr")
            if idx in zscore_outlier_set:
                detected_by.append("modified_zscore")
            outlier_rows.append({
                "row_index": int(idx),
                "value": round(float(row_val), 4) if not np.isnan(row_val) else None,
                "detected_by": detected_by,
            })

        results["outlier_samples"] = outlier_rows
        results["note"] = (
            "These are candidate outliers. Whether they represent errors, "
            "genuine extremes, or interesting observations requires domain knowledge."
        )
        return results

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    @staticmethod
    def _iqr_method(arr: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """Return boolean mask of IQR outliers and the fence bounds."""
        q25 = float(np.percentile(arr, 25))
        q75 = float(np.percentile(arr, 75))
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        return (arr < lower) | (arr > upper), (lower, upper)

    @staticmethod
    def _modified_zscore_method(arr: np.ndarray, threshold: float = 3.5) -> np.ndarray:
        """
        Return boolean mask using the modified Z-score (Iglewicz & Hoaglin, 1993).
        More robust than standard Z-score when the distribution is skewed.
        """
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad == 0:
            # Fall back to mean absolute deviation if MAD is zero
            mad = np.mean(np.abs(arr - median))
        if mad == 0:
            return np.zeros(len(arr), dtype=bool)
        modified_z = 0.6745 * np.abs(arr - median) / mad
        return modified_z > threshold
