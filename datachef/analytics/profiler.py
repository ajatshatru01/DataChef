"""
DatasetProfiler — computes the full DatasetProfile for an uploaded dataset.

Uses Polars for column statistics and type inference, SciPy for skewness /
kurtosis, and a simple IQR method for outlier candidate counts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

from datachef.context.models import (
    CategoricalStats,
    ColumnProfile,
    ColumnRole,
    DatasetProfile,
    NumericStats,
)

# Cardinality thresholds for role inference
_MAX_CATEGORICAL_UNIQUE = 50
_MIN_CATEGORICAL_UNIQUE = 1
_TEXT_LENGTH_THRESHOLD = 80  # avg string length above which → "text"
_MAX_TOP_VALUES = 10


class DatasetProfiler:
    """Computes a DatasetProfile from a Polars DataFrame."""

    def profile(self, df: pl.DataFrame, source_path: str) -> DatasetProfile:
        """
        Run the full profiling pipeline and return a DatasetProfile.

        Parameters
        ----------
        df:
            The loaded dataset (read-only; will not be modified).
        source_path:
            The original file path, stored in the profile for reference.
        """
        row_count, col_count = df.shape
        duplicate_row_count = row_count - df.unique().shape[0]

        columns: list[ColumnProfile] = []
        for col_name in df.columns:
            col_profile = self._profile_column(df, col_name)
            columns.append(col_profile)

        # Correlation matrix for numeric columns
        numeric_cols = [
            c.name for c in columns if c.inferred_role == "numeric"
        ]
        corr_matrix: dict[str, dict[str, float | None]] | None = None
        if len(numeric_cols) >= 2:
            corr_matrix = self._compute_correlations(df, numeric_cols)

        return DatasetProfile(
            source_path=str(source_path),
            dataset_fingerprint=_dataset_fingerprint(df, source_path),
            row_count=row_count,
            column_count=col_count,
            columns=columns,
            duplicate_row_count=duplicate_row_count,
            correlation_matrix=corr_matrix,
        )

    # ------------------------------------------------------------------
    # Column profiling
    # ------------------------------------------------------------------

    def _profile_column(self, df: pl.DataFrame, col_name: str) -> ColumnProfile:
        series = df[col_name]
        dtype_str = str(series.dtype)
        null_count = series.null_count()
        null_pct = null_count / max(len(series), 1) * 100
        n_unique = series.n_unique()

        # Get sample values (up to 5, excluding nulls)
        sample_values = (
            series.drop_nulls().head(5).to_list()
        )

        role = self._infer_role(series, dtype_str)
        stats: NumericStats | CategoricalStats | None = None

        if role == "numeric":
            stats = self._compute_numeric_stats(series)
        elif role == "categorical":
            stats = self._compute_categorical_stats(series)

        return ColumnProfile(
            name=col_name,
            dtype=dtype_str,
            inferred_role=role,
            null_count=null_count,
            null_pct=round(null_pct, 2),
            n_unique=n_unique,
            stats=stats,
            sample_values=sample_values,
        )

    def _infer_role(self, series: pl.Series, dtype_str: str) -> ColumnRole:
        """Infer the semantic role of a column from its dtype and content."""
        dtype = series.dtype

        # Datetime / Date
        if dtype in (pl.Date, pl.Datetime, pl.Duration, pl.Time):
            return "datetime"

        # Numeric types
        if dtype in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        ):
            # If very low cardinality and small range → might be categorical code
            n_unique = series.n_unique()
            n_rows = len(series)
            if n_unique <= 10 and n_rows > 20:
                return "categorical"
            return "numeric"

        # String types
        if dtype == pl.Utf8 or dtype == pl.String:
            n_unique = series.n_unique()
            n_rows = len(series)
            # ID heuristic: nearly every value unique
            if n_unique > 0.95 * n_rows and n_rows > 20:
                # Check if it looks like an identifier (short, consistent)
                non_null = series.drop_nulls()
                if len(non_null) > 0:
                    avg_len = non_null.str.len_chars().mean() or 0
                    if avg_len < 40:
                        return "id"
            # Text heuristic: long average string length
            non_null = series.drop_nulls()
            if len(non_null) > 0:
                avg_len = non_null.str.len_chars().mean() or 0
                if avg_len > _TEXT_LENGTH_THRESHOLD:
                    return "text"
            # Reasonable cardinality → categorical
            if _MIN_CATEGORICAL_UNIQUE <= n_unique <= _MAX_CATEGORICAL_UNIQUE:
                return "categorical"
            if n_unique > _MAX_CATEGORICAL_UNIQUE:
                return "text"
            return "categorical"

        # Boolean → categorical
        if dtype == pl.Boolean:
            return "categorical"

        # Categorical dtype
        if dtype == pl.Categorical:
            return "categorical"

        return "unknown"

    def _compute_numeric_stats(self, series: pl.Series) -> NumericStats:
        """Compute descriptive statistics for a numeric column."""
        arr = series.drop_nulls().cast(pl.Float64).to_numpy()
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return NumericStats()

        q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        iqr = q75 - q25

        # IQR outlier count
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        n_outliers = int(np.sum((arr < lower) | (arr > upper)))

        skewness: float | None = None
        kurtosis: float | None = None
        if len(arr) >= 4:
            try:
                skewness = float(scipy_stats.skew(arr))
                kurtosis = float(scipy_stats.kurtosis(arr))
                if not np.isfinite(skewness):
                    skewness = None
                if not np.isfinite(kurtosis):
                    kurtosis = None
            except Exception:
                pass

        return NumericStats(
            mean=round(float(np.mean(arr)), 6),
            median=round(float(np.median(arr)), 6),
            std=round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 6),
            min=round(float(np.min(arr)), 6),
            max=round(float(np.max(arr)), 6),
            q25=round(q25, 6),
            q75=round(q75, 6),
            iqr=round(iqr, 6),
            skewness=round(skewness, 4) if skewness is not None else None,
            kurtosis=round(kurtosis, 4) if kurtosis is not None else None,
            n_outliers_iqr=n_outliers,
        )

    def _compute_categorical_stats(self, series: pl.Series) -> CategoricalStats:
        """Compute value counts for a categorical column."""
        n_rows = len(series)
        vc = (
            series.drop_nulls()
            .value_counts(sort=True)
            .head(_MAX_TOP_VALUES)
        )
        top_values: list[dict] = []
        for row in vc.iter_rows(named=True):
            # polars value_counts returns {"<col_name>": value, "count": n}
            value = next(v for k, v in row.items() if k != "count")
            count = row["count"]
            top_values.append(
                {
                    "value": str(value),
                    "count": count,
                    "pct": round(count / max(n_rows, 1) * 100, 2),
                }
            )
        return CategoricalStats(
            top_values=top_values,
            cardinality=series.n_unique(),
        )

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    def _compute_correlations(
        self, df: pl.DataFrame, numeric_cols: list[str]
    ) -> dict[str, dict[str, float | None]]:
        """
        Compute pairwise Pearson correlation matrix for numeric columns.
        Handles nulls by dropping rows with any null in the selected columns.
        """
        sub = df.select(numeric_cols).drop_nulls()
        if sub.shape[0] < 5:
            return {}

        raw_arrays = [sub[col].cast(pl.Float64).to_numpy() for col in numeric_cols]
        finite_mask = np.logical_and.reduce([np.isfinite(arr) for arr in raw_arrays])
        if int(np.sum(finite_mask)) < 5:
            return {}

        matrix: dict[str, dict[str, float | None]] = {}
        for col_a in numeric_cols:
            matrix[col_a] = {}
            arr_a = sub[col_a].cast(pl.Float64).to_numpy()[finite_mask]
            for col_b in numeric_cols:
                if col_a == col_b:
                    matrix[col_a][col_b] = 1.0
                    continue
                arr_b = sub[col_b].cast(pl.Float64).to_numpy()[finite_mask]
                try:
                    r, _ = scipy_stats.pearsonr(arr_a, arr_b)
                    matrix[col_a][col_b] = round(float(r), 4) if np.isfinite(r) else None
                except Exception:
                    matrix[col_a][col_b] = None

        return matrix


def _dataset_fingerprint(df: pl.DataFrame, source_path: str) -> str:
    """Return a stable fingerprint without persisting any raw rows."""
    path = Path(source_path)
    if path.is_file():
        identity = f"{path.resolve()}:{path.stat().st_mtime_ns}"
    else:
        row_hashes = df.hash_rows(seed=0).to_list()
        identity = json.dumps(
            {
                "source": str(source_path),
                "schema": {name: str(dtype) for name, dtype in df.schema.items()},
                "row_hashes": row_hashes,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    return f"sha256:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"
