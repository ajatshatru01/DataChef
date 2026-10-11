"""
DatasetStore — read-only wrapper around the loaded Polars DataFrame.

No layer above the Analytical Execution Layer is permitted to mutate the stored
DataFrame. All access goes through this class.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    pass


class DatasetStore:
    """
    Manages the in-memory Polars DataFrame for an active session.

    The underlying DataFrame is treated as immutable after loading.
    All analytical functions receive a reference to this store and call
    `get_frame()` to obtain a read-only view.
    """

    def __init__(self) -> None:
        self._df: pl.DataFrame | None = None
        self._source_path: str = ""

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> pl.DataFrame:
        """
        Load a CSV or Parquet file into the store.

        Returns the loaded DataFrame. Raises ValueError for unsupported
        file types and IOError for unreadable files.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            df = pl.read_csv(str(path), infer_schema_length=10_000, try_parse_dates=True)
        elif suffix in {".parquet", ".pq"}:
            df = pl.read_parquet(str(path))
        else:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                "DataChef accepts CSV (.csv) and Parquet (.parquet / .pq) files."
            )

        self._df = df
        self._source_path = str(path)
        return df

    def load_dataframe(self, df: pl.DataFrame, source_path: str = "<in-memory>") -> None:
        """Load an already-constructed DataFrame (e.g. from test fixtures)."""
        self._df = df
        self._source_path = source_path

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------

    def get_frame(self) -> pl.DataFrame:
        """Return the stored DataFrame. Raises RuntimeError if no dataset is loaded."""
        if self._df is None:
            raise RuntimeError(
                "No dataset is loaded. Upload a CSV or Parquet file first."
            )
        return self._df

    def is_loaded(self) -> bool:
        return self._df is not None

    @property
    def source_path(self) -> str:
        return self._source_path

    @property
    def shape(self) -> tuple[int, int]:
        df = self.get_frame()
        return df.shape

    @property
    def column_names(self) -> list[str]:
        df = self.get_frame()
        return df.columns

    def column_exists(self, name: str) -> bool:
        return name in self.column_names

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate_columns(self, columns: list[str]) -> list[str]:
        """
        Return a list of column names that do NOT exist in the dataset.
        Empty list means all columns are valid.
        """
        return [c for c in columns if c not in self.column_names]

    def clear(self) -> None:
        """Release the stored DataFrame from memory."""
        self._df = None
        self._source_path = ""
