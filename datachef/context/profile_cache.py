"""
ProfileCache — caches DatasetProfile on disk by (file path, mtime) key.

Avoids re-running full profiling when the same file is re-uploaded
without modification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from datachef import config
from datachef.context.models import DatasetProfile


class ProfileCache:
    """Disk-backed cache keyed by SHA-256 of (source_path, mtime_ns)."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._dir = cache_dir or config.CACHE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, source_path: str) -> DatasetProfile | None:
        """Return the cached profile for *source_path*, or None on cache miss."""
        key = self._cache_key(source_path)
        if key is None:
            return None
        cache_file = self._dir / f"{key}.json"
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return DatasetProfile.model_validate(data)
        except Exception:
            # Corrupt cache entry — treat as miss
            return None

    def put(self, profile: DatasetProfile) -> None:
        """Write *profile* to the cache."""
        key = self._cache_key(profile.source_path)
        if key is None:
            return
        cache_file = self._dir / f"{key}.json"
        cache_file.write_text(
            profile.model_dump_json(indent=2), encoding="utf-8"
        )

    def invalidate(self, source_path: str) -> None:
        """Remove the cache entry for *source_path* if it exists."""
        key = self._cache_key(source_path)
        if key is None:
            return
        cache_file = self._dir / f"{key}.json"
        if cache_file.exists():
            cache_file.unlink()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(source_path: str) -> str | None:
        """
        Build a cache key from the file path and its modification time.
        Returns None if the file does not exist (so we cannot key by mtime).
        """
        p = Path(source_path)
        if not p.exists():
            return None
        mtime_ns = p.stat().st_mtime_ns
        raw = f"{source_path}:{mtime_ns}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
