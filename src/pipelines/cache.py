"""Cache expensive preprocessing results to disk."""

import hashlib
import pickle
import json
from pathlib import Path
from typing import Callable, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PipelineCache:
    """Cache expensive preprocessing results to disk.

    Cache key components:
    - experiment_id, cell_id, rpt_id (data identity)
    - pipeline_name + params (transform identity)
    - version (manual invalidation)

    Example usage:
        >>> cache = PipelineCache()
        >>> result = cache.get_or_compute(
        ...     experiment_id=5, cell_id='A', rpt_id=3,
        ...     pipeline_name='ica_peaks',
        ...     pipeline_params={'sg_window': 51, 'sg_order': 3},
        ...     compute_fn=lambda: expensive_ica_computation(voltage, capacity)
        ... )
    """

    def __init__(self, cache_dir: Path = Path("artifacts/cache"), version: str = "v1"):
        """Initialize the cache.

        Args:
            cache_dir: Directory for cache files
            version: Cache version (change to invalidate all cached data)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.version = version
        self.stats = {"hits": 0, "misses": 0}

    def _make_key(
        self,
        experiment_id: int,
        cell_id: str,
        rpt_id: Optional[int],
        pipeline_name: str,
        pipeline_params: dict,
    ) -> str:
        """Create deterministic cache key.

        Args:
            experiment_id: Experiment ID
            cell_id: Cell identifier
            rpt_id: RPT index (can be None)
            pipeline_name: Name of the pipeline
            pipeline_params: Pipeline parameters dict

        Returns:
            16-character hash key
        """
        # Sort params for consistent hashing
        params_str = json.dumps(pipeline_params, sort_keys=True, default=str)

        key_parts = [
            f"exp{experiment_id}",
            f"cell{cell_id}",
            f"rpt{rpt_id}" if rpt_id is not None else "rpt_none",
            pipeline_name,
            self.version,
            params_str,
        ]

        key_string = "_".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def _get_cache_path(self, key: str) -> Path:
        """Get path for cache file (organized by first 2 chars of hash).

        Args:
            key: Cache key

        Returns:
            Path to cache file
        """
        subdir = self.cache_dir / key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.pkl"

    def _get_meta_path(self, key: str) -> Path:
        """Get path for metadata file.

        Args:
            key: Cache key

        Returns:
            Path to metadata file
        """
        subdir = self.cache_dir / key[:2]
        return subdir / f"{key}.meta.json"

    def get(
        self,
        experiment_id: int,
        cell_id: str,
        rpt_id: Optional[int],
        pipeline_name: str,
        pipeline_params: dict,
    ) -> Optional[Any]:
        """Retrieve cached result if exists.

        Args:
            experiment_id: Experiment ID
            cell_id: Cell identifier
            rpt_id: RPT index
            pipeline_name: Pipeline name
            pipeline_params: Pipeline parameters

        Returns:
            Cached result or None if not found
        """
        key = self._make_key(
            experiment_id, cell_id, rpt_id, pipeline_name, pipeline_params
        )
        cache_path = self._get_cache_path(key)

        if cache_path.exists():
            self.stats["hits"] += 1
            logger.debug(f"Cache HIT: {pipeline_name} for cell {cell_id} RPT {rpt_id}")
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return None

        self.stats["misses"] += 1
        return None

    def set(
        self,
        experiment_id: int,
        cell_id: str,
        rpt_id: Optional[int],
        pipeline_name: str,
        pipeline_params: dict,
        result: Any,
    ) -> None:
        """Store result in cache.

        Args:
            experiment_id: Experiment ID
            cell_id: Cell identifier
            rpt_id: RPT index
            pipeline_name: Pipeline name
            pipeline_params: Pipeline parameters
            result: Result to cache
        """
        key = self._make_key(
            experiment_id, cell_id, rpt_id, pipeline_name, pipeline_params
        )
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)

        # Save result
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)

        # Save metadata for debugging
        meta = {
            "experiment_id": experiment_id,
            "cell_id": cell_id,
            "rpt_id": rpt_id,
            "pipeline_name": pipeline_name,
            "pipeline_params": pipeline_params,
            "cached_at": datetime.now().isoformat(),
            "version": self.version,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.debug(f"Cache SET: {pipeline_name} for cell {cell_id} RPT {rpt_id}")

    def get_or_compute(
        self,
        experiment_id: int,
        cell_id: str,
        rpt_id: Optional[int],
        pipeline_name: str,
        pipeline_params: dict,
        compute_fn: Callable[[], Any],
    ) -> Any:
        """Return cached result or compute and cache.

        Args:
            experiment_id: Experiment ID
            cell_id: Cell identifier
            rpt_id: RPT index
            pipeline_name: Pipeline name
            pipeline_params: Pipeline parameters
            compute_fn: Function to compute result if not cached

        Returns:
            Cached or computed result
        """
        # Try cache first
        result = self.get(
            experiment_id, cell_id, rpt_id, pipeline_name, pipeline_params
        )

        if result is not None:
            return result

        # Compute (expensive)
        logger.info(f"Computing {pipeline_name} for cell {cell_id} RPT {rpt_id}...")
        result = compute_fn()

        # Cache for next time
        self.set(experiment_id, cell_id, rpt_id, pipeline_name, pipeline_params, result)

        return result

    def clear(self, pipeline_name: Optional[str] = None) -> int:
        """Clear cache (optionally filtered by pipeline name).

        Args:
            pipeline_name: If provided, only clear this pipeline's cache

        Returns:
            Number of cache entries cleared
        """
        count = 0
        for meta_path in self.cache_dir.rglob("*.meta.json"):
            if pipeline_name:
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    if meta.get("pipeline_name") != pipeline_name:
                        continue
                except Exception:
                    continue

            # Remove both .pkl and .meta.json
            cache_path = meta_path.with_suffix("").with_suffix(".pkl")
            cache_path.unlink(missing_ok=True)
            meta_path.unlink()
            count += 1

        logger.info(f"Cleared {count} cache entries")
        return count

    def get_stats(self) -> dict:
        """Return cache hit/miss statistics.

        Returns:
            Dictionary with hits, misses, hit_rate
        """
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {**self.stats, "hit_rate": hit_rate, "total": total}

    def get_size(self) -> dict:
        """Get cache size statistics.

        Returns:
            Dictionary with num_files and total_bytes
        """
        pkl_files = list(self.cache_dir.rglob("*.pkl"))
        total_bytes = sum(f.stat().st_size for f in pkl_files)

        return {
            "num_files": len(pkl_files),
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
        }


# Global cache instance
_cache: Optional[PipelineCache] = None


def get_cache(cache_dir: str = "artifacts/cache", version: str = "v1") -> PipelineCache:
    """Get or create global cache instance.

    Args:
        cache_dir: Cache directory
        version: Cache version

    Returns:
        PipelineCache instance
    """
    global _cache
    if _cache is None:
        _cache = PipelineCache(Path(cache_dir), version)
    return _cache


def reset_cache() -> None:
    """Reset global cache instance."""
    global _cache
    _cache = None
