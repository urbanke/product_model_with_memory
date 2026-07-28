"""Header-only cache completeness scan: correct and truncation-safe."""

import tempfile
import time
from pathlib import Path

from product_model_with_memory.fast_tables import (
    _cache_file,
    _cached_levels,
    build_tables_fast,
    TableCache,
)


def test_warm_cache_scan_is_headers_only_and_correct():
    with tempfile.TemporaryDirectory() as tmp:
        r_values = [1, 2, 5, 9]
        c1 = build_tables_fast(
            max_L=4, r_values=r_values, cache_dir=tmp, materialize=False
        )
        assert isinstance(c1, TableCache)
        # warm restart: nothing missing, identical cache metadata
        t0 = time.time()
        c2 = build_tables_fast(
            max_L=4, r_values=r_values, cache_dir=tmp, materialize=False
        )
        assert time.time() - t0 < 5.0
        assert c2.cache_path == c1.cache_path
        for r in r_values:
            assert _cached_levels(c1.cache_path, r) == 4

        # truncated file must be detected as incomplete and rebuilt
        victim = _cache_file(c1.cache_path, 5)
        data = victim.read_bytes()
        victim.write_bytes(data[: len(data) // 2])
        assert _cached_levels(c1.cache_path, 5) == 0
        build_tables_fast(
            max_L=4, r_values=r_values, cache_dir=tmp, materialize=False
        )
        assert _cached_levels(c1.cache_path, 5) == 4


def test_headers_scan_agrees_with_full_load():
    with tempfile.TemporaryDirectory() as tmp:
        build_tables_fast(
            max_L=3, r_values=[1, 4], cache_dir=tmp, materialize=False
        )
        from product_model_with_memory.fast_tables import _load_cached

        cache_path = next(Path(tmp).iterdir())
        for r in [1, 4]:
            assert _cached_levels(cache_path, r) == _load_cached(
                cache_path, r
            ).shape[0]
