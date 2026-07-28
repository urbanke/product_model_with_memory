"""The parallel evaluation path must reproduce the serial path exactly."""

import random
import tempfile

from product_model_with_memory.codelength import depth_averaged_codelength_profiles


def test_parallel_matches_serial_exactly():
    rng = random.Random(3)
    profiles = {
        i: tuple(sorted(rng.randint(1, 60) for _ in range(rng.randint(1, 6))))
        for i in range(12)
    }
    with tempfile.TemporaryDirectory() as tmp:
        serial = depth_averaged_codelength_profiles(
            profiles, d=64, l_max=6, cache_dir=tmp, jobs=1
        )
        par = depth_averaged_codelength_profiles(
            profiles, d=64, l_max=6, cache_dir=tmp, jobs=3
        )
    for i in profiles:
        assert serial[i].log2_q_by_depth == par[i].log2_q_by_depth
        assert serial[i].log2_q_avg == par[i].log2_q_avg
