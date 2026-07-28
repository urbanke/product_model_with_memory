"""A simple product model but for sources with memory."""

from product_model_with_memory.codelength import (
    C_STAR,
    DepthAveragedCodelength,
    default_l_max,
    depth_averaged_codelength,
    needed_r_values,
    profile_of,
)
from product_model_with_memory.corpus import (
    empirical_conditional_entropy_bits,
    empirical_entropy_bits,
    load_tokens,
    prefix_counts,
)
from product_model_with_memory.fast_tables import build_tables_fast

__version__ = "0.1.0"

__all__ = [
    "C_STAR",
    "DepthAveragedCodelength",
    "build_tables_fast",
    "default_l_max",
    "depth_averaged_codelength",
    "empirical_conditional_entropy_bits",
    "empirical_entropy_bits",
    "load_tokens",
    "needed_r_values",
    "prefix_counts",
    "profile_of",
]
