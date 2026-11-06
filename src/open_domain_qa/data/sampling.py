"""
Deterministic sampling utility for evaluation and calibration splits.
"""

import random
from typing import Any, Dict, List, Optional


def sample_fixed_subset(
    records: List[Dict[str, Any]],
    sample_size: Optional[int] = None,
    size: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Samples exactly n items deterministically using a fixed random seed."""
    n = sample_size if sample_size is not None else (size if size is not None else 300)
    rng = random.Random(seed)
    if n > len(records):
        raise ValueError(f"Requested {n} samples, but dataset only contains {len(records)} items.")
    return rng.sample(records, n)
