"""Profiling utilities for MLX Metal performance analysis."""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict

import mlx.core as mx


@dataclass
class TimingResult:
    """Stores the name and elapsed time of a profiled operation."""

    name: str
    elapsed_ms: float

    def __repr__(self) -> str:
        return f"{self.name}: {self.elapsed_ms:.2f}ms"


@contextmanager
def timer(name: str = "op", verbose: bool = True):
    """Context manager that times an MLX operation.

    Calls ``mx.eval()`` before and after the block to ensure the lazy
    computation graph is fully materialised, giving accurate wall-clock
    timings of Metal GPU execution.

    Example::

        with timer("forward"):
            y = model(x)
    """
    mx.eval()  # drain any pending work
    start = time.perf_counter()
    result = TimingResult(name, 0.0)
    yield result
    mx.eval()  # materialise everything produced inside the block
    result.elapsed_ms = (time.perf_counter() - start) * 1000.0
    if verbose:
        print(result)


def get_memory_info() -> Dict[str, float]:
    """Return current MLX memory usage in megabytes.

    Keys:
        ``active_mb``  -- memory currently in use
        ``peak_mb``    -- peak memory used since last reset
        ``cache_mb``   -- memory held in the allocator cache
    """
    return {
        "active_mb": mx.metal.get_active_memory() / 1e6,
        "peak_mb": mx.metal.get_peak_memory() / 1e6,
        "cache_mb": mx.metal.get_cache_memory() / 1e6,
    }


def reset_peak_memory() -> None:
    """Reset the peak-memory watermark."""
    mx.metal.reset_peak_memory()
