"""High-resolution monotonic timer for timestamp collection.

On Windows, ``time.monotonic_ns()`` has ~15.6ms resolution (the default
timer tick at 64Hz).  This is insufficient for sub-2ms LED sync precision
— two samples taken 1ms apart may get identical timestamps.

This module provides ``now_ns()`` backed by ``time.perf_counter_ns()``
which uses QueryPerformanceCounter (QPC) on Windows — typically 100ns
resolution (10MHz frequency).  This is ~150,000x finer than
``time.monotonic_ns()``.

The column name ``t_monotonic_ns`` in the BIDS schema is kept for backward
compatibility (data contract), but the value is now QPC-based.

``time.perf_counter_ns()`` is monotonic (never goes backwards) on all
platforms, making it a safe drop-in replacement for ``time.monotonic_ns()``.
"""

from __future__ import annotations

import time

# Cache the clock resolution once at import time
_RESOLUTION_NS: int = int(time.get_clock_info("perf_counter").resolution * 1e9)


def now_ns() -> int:
    """Return current monotonic time in nanoseconds (QPC-backed, ~100ns resolution).

    This is a drop-in replacement for ``time.monotonic_ns()`` but with
    ~150,000x finer resolution on Windows.
    """
    return time.perf_counter_ns()


def timer_resolution_ns() -> int:
    """Return the resolution of the timer in nanoseconds.

    On Windows with QPC, this is typically 100ns (10MHz frequency).
    """
    return _RESOLUTION_NS
