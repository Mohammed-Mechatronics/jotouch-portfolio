"""Tests for apps.collection.timer — high-resolution monotonic timer.

On Windows, ``time.monotonic_ns()`` has ~15.6ms resolution (the default
timer tick).  This is insufficient for sub-2ms LED sync precision.  The
``timer`` module provides ``now_ns()`` backed by ``time.perf_counter_ns()``
which uses QueryPerformanceCounter (QPC) on Windows — ~100ns resolution.

The column name ``t_monotonic_ns`` is kept (data contract) but the value
is now QPC-based, not ``time.monotonic_ns()`` based.
"""

from __future__ import annotations

import time
import pytest

from apps.collection.timer import now_ns, timer_resolution_ns


class TestNowNs:
    def test_returns_int(self):
        ts = now_ns()
        assert isinstance(ts, int)

    def test_monotonic_increasing(self):
        ts1 = now_ns()
        ts2 = now_ns()
        assert ts2 >= ts1

    def test_resolution_better_than_1ms(self):
        """now_ns() must have sub-millisecond resolution."""
        # Collect consecutive timestamps that differ
        diffs = []
        t0 = now_ns()
        for _ in range(10000):
            t = now_ns()
            if t != t0:
                diffs.append(t - t0)
                t0 = t
                if len(diffs) >= 20:
                    break
        assert len(diffs) >= 5, "Could not collect enough changing timestamps"
        assert min(diffs) < 1_000_000, f"Resolution {min(diffs)}ns >= 1ms"

    def test_resolution_much_better_than_monotonic(self):
        """now_ns() resolution must be at least 10x better than monotonic."""
        # monotonic resolution on Windows is ~15.6ms
        monotonic_res = 15_600_000  # 15.6ms in ns
        assert timer_resolution_ns() < monotonic_res / 10

    def test_consecutive_calls_differ_within_loop(self):
        """Two calls in quick succession should often produce different values."""
        same_count = 0
        diff_count = 0
        for _ in range(1000):
            t1 = now_ns()
            t2 = now_ns()
            if t1 == t2:
                same_count += 1
            else:
                diff_count += 1
        # With 100ns resolution, most consecutive calls should differ
        assert diff_count > 0, "No consecutive calls differed — resolution too coarse"


class TestTimerResolution:
    def test_returns_int(self):
        res = timer_resolution_ns()
        assert isinstance(res, int)

    def test_less_than_1ms(self):
        """Reported resolution must be sub-millisecond."""
        assert timer_resolution_ns() < 1_000_000

    def test_less_than_1us(self):
        """On modern systems, QPC resolution is typically 100ns."""
        assert timer_resolution_ns() <= 1_000
