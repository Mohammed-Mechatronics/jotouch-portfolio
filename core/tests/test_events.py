"""Tests for core.events — classification label derivation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from core.events import derive_labels, _detect_active_period, _find_segments
from core.schema import PHYSIO_TIMESTAMP, TARGET_COLUMNS


def _make_merged_df(n: int = 100, active_pattern: bool = True) -> pd.DataFrame:
    """Create a merged physio+targets DataFrame for testing.

    If active_pattern=True, creates a clear active→rest→active pattern.
    """
    ts = np.arange(n, dtype=np.int64) * 16_666_666  # 60Hz
    data = {PHYSIO_TIMESTAMP: ts}

    if active_pattern:
        # First 30 samples: active (large frame-to-frame changes)
        # Middle 40 samples: rest (constant values, no changes) — 667ms > 500ms threshold
        # Last 30 samples: active again
        angles = np.zeros((n, 3))
        # Active: sine wave with large amplitude = clear movement
        t = np.arange(30)
        angles[:30, 0] = 45 + 20 * np.sin(t * 0.5)
        angles[:30, 1] = 30 + 15 * np.sin(t * 0.3)
        angles[:30, 2] = 50 + 10 * np.sin(t * 0.7)
        # Rest: constant (40 samples at 60Hz = 667ms > 500ms min rest)
        angles[30:70, :] = 45.0
        # Active: different sine pattern
        t2 = np.arange(30)
        angles[70:, 0] = 40 + 25 * np.sin(t2 * 0.4)
        angles[70:, 1] = 35 + 20 * np.sin(t2 * 0.6)
        angles[70:, 2] = 55 + 15 * np.sin(t2 * 0.2)
    else:
        angles = np.random.uniform(0, 90, size=(n, 3))

    for i, col in enumerate(TARGET_COLUMNS[:3]):
        data[col] = angles[:, i]

    return pd.DataFrame(data)


class TestDeriveLabels:
    def test_basic_labeling(self):
        df = _make_merged_df(100, active_pattern=True)
        target_cols = [c for c in TARGET_COLUMNS if c in df.columns]
        labels = derive_labels(df, "powerGrip", target_cols)
        assert len(labels) == 100
        assert set(np.unique(labels)).issubset({"powerGrip", "rest"})

    def test_no_targets(self):
        df = pd.DataFrame({PHYSIO_TIMESTAMP: np.arange(10)})
        labels = derive_labels(df, "powerGrip", [])
        assert len(labels) == 10
        assert all(l == "powerGrip" for l in labels)

    def test_empty_df(self):
        labels = derive_labels(pd.DataFrame(), "powerGrip", TARGET_COLUMNS)
        assert len(labels) == 0


class TestDetectActivePeriod:
    def test_active_rest_pattern(self):
        df = _make_merged_df(100, active_pattern=True)
        target_cols = [c for c in TARGET_COLUMNS if c in df.columns]
        mask = _detect_active_period(df, target_cols, movement_threshold=0.5)
        assert len(mask) == 100
        # Middle section (samples 40-60) should be rest (False)
        assert not mask[50]
        # Active sections should be True (at least some)
        assert mask[5] or mask[10] or mask[15]

    def test_all_active(self):
        df = _make_merged_df(50, active_pattern=False)
        target_cols = [c for c in TARGET_COLUMNS if c in df.columns]
        mask = _detect_active_period(df, target_cols, movement_threshold=0.01)
        # With random data and low threshold, most should be active
        assert mask.sum() > 30

    def test_mp_valid_0_rows_do_not_create_false_movement(self):
        """Rows with mp_valid==0 write all-zero angles; those zeros must not
        look like movement relative to real angle values (e.g. 90 deg).

        Without the fix, the diff 90→0 generates a large movement spike and
        the sample just before the dropout is incorrectly labelled "active".
        With the fix, invalid rows are masked to NaN → nan_to_num → 0, so
        the diff contribution is 0 and the surrounding constant region stays
        labelled "rest".
        """
        n = 40
        ts = np.arange(n, dtype=np.int64) * 16_666_666
        # All frames are genuinely at rest (constant 45 degrees)
        angles = np.full((n, 3), 45.0)
        # Middle 10 frames: camera lost tracking → write_targets wrote zeros
        dropout_start, dropout_end = 15, 25
        angles[dropout_start:dropout_end, :] = 0.0

        data = {PHYSIO_TIMESTAMP: ts}
        for i, col in enumerate(TARGET_COLUMNS[:3]):
            data[col] = angles[:, i]
        # Mark those frames as tracking-lost
        mp_valid = np.ones(n, dtype=int)
        mp_valid[dropout_start:dropout_end] = 0
        data["mp_valid"] = mp_valid

        df = pd.DataFrame(data)
        target_cols = TARGET_COLUMNS[:3]

        mask = _detect_active_period(df, target_cols, movement_threshold=2.0)

        # The whole sequence is rest — no frame should be labelled "active"
        # (the dropout zeros must be excluded from movement computation)
        assert mask.sum() == 0, (
            f"Expected all rest (0 active frames) but got {mask.sum()} active; "
            "mp_valid==0 zeros are being counted as movement."
        )

    def test_mp_valid_column_absent_unchanged_behaviour(self):
        """When mp_valid is not in the df, behaviour is identical to before."""
        df = _make_merged_df(100, active_pattern=True)
        target_cols = [c for c in TARGET_COLUMNS if c in df.columns]
        assert "mp_valid" not in df.columns
        mask = _detect_active_period(df, target_cols, movement_threshold=0.5)
        # Same basic sanity: rest segment is correctly identified
        assert not mask[50]


class TestFindSegments:
    def test_simple_segments(self):
        labels = np.array(["a", "a", "a", "rest", "rest", "a", "a"])
        timestamps = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64) * 1_000_000_000
        segments = _find_segments(labels, timestamps)
        assert len(segments) == 3
        assert segments[0][0] == "a"
        assert segments[1][0] == "rest"
        assert segments[2][0] == "a"

    def test_single_segment(self):
        labels = np.array(["a", "a", "a"])
        timestamps = np.array([0, 1, 2], dtype=np.int64) * 1_000_000_000
        segments = _find_segments(labels, timestamps)
        assert len(segments) == 1
        assert segments[0][0] == "a"

    def test_empty(self):
        segments = _find_segments(np.array([]), np.array([], dtype=np.int64))
        assert len(segments) == 0
