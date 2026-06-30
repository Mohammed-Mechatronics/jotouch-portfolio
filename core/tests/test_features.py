"""Tests for core.features — windowed feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.features import extract_windows, extract_trial_features, _compute_features, feature_names
from core.schema import PHYSIO_TIMESTAMP


def _make_physio_df(n: int = 100, n_sensors: int = 4) -> pd.DataFrame:
    """Create a minimal physio DataFrame for testing."""
    data = {PHYSIO_TIMESTAMP: np.arange(n, dtype=np.int64) * 16_666_666}
    for i in range(n_sensors):
        data[f"fsr{i}"] = np.random.randint(100, 900, size=n).astype(float)
    return pd.DataFrame(data)


class TestExtractWindows:
    def test_basic_extraction(self):
        df = _make_physio_df(200)
        X, ts = extract_windows(df, window_ms=200, step_ms=50)
        assert X.ndim == 2
        assert X.shape[1] == 4 * 6  # 4 sensors × 6 features
        assert len(ts) == X.shape[0]
        assert X.dtype == np.float32

    def test_empty_df(self):
        df = pd.DataFrame()
        X, ts = extract_windows(df)
        assert X.shape[0] == 0
        assert len(ts) == 0

    def test_small_df(self):
        df = _make_physio_df(5)
        X, ts = extract_windows(df, window_ms=200, step_ms=50)
        # Should produce at least 1 window even if small
        assert X.shape[0] >= 1

    def test_window_count(self):
        # 100 samples at 60Hz = ~1.67s
        # 200ms window = ~12 samples
        # 50ms step = ~3 samples
        # Expected windows ≈ (100 - 12) / 3 + 1 ≈ 30
        df = _make_physio_df(100)
        X, ts = extract_windows(df, window_ms=200, step_ms=50)
        assert 20 <= X.shape[0] <= 35  # allow some tolerance


class TestExtractTrialFeatures:
    def test_basic_trial(self):
        physio = _make_physio_df(100)
        X, y = extract_trial_features(physio, task="powerGrip")
        assert X.shape[0] == 1  # one trial
        assert X.shape[1] == 4 * 6  # 4 sensors × 6 features
        assert y[0] == "powerGrip"

    def test_empty_physio(self):
        X, y = extract_trial_features(pd.DataFrame(), task="powerGrip")
        assert X.shape[0] == 0
        assert len(y) == 0

    def test_with_targets(self):
        from core.schema import TARGETS_TIMESTAMP, TARGET_COLUMNS
        physio = _make_physio_df(100)
        targets = pd.DataFrame({
            TARGETS_TIMESTAMP: physio[PHYSIO_TIMESTAMP].values,
            **{col: np.random.uniform(0, 90, size=100) for col in TARGET_COLUMNS[:3]},
        })
        X, y = extract_trial_features(physio, targets, task="thumbCmcIso")
        assert X.shape[0] == 1
        assert y[0] == "thumbCmcIso"


class TestComputeFeatures:
    def test_mean(self):
        window = np.array([[1, 10], [3, 20], [5, 30]], dtype=np.float32)
        feats = _compute_features(window, ("mean",))
        assert feats[0] == pytest.approx(3.0)  # mean of [1,3,5]
        assert feats[1] == pytest.approx(20.0)  # mean of [10,20,30]

    def test_rms(self):
        window = np.array([[3], [4]], dtype=np.float32)
        feats = _compute_features(window, ("rms",))
        assert feats[0] == pytest.approx(3.535, rel=0.01)  # sqrt((9+16)/2)

    def test_all_features(self):
        window = np.array([[1, 10], [3, 20], [5, 30]], dtype=np.float32)
        feats = _compute_features(window)
        assert len(feats) == 2 * 6  # 2 sensors × 6 features


class TestFeatureNames:
    def test_naming(self):
        names = feature_names(["fsr0", "fsr1"], ("mean", "std"))
        assert names == ["fsr0_mean", "fsr0_std", "fsr1_mean", "fsr1_std"]
