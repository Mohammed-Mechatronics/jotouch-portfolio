"""Windowed feature extraction from BIDS physio data.

Extracts features from FSR signals in sliding windows. Used by both
regression and classification consumers.

Window features per channel:
    mean, std, min, max, range, rms

Usage:
    from core.features import extract_windows, extract_trial_features

    # Sliding windows (for frame-by-frame regression)
    X_windowed, timestamps = extract_windows(physio_df, window_ms=200, step_ms=50)

    # Per-trial features (for classification)
    X_trial, labels = extract_trial_features(physio_df, targets_df, task="powerGrip")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema

# Default feature set
DEFAULT_FEATURES = ("mean", "std", "min", "max", "range", "rms")


def extract_windows(
    physio_df: pd.DataFrame,
    *,
    window_ms: float = 200.0,
    step_ms: float = 50.0,
    sensor_cols: list[str] | None = None,
    features: tuple[str, ...] = DEFAULT_FEATURES,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract sliding-window features from physio data.

    Parameters
    ----------
    physio_df : DataFrame
        Physio data with timestamp and FSR columns.
    window_ms : float
        Window duration in milliseconds.
    step_ms : float
        Step size between windows in milliseconds.
    sensor_cols : list[str], optional
        FSR column names. Auto-detected if None.
    features : tuple[str, ...]
        Features to extract per channel.

    Returns
    -------
    X : (N_windows, n_sensors * n_features) float32
        Feature matrix.
    timestamps : (N_windows,) int64
        Center timestamp of each window in nanoseconds.
    """
    if physio_df.empty:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int64)

    if sensor_cols is None:
        sensor_cols = [c for c in physio_df.columns if c.startswith("fsr")]
    if not sensor_cols:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int64)

    ts_col = schema.PHYSIO_TIMESTAMP
    if ts_col not in physio_df.columns:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int64)

    timestamps = physio_df[ts_col].values.astype(np.int64)
    signals = physio_df[sensor_cols].values.astype(np.float32)

    window_ns = int(window_ms * 1e6)
    step_ns = int(step_ms * 1e6)

    if len(timestamps) < 2:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int64)

    dt = np.median(np.diff(timestamps))
    if dt <= 0:
        dt = int(1e9 / 60)  # assume 60Hz

    window_samples = max(1, int(window_ns / dt))
    step_samples = max(1, int(step_ns / dt))

    if len(signals) < window_samples:
        # Not enough data for one window — extract from what we have
        feats = _compute_features(signals, features)
        return feats.reshape(1, -1), np.array([timestamps[0]], dtype=np.int64)

    windows = []
    window_ts = []

    for start in range(0, len(signals) - window_samples + 1, step_samples):
        end = start + window_samples
        window_data = signals[start:end]
        feats = _compute_features(window_data, features)
        windows.append(feats)
        window_ts.append(timestamps[start + window_samples // 2])

    if not windows:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int64)

    X = np.stack(windows).astype(np.float32)
    ts = np.array(window_ts, dtype=np.int64)
    return X, ts


def extract_trial_features(
    physio_df: pd.DataFrame,
    targets_df: pd.DataFrame | None = None,
    *,
    task: str = "",
    sensor_cols: list[str] | None = None,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    edge_trim: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-trial features (one feature vector per run).

    Used for classification where each run = one trial.

    Parameters
    ----------
    physio_df : DataFrame
        Physio data for a single run.
    targets_df : DataFrame, optional
        Targets data (used to detect active vs rest periods).
    task : str
        Task name (becomes the label).
    sensor_cols : list[str], optional
        FSR column names. Auto-detected if None.
    features : tuple[str, ...]
        Features to extract per channel.
    edge_trim : int
        Number of samples to trim from start and end (removes transition artifacts).

    Returns
    -------
    X : (1, n_sensors * n_features) float32
        Single feature vector for the trial.
    y : (1,) str
        Label (task name).
    """
    if physio_df.empty:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=object)

    if sensor_cols is None:
        sensor_cols = [c for c in physio_df.columns if c.startswith("fsr")]
    if not sensor_cols:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=object)

    signals = physio_df[sensor_cols].values.astype(np.float32)

    # Edge trim
    if edge_trim > 0 and len(signals) > 2 * edge_trim:
        signals = signals[edge_trim:-edge_trim]

    # If targets provided, extract only the active portion
    if targets_df is not None and not targets_df.empty:
        from . import events
        target_cols = [c for c in schema.TARGET_COLUMNS if c in targets_df.columns]
        if target_cols:
            # Join physio and targets by timestamp
            from .loader import _join_by_timestamp
            merged = _join_by_timestamp(
                physio_df, targets_df,
                schema.PHYSIO_TIMESTAMP, schema.TARGETS_TIMESTAMP,
            )
            if not merged.empty:
                active_mask = events._detect_active_period(merged, target_cols)
                active_signals = merged.loc[active_mask, sensor_cols].values.astype(np.float32)
                if len(active_signals) > 2 * edge_trim:
                    signals = active_signals[edge_trim:-edge_trim]

    if len(signals) < 2:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=object)

    feats = _compute_features(signals, features)
    X = feats.reshape(1, -1).astype(np.float32)
    y = np.array([task], dtype=object)
    return X, y


def _compute_features(
    window: np.ndarray,
    features: tuple[str, ...] = DEFAULT_FEATURES,
) -> np.ndarray:
    """Compute features for a window of shape (N_samples, N_sensors).

    Returns a 1D array of length N_sensors * N_features.
    """
    out = []
    for c in range(window.shape[1]):
        s = window[:, c]
        for name in features:
            if name == "mean":
                out.append(float(s.mean()))
            elif name == "std":
                out.append(float(s.std()))
            elif name == "min":
                out.append(float(s.min()))
            elif name == "max":
                out.append(float(s.max()))
            elif name == "range":
                out.append(float(s.max() - s.min()))
            elif name == "rms":
                out.append(float(np.sqrt(np.mean(s ** 2))))
            elif name == "median":
                out.append(float(np.median(s)))
            else:
                raise ValueError(f"Unknown feature: {name}")
    return np.array(out, dtype=np.float32)


def feature_names(
    sensor_cols: list[str],
    features: tuple[str, ...] = DEFAULT_FEATURES,
) -> list[str]:
    """Return human-readable feature names in matrix-column order.

    >>> feature_names(["fsr0", "fsr1"], ("mean", "std"))
    ['fsr0_mean', 'fsr0_std', 'fsr1_mean', 'fsr1_std']
    """
    return [f"{sensor}_{feat}" for sensor in sensor_cols for feat in features]
