"""Classification label derivation from task names + joint angle thresholds.

This is THE dual-use bridge: the same BIDS data that serves regression
(continuous joint angles) also serves classification (discrete gesture labels).

Labels are derived post-hoc — never during collection. The collection app
only knows about regression tasks. This module converts task names + joint
angle patterns into discrete labels.

For structured tasks (powerGrip, thumbCmcIso, etc.):
    label = task_name for active portion, "rest" for rest portion

For freeform:
    cluster joint angles → discover gesture segments → label by closest match
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema


# ── Label derivation ──────────────────────────────────────────────────────────


def derive_labels(
    merged_df: pd.DataFrame,
    task: str,
    target_columns: list[str] | None = None,
) -> np.ndarray:
    """Derive classification labels from merged physio+targets data.

    Parameters
    ----------
    merged_df : DataFrame
        Merged physio + targets data (joined by timestamp).
    task : str
        Task name (e.g. "powerGrip", "thumbCmcIso").
    target_columns : list[str], optional
        Available target columns. If None, auto-detect from schema.

    Returns
    -------
    labels : np.ndarray of str
        One label per row: task name for active, "rest" for rest.
    """
    if target_columns is None:
        target_columns = [c for c in schema.TARGET_COLUMNS if c in merged_df.columns]

    if not target_columns:
        # No targets available — label everything as the task name
        return np.array([task] * len(merged_df), dtype=object)

    # Detect active vs rest periods from joint angle movement
    active_mask = _detect_active_period(merged_df, target_columns)

    labels = np.where(active_mask, task, "rest")
    return labels.astype(object)


def _detect_active_period(
    df: pd.DataFrame,
    target_columns: list[str],
    *,
    movement_threshold: float = 2.0,
    min_rest_duration_s: float = 0.5,
) -> np.ndarray:
    """Detect which samples are "active" (moving) vs "rest" (still).

    Uses the standard deviation of joint angles in a sliding window.
    If the std exceeds the threshold, the sample is "active".

    Parameters
    ----------
    df : DataFrame
        Merged data with target columns.
    target_columns : list[str]
        Joint angle columns to use for movement detection.
    movement_threshold : float
        Minimum std (in degrees) to consider a window "active".
    min_rest_duration_s : float
        Minimum rest duration to label as "rest" (avoids flicker).

    Returns
    -------
    active_mask : np.ndarray of bool
        True = active (moving), False = rest (still).
    """
    n = len(df)
    if n == 0:
        return np.array([], dtype=bool)

    targets = df[target_columns].values.astype(float)

    # Mask out rows where the camera lost tracking (mp_valid == 0).
    # When MediaPipe fails, write_targets() writes all-zeros for joint angles.
    # Those zeros are NOT real angles — including them in the diff would create
    # large false-movement spikes at the dropout boundaries (e.g. 45° → 0°).
    # Setting them to NaN before the diff causes np.nansum to treat any diff
    # that involves an invalid frame as 0, suppressing the false spike.
    if "mp_valid" in df.columns:
        invalid_mask = df["mp_valid"].values == 0
        targets[invalid_mask, :] = np.nan

    # Compute movement as frame-to-frame difference magnitude.
    # Use nansum so that diffs touching an invalid (NaN) frame contribute 0.
    if n < 2:
        return np.array([True], dtype=bool)

    diff = np.abs(np.diff(targets, axis=0))
    movement = np.nansum(diff, axis=1)  # total movement across all DOFs
    movement = np.concatenate([[movement[0]], movement])  # pad to same length

    # Smooth with a sliding window (5-sample moving average)
    window = 5
    if n >= window:
        kernel = np.ones(window) / window
        movement = np.convolve(movement, kernel, mode="same")

    active_mask = movement > movement_threshold

    # Enforce minimum rest duration to avoid flicker
    # (short active blips during rest are ignored)
    active_mask = _remove_short_rests(active_mask, min_rest_duration_s, df)

    return active_mask


def _remove_short_rests(
    active_mask: np.ndarray,
    min_rest_duration_s: float,
    df: pd.DataFrame,
) -> np.ndarray:
    """Remove short rest periods (less than min_rest_duration_s) from the mask.

    This prevents flicker between active/rest labels during transitions.
    """
    if "t_monotonic_ns" not in df.columns or len(df) < 2:
        return active_mask

    timestamps = df["t_monotonic_ns"].values
    if len(timestamps) < 2:
        return active_mask

    # Estimate sampling period
    dt_ns = np.median(np.diff(timestamps))
    if dt_ns == 0:
        return active_mask

    min_rest_samples = int(min_rest_duration_s * 1e9 / dt_ns)
    if min_rest_samples < 2:
        return active_mask

    # Find rest segments (consecutive False) shorter than min_rest_samples
    # and flip them to True (active)
    result = active_mask.copy()
    i = 0
    while i < len(result):
        if not result[i]:
            # Start of a rest segment
            j = i
            while j < len(result) and not result[j]:
                j += 1
            rest_len = j - i
            if rest_len < min_rest_samples:
                result[i:j] = True  # too short to be real rest → active
            i = j
        else:
            i += 1

    return result


# ── Events TSV generation ─────────────────────────────────────────────────────


def generate_events_tsv(
    sub: str,
    ses: str,
    *,
    data_root=None,
) -> pd.DataFrame:
    """Generate an events.tsv for a session (BIDS-compliant classification labels).

    Scans all runs in the session, derives labels, and produces a tabular
    events file with onset, duration, trial_type, and run columns.
    """
    from . import loader, naming, paths

    sdir = paths.session_dir(sub, ses, data_root=data_root)
    session = loader.load_session(sub, ses, data_root=data_root)

    events_rows = []
    for run in session.runs:
        if run.task in schema.FREEFORM_TASKS:
            continue  # skip freeform for discrete labels
        if run.physio.empty or run.targets.empty:
            continue

        merged = loader._join_by_timestamp(
            run.physio, run.targets,
            schema.PHYSIO_TIMESTAMP, schema.TARGETS_TIMESTAMP,
        )
        if merged.empty:
            continue

        target_cols = [c for c in schema.TARGET_COLUMNS if c in merged.columns]
        labels = derive_labels(merged, run.task, target_cols)

        # Find onset/duration of each active segment
        timestamps = merged[schema.PHYSIO_TIMESTAMP].values
        segments = _find_segments(labels, timestamps)

        for seg_label, onset_ns, duration_ns in segments:
            events_rows.append({
                schema.EVENTS_ONSET: onset_ns / 1e9,  # convert to seconds
                schema.EVENTS_DURATION: duration_ns / 1e9,
                schema.EVENTS_TRIAL_TYPE: seg_label,
                schema.EVENTS_RUN: run.run,
            })

    return pd.DataFrame(events_rows, columns=schema.EVENTS_COLUMNS)


def _find_segments(
    labels: np.ndarray,
    timestamps: np.ndarray,
) -> list[tuple[str, float, float]]:
    """Find contiguous segments of the same label.

    Returns list of (label, onset_ns, duration_ns) tuples.
    """
    if len(labels) == 0:
        return []

    segments = []
    current_label = labels[0]
    start_idx = 0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            onset = timestamps[start_idx]
            duration = timestamps[i - 1] - timestamps[start_idx]
            segments.append((str(current_label), float(onset), float(duration)))
            current_label = labels[i]
            start_idx = i

    # Last segment
    onset = timestamps[start_idx]
    duration = timestamps[-1] - timestamps[start_idx]
    segments.append((str(current_label), float(onset), float(duration)))

    return segments


# ── Freeform clustering (future work) ─────────────────────────────────────────


def cluster_freeform(
    targets_df: pd.DataFrame,
    target_columns: list[str] | None = None,
) -> np.ndarray:
    """Cluster freeform joint angles to discover gesture segments.

    This is a placeholder for future work. Currently returns "freeform"
    for all samples. A proper implementation would use KMeans or HMM
    to discover distinct hand postures during freeform movement.
    """
    if target_columns is None:
        target_columns = [c for c in schema.TARGET_COLUMNS if c in targets_df.columns]

    # TODO: implement clustering (KMeans on joint angle windows)
    return np.array(["freeform"] * len(targets_df), dtype=object)
