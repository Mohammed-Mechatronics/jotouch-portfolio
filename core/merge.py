"""Merge engine — combines physio + camera + targets into a single 100Hz DataFrame.

This module is the heart of the sample promotion pipeline. It takes the three
per-run CSVs (physio at 100Hz, camera at ~30Hz, targets at 100Hz), applies LED
sync correction to camera timestamps, forward-fills camera data onto the physio
timestamp grid, and joins targets by exact timestamp match.

The result is a single DataFrame with one row per FSR sample (100Hz) containing
all three modalities, ready for ML consumption.

Column layout of the merged DataFrame::

    t_monotonic_ns          # master timestamp (from physio)
    sample_idx              # row index within the run
    run_idx                 # global run index (0-based)
    phase                   # PREP / RECORD / REST
    participant_id          # e.g. P01
    session_id              # e.g. S01
    task                    # e.g. thumbCmcIso
    run                     # run number (int)
    rep                     # rep number within task (1, 2, 3)
    trial_type              # task name during RECORD, None otherwise
    fsr0, fsr1, fsr2, fsr3  # FSR signals @ 100Hz
    led_fsr                 # FSR LED state
    cue_event               # cue event marker
    cam_frame_new           # bool: True when a new camera frame arrived
    mp_valid, mp_confidence, mp_handedness  # camera metadata (fwd-filled)
    mp_lm00_x ... mp_lm20_z # 63 MediaPipe landmarks (fwd-filled)
    led_cam                 # camera LED brightness (fwd-filled)
    target_thumb_cmc_flex ... target_pinky_dip_flex  # 15 joint angles
    quality_flag            # from physio
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import naming, paths, schema
from .metadata import load_led_sync

logger = logging.getLogger(__name__)


# ── LED sync correction ──────────────────────────────────────────────────────


@dataclass
class SyncCorrection:
    """Linear correction: t_cam_corrected = a * t_cam + b."""

    a: float = 1.0
    b: float = 0.0
    passed: bool = False
    method: str = ""

    @property
    def is_identity(self) -> bool:
        return self.a == 1.0 and self.b == 0.0


def load_sync_correction(
    sub: str,
    ses: str,
    *,
    data_root: Path | None = None,
) -> SyncCorrection:
    """Load LED sync correction coefficients from led_sync.json."""
    sync = load_led_sync(sub, ses, data_root=data_root)
    if sync.passed and sync.a is not None and sync.b is not None:
        return SyncCorrection(a=sync.a, b=sync.b, passed=True, method=sync.method)
    return SyncCorrection(passed=False, method=sync.method)


def apply_sync_correction(
    camera_df: pd.DataFrame,
    correction: SyncCorrection,
) -> pd.DataFrame:
    """Apply linear correction to camera timestamps in-place (returns a copy)."""
    df = camera_df.copy()
    if correction.is_identity or schema.CAMERA_TIMESTAMP not in df.columns:
        return df
    df[schema.CAMERA_TIMESTAMP] = (
        correction.a * df[schema.CAMERA_TIMESTAMP].astype(np.float64) + correction.b
    ).astype(np.int64)
    return df


# ── Forward-fill camera onto physio grid ─────────────────────────────────────


def forward_fill_camera(
    physio_ts_ns: np.ndarray,
    cam_df: pd.DataFrame,
) -> pd.DataFrame:
    """Forward-fill camera data onto the physio timestamp grid.

    For each physio timestamp, finds the most recent camera frame with
    ``cam_ts <= physio_ts`` and copies its values. Sets ``cam_frame_new=True``
    on rows where a new camera frame first becomes available.

    Parameters
    ----------
    physio_ts_ns : np.ndarray
        Physio timestamps (int64, nanoseconds), sorted ascending.
    cam_df : pd.DataFrame
        Camera DataFrame with corrected timestamps, sorted by cam_ts_ns.

    Returns
    -------
    pd.DataFrame
        Camera data aligned to physio timestamps, with an added
        ``cam_frame_new`` boolean column. One row per physio timestamp.
    """
    if cam_df.empty or len(physio_ts_ns) == 0:
        return pd.DataFrame()

    cam_ts = cam_df[schema.CAMERA_TIMESTAMP].values.astype(np.int64)
    cam_data = cam_df.drop(columns=[schema.CAMERA_TIMESTAMP])

    # For each physio timestamp, find the index of the last camera frame
    # with cam_ts <= physio_ts (searchsorted with side='right' - 1)
    indices = np.searchsorted(cam_ts, physio_ts_ns, side="right") - 1

    # Build the output DataFrame
    n_physio = len(physio_ts_ns)
    n_cam_cols = len(cam_data.columns)
    out = pd.DataFrame(index=range(n_physio), columns=cam_data.columns, dtype=object)

    # Fill with camera data where index >= 0 (frame exists before physio ts)
    valid_mask = indices >= 0
    valid_indices = indices[valid_mask]

    # Bulk copy valid rows
    for col_idx, col_name in enumerate(cam_data.columns):
        col_values = cam_data[col_name].values
        out.loc[valid_mask, col_name] = col_values[valid_indices]

    # cam_frame_new: True on the physio row where a new camera frame first appears
    # A new frame "appears" at the first physio timestamp that maps to it
    cam_frame_new = np.zeros(n_physio, dtype=bool)
    prev_idx = -1
    for i, idx in enumerate(indices):
        if idx >= 0 and idx != prev_idx:
            cam_frame_new[i] = True
            prev_idx = idx
    out["cam_frame_new"] = cam_frame_new

    return out


# ── Single-run merge ─────────────────────────────────────────────────────────


def merge_run(
    sub: str,
    ses: str,
    task: str,
    run: int,
    run_idx: int,
    rep: int,
    *,
    data_root: Path | None = None,
    sync_correction: SyncCorrection | None = None,
) -> pd.DataFrame:
    """Merge physio + camera + targets for a single run into one 100Hz DataFrame.

    Parameters
    ----------
    sub, ses, task, run : str/int
        Run identifiers.
    run_idx : int
        Global run index (0-based) for cross-run ordering.
    rep : int
        Repetition number within the task (1, 2, 3, ...).
    data_root : Path, optional
        Data root directory (defaults to data/raw/).
    sync_correction : SyncCorrection, optional
        Pre-loaded sync correction. If None, loads from led_sync.json.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with one row per FSR sample (100Hz).
        Empty DataFrame if physio data is missing.
    """
    sdir = paths.session_dir(sub, ses, data_root=data_root)

    # Load physio
    physio_name = naming.build_filename(sub, ses, task, run, naming.SUFFIX_PHYSIO)
    physio_path = sdir / physio_name
    if not physio_path.exists():
        logger.warning("No physio file for %s/%s/%s/%d", sub, ses, task, run)
        return pd.DataFrame()
    physio = pd.read_csv(physio_path)

    if physio.empty:
        return pd.DataFrame()

    # Load sync correction if not provided
    if sync_correction is None:
        sync_correction = load_sync_correction(sub, ses, data_root=data_root)

    # Load and correct camera
    camera_name = naming.build_filename(sub, ses, task, run, naming.SUFFIX_CAMERA)
    camera_path = sdir / camera_name
    camera_df = pd.DataFrame()
    if camera_path.exists():
        camera_df = pd.read_csv(camera_path)
        if not camera_df.empty and not sync_correction.is_identity:
            camera_df = apply_sync_correction(camera_df, sync_correction)
        camera_df = camera_df.sort_values(schema.CAMERA_TIMESTAMP).reset_index(drop=True)

    # Load targets
    targets_name = naming.build_filename(sub, ses, task, run, naming.SUFFIX_TARGETS)
    targets_path = sdir / targets_name
    targets_df = pd.DataFrame()
    if targets_path.exists():
        targets_df = pd.read_csv(targets_path)

    # Start with physio as the base (100Hz grid)
    merged = physio.copy()
    n = len(merged)

    # Add run_idx and rep columns
    merged["run_idx"] = run_idx
    merged["rep"] = rep

    # Add trial_type: task name during RECORD, None otherwise
    if schema.PHYSIO_PHASE in merged.columns:
        merged["trial_type"] = np.where(
            merged[schema.PHYSIO_PHASE] == "RECORD", task, None
        )
    else:
        merged["trial_type"] = task

    # Forward-fill camera data onto physio timestamps
    if not camera_df.empty:
        physio_ts = merged[schema.PHYSIO_TIMESTAMP].values.astype(np.int64)
        cam_aligned = forward_fill_camera(physio_ts, camera_df)
        if not cam_aligned.empty:
            # Merge camera columns into the result
            for col in cam_aligned.columns:
                merged[col] = cam_aligned[col].values

    # Join targets by exact timestamp match
    if not targets_df.empty:
        merged = _merge_targets(merged, targets_df)

    return merged


def _merge_targets(physio_df: pd.DataFrame, targets_df: pd.DataFrame) -> pd.DataFrame:
    """Join targets to physio by exact timestamp match.

    Targets share the same t_monotonic_ns as physio (both at 100Hz).
    Uses a left merge on the timestamp column.
    """
    ts_col = schema.PHYSIO_TIMESTAMP
    target_ts_col = schema.TARGETS_TIMESTAMP

    # Rename target timestamp to match physio for the merge
    targets_renamed = targets_df.rename(columns={target_ts_col: ts_col})

    # Left merge on timestamp
    merged = physio_df.merge(
        targets_renamed,
        on=ts_col,
        how="left",
        suffixes=("", "_targets"),
    )

    return merged


# ── Full-session merge ───────────────────────────────────────────────────────


def merge_session(
    sub: str,
    ses: str,
    *,
    data_root: Path | None = None,
    sync_correction: SyncCorrection | None = None,
) -> pd.DataFrame:
    """Merge all runs in a session into a single concatenated DataFrame.

    Each run gets a unique ``run_idx`` (0-based, in protocol order) and
    ``rep`` number. The result has all runs stacked vertically, with
    ``sample_idx`` reset within each run.

    Parameters
    ----------
    sub, ses : str
        Subject and session labels.
    data_root : Path, optional
        Data root (defaults to data/raw/).
    sync_correction : SyncCorrection, optional
        Pre-loaded sync correction. If None, loads from led_sync.json.

    Returns
    -------
    pd.DataFrame
        Concatenated merged data for all runs. Empty if no runs found.
    """
    sdir = paths.session_dir(sub, ses, data_root=data_root)
    if not sdir.exists():
        return pd.DataFrame()

    # Load sync correction once for the whole session
    if sync_correction is None:
        sync_correction = load_sync_correction(sub, ses, data_root=data_root)

    # Find all runs — deduplicate by (task, run) since each run has 3 files
    run_key_set: set[tuple[str, int]] = set()
    for path in sorted(sdir.iterdir()):
        parsed = naming.parse_filename(path)
        if parsed is None:
            continue
        run_key_set.add((parsed.task, parsed.run))

    if not run_key_set:
        return pd.DataFrame()

    run_keys = sorted(run_key_set)

    # Build rep mapping: for each task, rep = 1, 2, 3, ... in order of run number
    from collections import defaultdict
    task_run_list: dict[str, list[int]] = defaultdict(list)
    for task, run_num in run_keys:
        task_run_list[task].append(run_num)
    rep_map: dict[tuple[str, int], int] = {}
    for task, runs in task_run_list.items():
        for rep_idx, run_num in enumerate(sorted(runs), start=1):
            rep_map[(task, run_num)] = rep_idx

    # Merge each run
    all_runs: list[pd.DataFrame] = []
    for run_idx, (task, run_num) in enumerate(run_keys):
        rep = rep_map.get((task, run_num), 1)
        merged = merge_run(
            sub, ses, task, run_num, run_idx, rep,
            data_root=data_root,
            sync_correction=sync_correction,
        )
        if not merged.empty:
            all_runs.append(merged)

    if not all_runs:
        return pd.DataFrame()

    result = pd.concat(all_runs, ignore_index=True)
    return result


# ── Metadata summary ─────────────────────────────────────────────────────────


def session_metadata_summary(
    sub: str,
    ses: str,
    merged_df: pd.DataFrame,
    *,
    sync_correction: SyncCorrection | None = None,
) -> dict[str, Any]:
    """Build a metadata dict describing the merged session record.

    This is written as the sidecar JSON alongside the merged CSV.
    """
    # Count runs and tasks
    runs = merged_df.groupby(["task", "run", "run_idx", "rep"]).size().reset_index()
    n_runs = len(runs)
    tasks = sorted(runs["task"].unique().tolist())

    # Count samples per phase
    phase_counts = {}
    if "phase" in merged_df.columns:
        phase_counts = merged_df["phase"].value_counts().to_dict()

    # Camera coverage
    cam_frame_new_count = int(merged_df.get("cam_frame_new", pd.Series(dtype=bool)).sum()) if "cam_frame_new" in merged_df.columns else 0

    # Column list
    columns = merged_df.columns.tolist()

    summary: dict[str, Any] = {
        "sub": sub,
        "ses": ses,
        "n_runs": n_runs,
        "n_tasks": len(tasks),
        "tasks": tasks,
        "n_samples": len(merged_df),
        "n_columns": len(columns),
        "columns": columns,
        "phase_counts": {str(k): int(v) for k, v in phase_counts.items()},
        "cam_frame_new_count": cam_frame_new_count,
        "cam_coverage_pct": round(100 * cam_frame_new_count / len(merged_df), 2) if len(merged_df) > 0 else 0,
        "sampling_frequency_hz": 100.0,
        "duration_s": round(len(merged_df) / 100.0, 2),
        "sync_method": sync_correction.method if sync_correction else "none",
        "sync_passed": sync_correction.passed if sync_correction else False,
        "sync_a": sync_correction.a if sync_correction else 1.0,
        "sync_b": sync_correction.b if sync_correction else 0.0,
    }

    return summary
