"""BIDS data loader — joins physio + camera + targets by timestamp.

The loader is the heart of data access. Every app and script uses it:

    from core.loader import load_run, load_session, load_regression_data

    # Load a single run (3 modalities joined by timestamp)
    run = load_run(sub="P01", ses="S01", task="powerGrip", run=1)
    print(run.physio.shape)    # (N, 4) FSR samples
    print(run.camera.shape)   # (M, 67) MediaPipe landmarks
    print(run.targets.shape)   # (N, 15) joint angles

    # Load all runs in a session
    session = load_session(sub="P01", ses="S01")
    for run in session.runs:
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import naming, paths, schema
from .metadata import load_led_sync

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class BIDSRun:
    """A single run with 3 modalities loaded."""

    sub: str
    ses: str
    task: str
    run: int
    physio: pd.DataFrame = field(default_factory=pd.DataFrame)
    camera: pd.DataFrame = field(default_factory=pd.DataFrame)
    targets: pd.DataFrame = field(default_factory=pd.DataFrame)
    physio_path: Path | None = None
    camera_path: Path | None = None
    targets_path: Path | None = None

    @property
    def is_baseline(self) -> bool:
        return self.run == 0

    @property
    def phase(self) -> str:
        return schema.task_phase(self.task)

    @property
    def n_samples(self) -> int:
        return len(self.physio)

    @property
    def n_camera_frames(self) -> int:
        return len(self.camera)

    @property
    def sensor_columns(self) -> list[str]:
        """Return FSR column names from the physio DataFrame."""
        return [c for c in self.physio.columns if c.startswith("fsr")]

    @property
    def n_sensors(self) -> int:
        return len(self.sensor_columns)


@dataclass
class BIDSSession:
    """All runs in a session."""

    sub: str
    ses: str
    runs: list[BIDSRun] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    def runs_for_task(self, task: str) -> list[BIDSRun]:
        return [r for r in self.runs if r.task == task]

    def tasks(self) -> list[str]:
        return sorted(set(r.task for r in self.runs))

    def runs_in_phase(self, phase: str) -> list[BIDSRun]:
        return [r for r in self.runs if r.phase == phase]


# ── Single-run loader ─────────────────────────────────────────────────────────


def load_run(
    sub: str,
    ses: str,
    task: str,
    run: int,
    *,
    data_root: Path | None = None,
    require_manifest: bool = False,
    _led_sync=None,
) -> BIDSRun:
    """Load a single run (physio + camera + targets) from BIDS data.

    Missing files are silently skipped (the corresponding DataFrame is empty).

    If ``require_manifest`` is True, the run is only loaded if a valid
    ``manifest.json`` sidecar exists with ``complete == True``. This gates
    out partial runs left by a mid-session crash. When False (default),
    runs are loaded regardless of manifest presence (backward compatible).

    If a ``led_sync.json`` sidecar exists for the session with ``passed == True``
    and linear correction coefficients ``a`` and ``b``, the camera timestamps
    are corrected as ``t_cam_corrected = a * t_cam + b``.

    Parameters
    ----------
    require_manifest : bool
        If True, skip runs without a valid complete manifest.
    _led_sync : LedSyncMetadata | None
        Pre-loaded LED sync metadata.  When ``None`` (the default, i.e. when
        this function is called directly), the sidecar is read from disk.
        ``load_session()`` passes a cached value to avoid reading the same
        JSON file once per run (N+1 reads).
    """
    sdir = paths.session_dir(sub, ses, data_root=data_root)
    result = BIDSRun(sub=sub, ses=ses, task=task, run=run)

    # Manifest gate — skip partial runs when required.
    if require_manifest:
        from apps.collection.bids_writer import manifest_path as _manifest_path
        mpath = _manifest_path(sub, ses, task, run, data_root=data_root)
        if not mpath.exists():
            logger.warning(
                "Skipping run %s/%s/%s/%d — no manifest (partial run)",
                sub, ses, task, run,
            )
            return result
        try:
            import json
            with open(mpath, encoding="utf-8") as f:
                manifest = json.load(f)
            if not manifest.get("complete", False):
                logger.warning(
                    "Skipping run %s/%s/%s/%d — manifest complete=False",
                    sub, ses, task, run,
                )
                return result
        except Exception as exc:
            logger.warning(
                "Skipping run %s/%s/%s/%d — manifest read error: %s",
                sub, ses, task, run, exc,
            )
            return result

    # Check for LED sync correction — use the caller-supplied cache when available
    led_sync = _led_sync if _led_sync is not None else load_led_sync(sub, ses, data_root=data_root)
    apply_correction = (
        led_sync.passed is True
        and led_sync.a is not None
        and led_sync.b is not None
    )

    # Physio
    physio_name = naming.build_filename(sub, ses, task, run, naming.SUFFIX_PHYSIO)
    physio_path = sdir / physio_name
    if physio_path.exists():
        result.physio = pd.read_csv(physio_path)
        result.physio_path = physio_path

    # Camera
    camera_name = naming.build_filename(sub, ses, task, run, naming.SUFFIX_CAMERA)
    camera_path = sdir / camera_name
    if camera_path.exists():
        result.camera = pd.read_csv(camera_path)
        result.camera_path = camera_path
        # Apply LED sync correction to camera timestamps
        if apply_correction and schema.CAMERA_TIMESTAMP in result.camera.columns:
            result.camera[schema.CAMERA_TIMESTAMP] = (
                led_sync.a * result.camera[schema.CAMERA_TIMESTAMP] + led_sync.b
            )

    # Targets
    targets_name = naming.build_filename(sub, ses, task, run, naming.SUFFIX_TARGETS)
    targets_path = sdir / targets_name
    if targets_path.exists():
        result.targets = pd.read_csv(targets_path)
        result.targets_path = targets_path

    return result


# ── Session loader ────────────────────────────────────────────────────────────


def load_session(
    sub: str,
    ses: str,
    *,
    data_root: Path | None = None,
    require_manifest: bool = False,
) -> BIDSSession:
    """Load all runs in a session.

    When ``require_manifest`` is True, partial runs (no valid manifest) are
    skipped — used by ML consumers to avoid training on crash-incomplete data.
    """
    sdir = paths.session_dir(sub, ses, data_root=data_root)
    session = BIDSSession(sub=sub, ses=ses)

    if not sdir.exists():
        return session

    # Load LED sync once for the whole session (avoids N+1 file reads per run)
    cached_led_sync = load_led_sync(sub, ses, data_root=data_root)

    # Find all run files and group by (task, run)
    run_keys: set[tuple[str, int]] = set()
    for path in sorted(sdir.iterdir()):
        parsed = naming.parse_filename(path)
        if parsed is None:
            continue
        run_keys.add((parsed.task, parsed.run))

    for task, run_num in sorted(run_keys):
        run = load_run(
            sub, ses, task, run_num,
            data_root=data_root, require_manifest=require_manifest,
            _led_sync=cached_led_sync,
        )
        session.runs.append(run)

    return session


# ── Multi-session loaders (for ML consumers) ──────────────────────────────────


def load_all_sessions(
    data_root: Path | None = None,
    *,
    require_manifest: bool = True,
) -> list[BIDSSession]:
    """Load all sessions from all subjects in a data root.

    By default ``require_manifest=True`` so ML consumers skip partial runs
    left by mid-session crashes. Pass ``require_manifest=False`` for
    debugging or legacy data without manifests.
    """
    root = data_root or paths.SAMPLE_DIR
    subjects = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.startswith("sub-"):
            subjects.append(d.name[4:])

    sessions = []
    for sub in subjects:
        ses_labels = []
        subj_dir = root / f"sub-{sub}"
        for d in sorted(subj_dir.iterdir()):
            if d.is_dir() and d.name.startswith("ses-"):
                ses_labels.append(d.name[4:])
        for ses in ses_labels:
            session = load_session(
                sub, ses, data_root=root, require_manifest=require_manifest
            )
            sessions.append(session)
    return sessions


def load_regression_data(
    data_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load all runs as regression data (FSR → 15 joint angles).

    Uses **protocol-phase-based labeling**: only RECORD-phase samples are
    included.  PREP and REST phases are excluded.

    Freeform is excluded (no structured targets).  MVC is included (provides
    maximum force range data useful for regression).

    Returns
    -------
    X : (N_samples, n_sensors) float32
        FSR signals, RECORD phase only, time-aligned with targets.
    Y : (N_samples, 15) float32
        Joint angle targets.
    meta : DataFrame
        One row per sample with sub, ses, task, run, phase columns.
    """
    sessions = load_all_sessions(data_root)
    X_list, Y_list, meta_list = [], [], []

    for session in sessions:
        for run in session.runs:
            if run.physio.empty or run.targets.empty:
                continue
            # Exclude freeform (no structured targets); keep MVC (max force range)
            if run.task in schema.FREEFORM_TASKS:
                continue

            # Filter to RECORD phase only
            physio = run.physio
            if "phase" in physio.columns:
                physio = physio[physio["phase"] == "RECORD"]
            if physio.empty:
                continue

            # Join physio and targets by timestamp
            merged = _join_by_timestamp(
                physio, run.targets,
                schema.PHYSIO_TIMESTAMP, schema.TARGETS_TIMESTAMP,
            )
            if merged.empty:
                continue

            sensor_cols = [c for c in run.physio.columns if c.startswith("fsr")]
            target_cols = schema.TARGET_COLUMNS

            available_targets = [c for c in target_cols if c in merged.columns]
            if not available_targets or not sensor_cols:
                continue

            X_list.append(merged[sensor_cols].values.astype(np.float32))
            Y_list.append(merged[available_targets].values.astype(np.float32))

            n = len(merged)
            meta_list.append(pd.DataFrame({
                "sub": [run.sub] * n,
                "ses": [run.ses] * n,
                "task": [run.task] * n,
                "run": [run.run] * n,
                "phase": ["RECORD"] * n,
            }))

    if not X_list:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32), pd.DataFrame()

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    meta = pd.concat(meta_list, ignore_index=True)
    return X, Y, meta


def load_classification_data(
    data_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load all runs as classification data (FSR → discrete gesture labels).

    Uses **protocol-phase-based labeling**: only RECORD-phase samples are
    included, labeled with the task name.  PREP and REST phases are excluded.
    No 'rest' class — the collection protocol's phase labels are the ground
    truth for when the subject was performing the gesture.

    MVC (baseline calibration) and freeform (unstructured) are excluded —
    they are not gesture classes.

    Returns
    -------
    X : (N_samples, n_sensors) float32
        FSR signals, RECORD phase only.
    y : (N_samples,) str
        Gesture labels (task name).
    meta : DataFrame
        One row per sample with sub, ses, task, run, phase columns.
    """
    sessions = load_all_sessions(data_root)
    X_list, y_list, meta_list = [], [], []

    for session in sessions:
        for run in session.runs:
            if run.physio.empty or run.targets.empty:
                continue
            # Exclude MVC (baseline) and freeform (unstructured)
            if run.task in schema.FREEFORM_TASKS or run.task == schema.BASELINE_TASK:
                continue

            # Filter to RECORD phase only (protocol-phase-based labeling)
            physio = run.physio
            if "phase" in physio.columns:
                physio = physio[physio["phase"] == "RECORD"]
            if physio.empty:
                continue

            # Join physio and targets by timestamp
            merged = _join_by_timestamp(
                physio, run.targets,
                schema.PHYSIO_TIMESTAMP, schema.TARGETS_TIMESTAMP,
            )
            if merged.empty:
                continue

            sensor_cols = [c for c in run.physio.columns if c.startswith("fsr")]
            if not sensor_cols:
                continue

            n = len(merged)
            X_list.append(merged[sensor_cols].values.astype(np.float32))
            y_list.append(np.array([run.task] * n, dtype=object))
            meta_list.append(pd.DataFrame({
                "sub": [run.sub] * n,
                "ses": [run.ses] * n,
                "task": [run.task] * n,
                "run": [run.run] * n,
                "phase": ["RECORD"] * n,
            }))

    if not X_list:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=object), pd.DataFrame()

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    meta = pd.concat(meta_list, ignore_index=True)
    return X, y, meta


# ── Helpers ───────────────────────────────────────────────────────────────────


def _join_by_timestamp(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_ts: str,
    right_ts: str,
) -> pd.DataFrame:
    """Join two DataFrames by nearest timestamp.

    Uses merge_asof to align by nearest timestamp (within 50ms tolerance).
    """
    if left.empty or right.empty:
        return pd.DataFrame()

    left_sorted = left.sort_values(left_ts).copy()
    right_sorted = right.sort_values(right_ts).copy()

    # Convert to common unit (nanoseconds)
    left_sorted["_join_ts"] = left_sorted[left_ts].astype("int64")
    right_sorted["_join_ts"] = right_sorted[right_ts].astype("int64")

    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        on="_join_ts",
        direction="nearest",
        tolerance=50_000_000,  # 50ms in nanoseconds
        suffixes=("", "_targets"),
    )
    merged = merged.drop(columns=["_join_ts"])
    return merged
