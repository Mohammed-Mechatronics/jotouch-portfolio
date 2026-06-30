"""BIDS writer — writes physio, camera, and targets CSVs in BIDS format.

This is the replacement for the legacy ``DataLogger``. Instead of writing a
single flat CSV per session, it writes three BIDS-compliant CSVs per run:

  sub-P01_ses-S01_task-powerGrip_run-01_physio.csv   (FSR signals)
  sub-P01_ses-S01_task-powerGrip_run-01_camera.csv   (MediaPipe landmarks)
  sub-P01_ses-S01_task-powerGrip_run-01_targets.csv  (15 joint angles)

All three files share the same monotonic timestamp (``t_monotonic_ns``)
so they can be joined by ``core.loader._join_by_timestamp``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core import naming, paths, schema

logger = logging.getLogger(__name__)


def manifest_path(
    sub: str, ses: str, task: str, run: int, *, data_root: Path | None = None
) -> Path:
    """Return the path to a run's manifest.json sidecar."""
    sdir = paths.session_dir(sub, ses, data_root=data_root or paths.RAW_DIR)
    return sdir / f"sub-{sub}_ses-{ses}_task-{task}_run-{run:02d}_manifest.json"


def _sha1_of_file(path: Path) -> str:
    """Return the SHA1 hex digest of a file, or '' if missing/unreadable."""
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def quarantine_partial_run(
    sub: str, ses: str, task: str, run: int, *, data_root: Path | None = None
) -> Path:
    """Move a partial/aborted run's files to ``_partial/`` and free the run number.

    Moves the 3 CSVs + manifest (if it exists) + sentinel (if it exists) into
    ``sub-.../ses-.../_partial/task-{task}_run-{NN}_attempt{K}/``, where ``K``
    auto-increments so repeated failures of the same run number don't collide.

    After quarantine, the run number is free: a new ``BIDSRunWriter`` with the
    same ``run`` can ``open()`` (mode ``"x"``) without ``FileExistsError``.

    Returns the path to the attempt subdirectory.
    """
    import shutil

    sdir = paths.session_dir(sub, ses, data_root=data_root or paths.RAW_DIR)
    partial_root = sdir / "_partial"

    # Determine the next attempt number for this (task, run) pair.
    prefix = f"task-{task}_run-{run:02d}_attempt"
    attempt = 1
    if partial_root.exists():
        existing = [d.name for d in partial_root.iterdir() if d.is_dir()]
        for name in existing:
            if name.startswith(prefix):
                try:
                    n = int(name[len(prefix):])
                    attempt = max(attempt, n + 1)
                except ValueError:
                    pass

    attempt_dir = partial_root / f"{prefix}{attempt}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    # Move all files matching this run's naming pattern.
    stem = f"sub-{sub}_ses-{ses}_task-{task}_run-{run:02d}"
    moved_any = False
    for path in sorted(sdir.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith(stem) and path.suffix in (".csv", ".json", ".complete"):
            shutil.move(str(path), str(attempt_dir / path.name))
            moved_any = True

    if not moved_any:
        logger.warning(
            "quarantine_partial_run: no files found for %s run-%02d — already moved?",
            task, run,
        )

    logger.info(
        "Quarantined partial run %s/%s/%s/%02d → %s (attempt %d)",
        sub, ses, task, run, attempt_dir, attempt,
    )
    return attempt_dir


def sweep_orphan_partials(
    sub: str, ses: str, *, data_root: Path | None = None
) -> int:
    """Quarantine orphan/partial run files left by a crashed process.

    Scans the session directory for BIDS run files whose manifest is either
    missing (process killed before ``close()`` ran) or has ``complete=false``
    (run was aborted but not yet quarantined). Each such orphan is moved to
    ``_partial/`` via :func:`quarantine_partial_run`, freeing the run number
    for reuse.

    Completed runs (manifest with ``complete=true`` + ``.complete`` sentinel)
    are preserved and never swept.

    Called at the start of every session (before any runs) so the session
    directory is clean and ``next_run_number`` doesn't count orphan files.

    Parameters
    ----------
    sub, ses : str
        Subject and session labels.
    data_root : Path | None
        Data root (defaults to ``core.paths.RAW_DIR``).

    Returns
    -------
    int
        Number of orphan runs quarantined.
    """
    sdir = paths.session_dir(sub, ses, data_root=data_root or paths.RAW_DIR)
    if not sdir.exists():
        return 0

    # Collect all (task, run) pairs that have run files in the session dir.
    seen_pairs: dict[tuple[str, int], bool] = {}  # (task, run) -> is_complete
    for path in sorted(sdir.iterdir()):
        if not path.is_file():
            continue
        parsed = naming.parse_filename(path)
        if parsed is None:
            continue
        key = (parsed.task, parsed.run)
        if key not in seen_pairs:
            # Check the manifest for this run
            mpath = manifest_path(sub, ses, parsed.task, parsed.run, data_root=data_root)
            is_complete = False
            if mpath.exists():
                try:
                    import json as _json
                    with open(mpath, encoding="utf-8") as f:
                        m = _json.load(f)
                    is_complete = m.get("complete", False)
                except Exception:
                    pass
            seen_pairs[key] = is_complete

    # Quarantine any (task, run) that is not complete.
    swept = 0
    for (task, run), is_complete in seen_pairs.items():
        if not is_complete:
            logger.info(
                "sweep_orphan_partials: quarantining orphan %s run-%02d (incomplete or no manifest)",
                task, run,
            )
            quarantine_partial_run(sub, ses, task, run, data_root=data_root)
            swept += 1

    if swept > 0:
        logger.info("sweep_orphan_partials: quarantined %d orphan run(s) for %s/%s", swept, sub, ses)

    return swept


def _session_is_protocol_complete(sdir: Path) -> bool:
    """Return True if the session has ALL required protocol runs completed.

    A session is "protocol-complete" when every task in the session's
    CHOSEN protocol (see :func:`apps.collection.protocol.build_protocol`,
    using the ``n_reps`` / ``include_freeform`` persisted in physio.json)
    has at least the expected number of completed runs.

    A run is "completed" when its manifest has ``complete=true``.  The
    ``.complete`` sentinel is NOT required — the manifest is the
    authoritative source, and a crash between writing the manifest and
    touching the sentinel must not cause a completed run to be missed.

    If physio.json is missing or corrupted (the chosen protocol config
    can't be read), the session is considered complete if it has ANY
    completed runs — data preservation takes priority over
    protocol-completeness when the expected protocol is unknown.
    """
    from collections import defaultdict
    from apps.collection.protocol import build_protocol

    # Count completed runs per task in this session directory.
    # A run is complete if its manifest says complete=true (the manifest is
    # authoritative; the .complete sentinel is a convenience for filesystem
    # scanning and may be missing if a crash occurred between the manifest
    # write and the sentinel touch).
    actual: dict[str, int] = defaultdict(int)
    has_any_completed = False
    for mf in sdir.glob("*_manifest.json"):
        try:
            with open(mf, encoding="utf-8") as f:
                m = json.load(f)
            if not m.get("complete"):
                continue
            task = m.get("task", "")
            actual[task] += 1
            has_any_completed = True
        except Exception:
            continue

    # Read the session's chosen protocol config from physio.json.
    n_reps = 3
    include_freeform = True
    config_read_ok = False
    _physio_candidates = sorted(sdir.glob("*_physio.json"))
    physio_json = _physio_candidates[0] if _physio_candidates else None
    if physio_json is not None and physio_json.exists():
        try:
            with open(physio_json, encoding="utf-8") as f:
                meta = json.load(f)
            n_reps = int(meta.get("NReps", 3))
            include_freeform = bool(meta.get("IncludeFreeform", True))
            config_read_ok = True
        except Exception:
            pass  # config unreadable — handle below

    # If the protocol config can't be read (physio.json missing/corrupted),
    # preserve the session if it has ANY completed runs.  Falling back to
    # the 76-run default would wrongly quarantine a session that completed
    # its chosen (smaller) protocol.
    if not config_read_ok:
        return has_any_completed

    # Build the expected per-task run counts from the chosen protocol.
    protocol = build_protocol(n_reps=n_reps, include_freeform=include_freeform)
    expected: dict[str, int] = defaultdict(int)
    for spec in protocol:
        expected[spec.task] += 1

    # Check every required task has at least the expected number of runs.
    for task in expected:
        if actual.get(task, 0) < expected[task]:
            return False
    return True


def _remove_from_tsv(tsv_path: Path, sub: str, ses: str | None = None) -> int:
    """Remove rows matching sub (and optionally ses) from a TSV file.

    Returns the number of rows removed.  Preserves the header line.
    """
    if not tsv_path.exists():
        return 0
    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return 0
    header = lines[0]
    body = lines[1:]
    sub_id = f"sub-{sub}"
    ses_id = f"ses-{ses}" if ses else None
    kept: list[str] = []
    removed = 0
    for line in body:
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0] == sub_id:
            if ses_id is None:
                # participants.tsv: match by participant_id only
                removed += 1
                continue
            elif parts[1] == ses_id:
                # sessions.tsv: match by participant_id + session_id
                removed += 1
                continue
        kept.append(line)
    if removed > 0:
        with open(tsv_path, "w", encoding="utf-8", newline="") as f:
            f.write(header)
            f.writelines(kept)
    return removed


def sweep_incomplete_sessions(*, data_root: Path | None = None) -> int:
    """Quarantine sessions that have not completed their chosen protocol.

    Scans all ``sub-*/ses-*`` directories in the data root.  If a session
    directory does NOT have its chosen protocol completed (every task in
    :func:`apps.collection.protocol.build_protocol` with the expected number
    of completed runs, using the session's own ``n_reps`` /
    ``include_freeform`` config), the entire session directory is moved to
    ``data_root/_incomplete/sub-{sub}/ses-{ses}/`` and the corresponding
    rows are removed from ``participants.tsv`` and ``sessions.tsv``.

    This frees the subject and session labels for reuse —
    ``next_subject_label()`` and ``next_session_label()`` will no longer
    count the quarantined directories.

    Called automatically before ``/api/next_subject`` and
    ``/api/next_session`` respond, so the UI always suggests the correct
    next label even if a previous session was started but never completed
    the full protocol.

    Parameters
    ----------
    data_root : Path | None
        Data root (defaults to ``core.paths.RAW_DIR``).

    Returns
    -------
    int
        Number of incomplete sessions quarantined.
    """
    import shutil

    root = data_root if data_root is not None else paths.RAW_DIR
    if not root.exists():
        return 0

    _sub_dir_re = __import__("re").compile(r"^sub-(.+)$")
    _ses_dir_re = __import__("re").compile(r"^ses-(.+)$")

    incomplete_root = root / "_incomplete"
    swept = 0

    for sub_dir in sorted(root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        m = _sub_dir_re.match(sub_dir.name)
        if m is None:
            continue
        sub = m.group(1)

        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue
            ses_m = _ses_dir_re.match(ses_dir.name)
            if ses_m is None:
                continue
            ses = ses_m.group(1)

            if _session_is_protocol_complete(ses_dir):
                continue  # has full protocol data — leave it alone

            # Quarantine this session
            dest = incomplete_root / sub_dir.name / ses_dir.name
            dest.mkdir(parents=True, exist_ok=True)

            # Move all files (not subdirectories like _partial) into dest
            for item in sorted(ses_dir.iterdir()):
                if item.is_file():
                    shutil.move(str(item), str(dest / item.name))
                elif item.is_dir() and item.name == "_partial":
                    # Move the _partial directory too — merge if dest already
                    # has a _partial/ from a previous sweep attempt.
                    dest_partial = dest / item.name
                    if dest_partial.exists():
                        # Merge: move each item inside _partial/ into dest/_partial/
                        for inner in sorted(item.iterdir()):
                            shutil.move(str(inner), str(dest_partial / inner.name))
                        item.rmdir()
                    else:
                        shutil.move(str(item), str(dest / item.name))

            # Remove the now-empty session directory (use rmtree in case of leftovers)
            try:
                shutil.rmtree(ses_dir)
            except OSError:
                pass  # not removable — leave it

            # Remove from TSVs
            _remove_from_tsv(root / "sessions.tsv", sub, ses)

            # If the subject has no more session dirs, remove from participants.tsv
            remaining_sessions = [
                d for d in sub_dir.glob("ses-*") if d.is_dir()
            ]
            if not remaining_sessions:
                _remove_from_tsv(root / "participants.tsv", sub)
                # Remove the empty subject directory
                try:
                    sub_dir.rmdir()
                except OSError:
                    pass

            swept += 1
            logger.info(
                "sweep_incomplete_sessions: quarantined %s/%s (0 completed runs) → %s",
                sub, ses, dest,
            )

    if swept > 0:
        logger.info("sweep_incomplete_sessions: quarantined %d incomplete session(s)", swept)

    return swept


@dataclass
class BIDSRunWriter:
    """Write a single BIDS run (3 CSVs) to disk.

    Usage:
        writer = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1)
        writer.open()
        for sample in stream:
            writer.write_physio(t_ns, fsr_values, phase="RECORD")
        writer.write_camera(t_ns, landmarks_dict)
        writer.write_targets(t_ns, joint_angles)
        writer.close()

    If ``run`` is ``None``, the next free run number for ``(sub, ses, task)``
    is computed automatically via :func:`core.naming.next_run_number` so
    re-runs never silently overwrite existing data. When ``run`` is given
    explicitly, :meth:`open` uses mode ``"x"`` and raises ``FileExistsError``
    if the run already exists — callers must explicitly delete or rename the
    old run first.
    """

    sub: str
    ses: str
    task: str
    run: Optional[int] = None
    n_sensors: int = 4
    data_root: Path | None = None  # defaults to data/raw/

    _physio_file: object = field(default=None, init=False, repr=False)
    _physio_writer: object = field(default=None, init=False, repr=False)
    _camera_file: object = field(default=None, init=False, repr=False)
    _camera_writer: object = field(default=None, init=False, repr=False)
    _targets_file: object = field(default=None, init=False, repr=False)
    _targets_writer: object = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _physio_count: int = field(default=0, init=False, repr=False)
    _camera_count: int = field(default=0, init=False, repr=False)
    _targets_count: int = field(default=0, init=False, repr=False)
    _bad_physio_count: int = field(default=0, init=False, repr=False)
    _bad_camera_count: int = field(default=0, init=False, repr=False)
    _bad_targets_count: int = field(default=0, init=False, repr=False)
    _last_physio_ts: int = field(default=-1, init=False, repr=False)
    _last_camera_ts: int = field(default=-1, init=False, repr=False)
    _last_targets_ts: int = field(default=-1, init=False, repr=False)
    # WS-5: Camera quality tracking for manifest enrichment
    _first_cam_ts: int = field(default=-1, init=False, repr=False)
    _valid_camera_count: int = field(default=0, init=False, repr=False)
    _led_cam_sum: int = field(default=0, init=False, repr=False)
    _led_cam_visible_count: int = field(default=0, init=False, repr=False)
    _started_at: str = field(default="", init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _aborted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # Auto-increment run number when not explicitly provided.
        if self.run is None:
            self.run = naming.next_run_number(
                self.sub, self.ses, self.task, data_root=self.data_root
            )

    @property
    def session_dir(self) -> Path:
        return paths.session_dir(self.sub, self.ses, data_root=self.data_root or paths.RAW_DIR)

    @property
    def physio_path(self) -> Path:
        return self.session_dir / naming.build_filename(
            self.sub, self.ses, self.task, self.run, naming.SUFFIX_PHYSIO
        )

    @property
    def camera_path(self) -> Path:
        return self.session_dir / naming.build_filename(
            self.sub, self.ses, self.task, self.run, naming.SUFFIX_CAMERA
        )

    @property
    def targets_path(self) -> Path:
        return self.session_dir / naming.build_filename(
            self.sub, self.ses, self.task, self.run, naming.SUFFIX_TARGETS
        )

    @property
    def sentinel_path(self) -> Path:
        return self.session_dir / (
            f"sub-{self.sub}_ses-{self.ses}_task-{self.task}_run-{self.run:02d}.complete"
        )

    def open(self) -> None:
        """Open all three CSV files and write headers.

        Raises ``FileExistsError`` if any of the three CSVs already exists
        (mode ``"x"``). This prevents silent data destruction on re-runs.
        """
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Physio — mode "x" fails if the file already exists.
        self._physio_file = open(self.physio_path, "x", newline="", encoding="utf-8")
        self._physio_writer = csv.writer(self._physio_file)
        header = schema.physio_all_columns(self.n_sensors)
        self._physio_writer.writerow(header)

        # Camera
        self._camera_file = open(self.camera_path, "x", newline="", encoding="utf-8")
        self._camera_writer = csv.writer(self._camera_file)
        self._camera_writer.writerow(schema.CAMERA_ALL_COLUMNS)

        # Targets
        self._targets_file = open(self.targets_path, "x", newline="", encoding="utf-8")
        self._targets_writer = csv.writer(self._targets_file)
        self._targets_writer.writerow(schema.TARGETS_ALL_COLUMNS)

    def write_physio(
        self,
        t_monotonic_ns: int,
        sample_idx: int,
        fsr_values: list[int] | list[float],
        *,
        phase: str = "RECORD",
        cue_event: str = "",
        led_fsr: int = 0,
    ) -> None:
        """Write one FSR sample row.

        Sets ``quality_flag = 1`` if any FSR value is outside the 0–1023 ADC
        range or the timestamp is non-monotonic; the bad value is still
        written (raw forensic value preserved). The per-run bad-sample count
        is recorded in the run manifest written by :meth:`close`.
        """
        with self._lock:
            if not self._physio_writer:
                return
            bad = 0
            # Validate timestamps are non-decreasing within a run
            if self._physio_count > 0 and t_monotonic_ns < self._last_physio_ts:
                logger.warning(
                    "Non-monotonic physio timestamp: %d < %d (task=%s run=%d)",
                    t_monotonic_ns, self._last_physio_ts, self.task, self.run,
                )
                bad = 1
            self._last_physio_ts = t_monotonic_ns
            # Validate FSR values are within ADC range
            for i, v in enumerate(fsr_values[: self.n_sensors]):
                if not (0 <= v <= 1023):
                    logger.warning(
                        "FSR value out of range: fsr%d=%d (task=%s run=%d)",
                        i, v, self.task, self.run,
                    )
                    bad = 1
            row = [
                t_monotonic_ns,
                sample_idx,
                phase,
                self.sub,
                self.ses,
                self.task,
                self.run,
            ]
            row.extend(fsr_values[: self.n_sensors])
            # Pad if fewer sensors than expected
            while len(row) < 7 + self.n_sensors:
                row.append(0)
            row.append(cue_event)
            row.append(led_fsr)
            row.append(bad)
            self._physio_writer.writerow(row)
            self._physio_file.flush()
            self._physio_count += 1
            if bad:
                self._bad_physio_count += 1

    def write_camera(
        self,
        cam_ts_ns: int,
        landmarks: list[float] | None,
        *,
        valid: bool = True,
        confidence: float = 0.0,
        handedness: str = "Right",
        led_cam: int = 0,
    ) -> None:
        """Write one camera frame row.

        ``landmarks`` should be a flat list of 63 floats (21 landmarks x 3 coords).
        If ``landmarks`` is None or ``valid`` is False, writes zeros.

        Sets ``quality_flag = 1`` if any landmark x/y is outside [0, 1] or the
        timestamp is non-monotonic. ``z`` is unconstrained relative depth and
        is not validated.
        """
        with self._lock:
            if not self._camera_writer:
                return
            bad = 0
            # Validate timestamps are non-decreasing within a run
            if self._camera_count > 0 and cam_ts_ns < self._last_camera_ts:
                logger.warning(
                    "Non-monotonic camera timestamp: %d < %d (task=%s run=%d)",
                    cam_ts_ns, self._last_camera_ts, self.task, self.run,
                )
                bad = 1
            self._last_camera_ts = cam_ts_ns
            row = [cam_ts_ns, int(valid), confidence, handedness]
            if landmarks and len(landmarks) == 63:
                # Validate MediaPipe normalized landmark coordinates.
                for i in range(21):
                    x, y = landmarks[i * 3], landmarks[i * 3 + 1]
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                        logger.warning(
                            "Landmark coordinate out of range: lm%d=(x=%.3f, y=%.3f) (task=%s run=%d)",
                            i, x, y, self.task, self.run,
                        )
                        bad = 1
                row.extend(landmarks)
            else:
                row.extend([0.0] * 63)
            row.append(led_cam)
            row.append(bad)
            self._camera_writer.writerow(row)
            self._camera_file.flush()
            self._camera_count += 1
            if bad:
                self._bad_camera_count += 1
            # WS-5: Accumulate camera quality stats for manifest
            if self._first_cam_ts < 0:
                self._first_cam_ts = cam_ts_ns
            if valid:
                self._valid_camera_count += 1
            self._led_cam_sum += int(led_cam)
            if int(led_cam) > 10:
                self._led_cam_visible_count += 1

    def write_targets(
        self,
        t_monotonic_ns: int,
        joint_angles: list[float] | dict[str, float],
    ) -> None:
        """Write one targets row (15 joint angles).

        Accepts either a list of 15 floats (in schema.TARGET_COLUMNS order) or
        a dict mapping column names to values. Missing dict keys default to 0.0.

        Sets ``quality_flag = 1`` if any joint angle is outside [0, 180] degrees
        or the timestamp is non-monotonic.
        """
        with self._lock:
            if not self._targets_writer:
                return
            bad = 0
            # Validate timestamps are non-decreasing within a run
            if self._targets_count > 0 and t_monotonic_ns < self._last_targets_ts:
                logger.warning(
                    "Non-monotonic targets timestamp: %d < %d (task=%s run=%d)",
                    t_monotonic_ns, self._last_targets_ts, self.task, self.run,
                )
                bad = 1
            self._last_targets_ts = t_monotonic_ns
            row = [t_monotonic_ns]
            if isinstance(joint_angles, dict):
                for col in schema.TARGET_COLUMNS:
                    row.append(joint_angles.get(col, 0.0))
            else:
                row.extend(joint_angles[:15])
                while len(row) < 16:
                    row.append(0.0)
            # Validate joint angles are within [0, 180] degrees
            for i, v in enumerate(row[1:], start=1):
                if not (0.0 <= v <= 180.0):
                    logger.warning(
                        "Joint angle out of range: target%d=%.2f (task=%s run=%d)",
                        i, v, self.task, self.run,
                    )
                    bad = 1
            row.append(bad)
            self._targets_writer.writerow(row)
            self._targets_file.flush()
            self._targets_count += 1
            if bad:
                self._bad_targets_count += 1

    def close(self, *, aborted: bool = False) -> dict:
        """Close all files, write the run manifest + completion sentinel, return counts.

        The manifest (``sub-..._run-NN_manifest.json``) and an empty
        ``.complete`` sentinel are written atomically. Consumers
        (``core.loader.load_run``) skip runs whose manifest is missing or whose
        ``complete`` flag is not ``True``, so a mid-run crash that prevents
        ``close()`` from running leaves a partial run that is never loaded.

        When ``aborted=True`` (operator stopped mid-run), the manifest is
        written with ``complete=false, aborted=true`` and **no** ``.complete``
        sentinel is created. This marks the run as partial so the ML gate
        skips it and the session loop can quarantine it to ``_partial/``.

        Idempotent: calling ``close()`` twice is safe and returns the same
        dict. The manifest/sentinel are written only once.
        """
        with self._lock:
            if self._closed:
                return self._result_dict()
            for f in (self._physio_file, self._camera_file, self._targets_file):
                if f:
                    f.close()
            self._physio_file = None
            self._camera_file = None
            self._targets_file = None
            self._physio_writer = None
            self._camera_writer = None
            self._targets_writer = None
            self._closed = True
            self._aborted = aborted
            self._write_manifest()
            return self._result_dict()

    def _result_dict(self) -> dict:
        return {
            "physio_rows": self._physio_count,
            "camera_rows": self._camera_count,
            "targets_rows": self._targets_count,
            "bad_physio_count": self._bad_physio_count,
            "bad_camera_count": self._bad_camera_count,
            "bad_targets_count": self._bad_targets_count,
            "physio_path": str(self.physio_path),
            "camera_path": str(self.camera_path),
            "targets_path": str(self.targets_path),
            "manifest_path": str(manifest_path(
                self.sub, self.ses, self.task, self.run, data_root=self.data_root
            )),
            "camera_quality": self._compute_camera_quality(),
        }

    def _compute_camera_quality(self) -> dict:
        """WS-5: Compute camera quality stats for the manifest.

        Returns a dict with:
          - achieved_fps: measured FPS from timestamp intervals
          - mean_led_cam: average LED brightness across all frames
          - led_visible_pct: percentage of frames where led_cam > 10
          - total_frames: total camera frames written
          - valid_pct: percentage of frames with valid hand detection
        """
        total = self._camera_count
        if total == 0:
            return {
                "achieved_fps": 0.0,
                "mean_led_cam": 0.0,
                "led_visible_pct": 0.0,
                "total_frames": 0,
                "valid_pct": 0.0,
            }
        # achieved_fps from first to last timestamp
        if self._first_cam_ts >= 0 and self._last_camera_ts > self._first_cam_ts:
            elapsed_s = (self._last_camera_ts - self._first_cam_ts) / 1e9
            achieved_fps = (total - 1) / elapsed_s if elapsed_s > 0 else 0.0
        else:
            achieved_fps = 0.0
        mean_led = self._led_cam_sum / total
        led_visible_pct = (self._led_cam_visible_count / total) * 100.0
        valid_pct = (self._valid_camera_count / total) * 100.0
        return {
            "achieved_fps": round(achieved_fps, 1),
            "mean_led_cam": round(mean_led, 1),
            "led_visible_pct": round(led_visible_pct, 1),
            "total_frames": total,
            "valid_pct": round(valid_pct, 1),
        }

    def _write_manifest(self) -> None:
        """Write manifest.json + .complete sentinel atomically.

        For aborted runs (``self._aborted == True``), the manifest gets
        ``complete=false, aborted=true`` and **no** ``.complete`` sentinel
        is written. This marks the run as partial so the ML gate skips it
        and the session loop can quarantine it to ``_partial/``.
        """
        finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest = {
            "sub": self.sub,
            "ses": self.ses,
            "task": self.task,
            "run": self.run,
            "physio_rows": self._physio_count,
            "camera_rows": self._camera_count,
            "targets_rows": self._targets_count,
            "bad_physio_count": self._bad_physio_count,
            "bad_camera_count": self._bad_camera_count,
            "bad_targets_count": self._bad_targets_count,
            "started_at": self._started_at,
            "finished_at": finished_at,
            "complete": not self._aborted,
            "aborted": self._aborted,
            "physio_sha1": _sha1_of_file(self.physio_path),
            "camera_sha1": _sha1_of_file(self.camera_path),
            "targets_sha1": _sha1_of_file(self.targets_path),
            "software_version": schema.SOFTWARE_VERSION,
            "bids_version": schema.BIDS_VERSION,
            "camera_quality": self._compute_camera_quality(),
        }
        mpath = manifest_path(self.sub, self.ses, self.task, self.run, data_root=self.data_root)
        # Atomic write: write to temp then rename.
        tmp = mpath.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, mpath)
        # Sentinel — only for completed (non-aborted) runs.
        if not self._aborted:
            self.sentinel_path.touch()

    @property
    def is_open(self) -> bool:
        return self._physio_file is not None


def write_session_metadata(
    sub: str,
    ses: str,
    *,
    sampling_frequency: float = 100.0,
    sensor_count: int = 4,
    camera_fps: float = 30,
    band_placement: str = "forearm_2_3",
    band_tension: str = "medium",
    n_reps: int = 3,
    include_freeform: bool = True,
    data_root: Path | None = None,
) -> None:
    """Write session-level BIDS metadata files (physio.json, channels.tsv).

    Called once at the start of a session, before any runs.

    ``n_reps`` and ``include_freeform`` are persisted to physio.json so the
    post-session sweep (:func:`sweep_incomplete_sessions`) can check
    completeness against the session's CHOSEN protocol, not the hardcoded
    76-run default.
    """
    import json

    sdir = paths.session_dir(sub, ses, data_root=data_root or paths.RAW_DIR)
    sdir.mkdir(parents=True, exist_ok=True)

    # physio.json
    physio_json = {
        "SamplingFrequency": sampling_frequency,
        "Manufacturer": "Interlink",
        "ManufacturersModelName": "FSR-400",
        "SoftwareVersion": schema.SOFTWARE_VERSION,
        "PlacementScheme": "Other",
        "PlacementDescription": f"FSR band at {band_placement}",
        "BandPlacement": band_placement,
        "BandTension": band_tension,
        "SensorCount": sensor_count,
        "CameraFPS": camera_fps,
        "CameraManufacturer": "n/a",
        "CameraModel": "n/a",
        "TaskList": schema.ALL_TASKS,
        "Instructions": "Follow on-screen cues. Phase 1: single DOF. Phase 2: grasps. Phase 3: freeform.",
        # Protocol config — used by sweep_incomplete_sessions to decide
        # whether the session completed its CHOSEN protocol.
        "NReps": n_reps,
        "IncludeFreeform": include_freeform,
    }
    physio_path = sdir / f"sub-{sub}_ses-{ses}_physio.json"
    with open(physio_path, "w", encoding="utf-8") as f:
        json.dump(physio_json, f, indent=2)

    # channels.tsv
    channels = [
        ("fsr0", "FSR-001", "FSR", "raw", "ulnar_side", "flexor_carpi_ulnaris"),
        ("fsr1", "FSR-002", "FSR", "raw", "ventral_mid", "flexor_digitorum"),
        ("fsr2", "FSR-003", "FSR", "raw", "radial_mid", "extensor_digitorum"),
        ("fsr3", "FSR-004", "FSR", "raw", "dorsal_side", "extensor_carpi_radialis"),
    ][:sensor_count]
    channels_path = sdir / f"sub-{sub}_ses-{ses}_channels.tsv"
    with open(channels_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["channel_name", "sensor_id", "type", "units", "placement_description", "target_muscle"])
        for ch in channels:
            writer.writerow(ch)


def write_dataset_description(
    data_root: Path | None = None,
    *,
    name: str = "JoTouch FMG Hand Gesture Dataset",
    bids_version: str | None = None,
    authors: list[str] | None = None,
    license: str = "n/a",
) -> None:
    """Write a BIDS dataset_description.json at the data root if absent."""
    import json

    root = data_root or paths.RAW_DIR
    root.mkdir(parents=True, exist_ok=True)
    desc_path = root / "dataset_description.json"
    if desc_path.exists():
        return
    description = {
        "Name": name,
        "BIDSVersion": bids_version or schema.BIDS_VERSION,
        "DatasetType": "raw",
        "Authors": authors or ["JoTouch Project"],
        "License": license,
        "Acknowledgements": "",
        "Funding": "",
        "EthicsApprovals": [],
        "ReferencesAndLinks": [],
        "DatasetDOI": "",
    }
    with open(desc_path, "w", encoding="utf-8") as f:
        json.dump(description, f, indent=2)


def update_mvc_calibration(
    sub: str,
    ses: str,
    *,
    data_root: Path | None = None,
) -> None:
    """Compute baseline and MVC ADC vectors from the MVC run and update physio.json.

    Reads the MVC run physio CSV, uses the PREP/REST phases as baseline and the
    RECORD phase as the maximum voluntary contraction, and writes the two vectors
    into the session's ``physio.json`` sidecar.
    """
    import json

    import pandas as pd

    sdir = paths.session_dir(sub, ses, data_root=data_root)
    physio_json = sdir / f"sub-{sub}_ses-{ses}_physio.json"
    physio_csv = sdir / f"sub-{sub}_ses-{ses}_task-mvc_run-00_physio.csv"
    if not physio_csv.exists() or not physio_json.exists():
        return

    try:
        df = pd.read_csv(physio_csv)
        sensor_cols = [c for c in df.columns if c.startswith("fsr")]
        if not sensor_cols:
            logger.warning("MVC calibration: no FSR columns in %s", physio_csv)
            return
        # Use REST-phase rows as the rest baseline and RECORD-phase rows as the
        # MVC window. Falls back to "first 1 s of the recording" only if no
        # REST rows exist (legacy data collected before phase tagging).
        phase_col = schema.PHYSIO_PHASE if schema.PHYSIO_PHASE in df.columns else None
        if phase_col is not None and "REST" in df[phase_col].unique():
            baseline_df = df[df[phase_col] == "REST"]
            mvc_df = df[df[phase_col] == "RECORD"]
        else:
            # Legacy fallback: no REST phase recorded.
            t0 = df["t_monotonic_ns"].min()
            df["rel_s"] = (df["t_monotonic_ns"] - t0) / 1e9
            baseline_df = df[df["rel_s"] <= 1.0]
            mvc_df = df
        baseline = (
            baseline_df[sensor_cols].mean().values.tolist()
            if not baseline_df.empty
            else [0.0] * len(sensor_cols)
        )
        mvc = (
            mvc_df[sensor_cols].max().values.tolist()
            if not mvc_df.empty
            else [0.0] * len(sensor_cols)
        )
        with open(physio_json, encoding="utf-8") as f:
            meta = json.load(f)
        meta["BaselineADCVector"] = [int(round(v)) for v in baseline]
        meta["MVCADCVector"] = [int(round(v)) for v in mvc]
        with open(physio_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as exc:
        logger.warning("Could not update MVC calibration from %s: %s", physio_csv, exc)


def append_participants_tsv(
    sub: str,
    *,
    age: str = "n/a",
    sex: str = "n/a",
    handedness: str = "n/a",
    forearm_circumference_mm: str = "n/a",
    forearm_length_mm: str = "n/a",
    data_root: Path | None = None,
) -> None:
    """Append a participant to participants.tsv (if not already present)."""
    root = data_root or paths.RAW_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / "participants.tsv"

    # Read existing
    existing_lines = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()

    # Check if sub already present
    sub_id = f"sub-{sub}"
    for line in existing_lines:
        if line.startswith(sub_id + "\t") or line.startswith(sub_id + " "):
            return  # already present

    # Append
    header = "participant_id\tage\tsex\thandedness\tforearm_circumference_mm\tforearm_length_mm\n"
    with open(path, "a", encoding="utf-8", newline="") as f:
        if not existing_lines or "participant_id" not in existing_lines[0]:
            f.write(header)
        f.write(f"{sub_id}\t{age}\t{sex}\t{handedness}\t{forearm_circumference_mm}\t{forearm_length_mm}\n")


def append_sessions_tsv(
    sub: str,
    ses: str,
    *,
    acq_time: str = "",
    band_placement: str = "forearm_2_3",
    band_tension: str = "medium",
    sensor_count: int = 4,
    sampling_frequency_hz: float = 100.0,
    data_root: Path | None = None,
) -> None:
    """Append a session to sessions.tsv (if not already present)."""
    root = data_root or paths.RAW_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / "sessions.tsv"

    existing_lines = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()

    sub_id = f"sub-{sub}"
    ses_id = f"ses-{ses}"
    for line in existing_lines:
        if line.startswith(f"{sub_id}\t{ses_id}"):
            return  # already present

    if not acq_time:
        from datetime import datetime, timezone
        acq_time = datetime.now(timezone.utc).isoformat()

    header = "participant_id\tsession_id\tacq_time\tband_placement\tband_tension\tsensor_count\tsampling_frequency_hz\n"
    with open(path, "a", encoding="utf-8", newline="") as f:
        if not existing_lines or "participant_id" not in existing_lines[0]:
            f.write(header)
        f.write(f"{sub_id}\t{ses_id}\t{acq_time}\t{band_placement}\t{band_tension}\t{sensor_count}\t{sampling_frequency_hz}\n")
