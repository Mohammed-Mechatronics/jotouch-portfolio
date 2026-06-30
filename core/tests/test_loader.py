"""Tests for core.loader — BIDS data loading."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from core.loader import load_run, load_session, load_all_sessions, _join_by_timestamp
from core.naming import build_filename
from core.schema import PHYSIO_TIMESTAMP, TARGETS_TIMESTAMP, TARGET_COLUMNS, physio_all_columns


def _make_physio_df(n: int = 100, n_sensors: int = 4) -> pd.DataFrame:
    """Create a minimal physio DataFrame for testing."""
    data = {
        PHYSIO_TIMESTAMP: np.arange(n, dtype=np.int64) * 16_666_666,  # 60Hz in ns
        "sample_idx": np.arange(n),
        "phase": ["ACTIVE"] * n,
        "participant_id": ["P01"] * n,
        "session_id": ["S01"] * n,
        "task": ["powerGrip"] * n,
        "run": [1] * n,
    }
    for i in range(n_sensors):
        data[f"fsr{i}"] = np.random.randint(100, 900, size=n)
    data["cue_event"] = [0] * n
    data["led_fsr"] = [0] * n
    return pd.DataFrame(data)


def _make_targets_df(n: int = 100) -> pd.DataFrame:
    """Create a minimal targets DataFrame for testing."""
    data = {PHYSIO_TIMESTAMP: np.arange(n, dtype=np.int64) * 16_666_666}
    for col in TARGET_COLUMNS:
        data[col] = np.random.uniform(0, 90, size=n)
    return pd.DataFrame(data)


def _make_camera_df(n: int = 50) -> pd.DataFrame:
    """Create a minimal camera DataFrame for testing."""
    data = {
        "cam_ts_ns": np.arange(n, dtype=np.int64) * 33_333_333,  # 30Hz in ns
        "mp_valid": [1] * n,
        "mp_confidence": [0.95] * n,
        "mp_handedness": ["Right"] * n,
    }
    for i in range(21):
        for axis in ("x", "y", "z"):
            data[f"mp_lm{i:02d}_{axis}"] = np.random.uniform(-0.5, 0.5, size=n)
    data["led_cam"] = [0] * n
    return pd.DataFrame(data)


def _write_bids_run(session_dir: Path, sub: str, ses: str, task: str, run: int):
    """Write a complete BIDS run (3 CSVs) to a session directory."""
    physio = _make_physio_df()
    camera = _make_camera_df()
    targets = _make_targets_df()

    physio.to_csv(session_dir / build_filename(sub, ses, task, run, "physio"), index=False)
    camera.to_csv(session_dir / build_filename(sub, ses, task, run, "camera"), index=False)
    targets.to_csv(session_dir / build_filename(sub, ses, task, run, "targets"), index=False)


def _write_manifest(session_dir: Path, sub: str, ses: str, task: str, run: int,
                    complete: bool = True):
    """Write a manifest.json sidecar for a run."""
    manifest = {
        "sub": sub, "ses": ses, "task": task, "run": run,
        "physio_rows": 100, "camera_rows": 50, "targets_rows": 100,
        "bad_physio_count": 0, "bad_camera_count": 0, "bad_targets_count": 0,
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        "complete": complete,
    }
    mpath = session_dir / f"sub-{sub}_ses-{ses}_task-{task}_run-{run:02d}_manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f)


class TestLoadRun:
    def test_load_existing_run(self, tmp_path):
        sub, ses = "P01", "S01"
        sdir = tmp_path / f"sub-{sub}" / f"ses-{ses}"
        sdir.mkdir(parents=True)
        _write_bids_run(sdir, sub, ses, "powerGrip", 1)

        run = load_run(sub, ses, "powerGrip", 1, data_root=tmp_path)
        assert run.sub == sub
        assert run.ses == ses
        assert run.task == "powerGrip"
        assert run.run == 1
        assert len(run.physio) == 100
        assert len(run.camera) == 50
        assert len(run.targets) == 100
        assert run.n_sensors == 4
        assert run.is_baseline is False
        assert run.phase == "multi_dof"

    def test_load_missing_run(self, tmp_path):
        run = load_run("P99", "S99", "nonexistent", 1, data_root=tmp_path)
        assert run.physio.empty
        assert run.camera.empty
        assert run.targets.empty

    def test_load_baseline(self, tmp_path):
        sub, ses = "P01", "S01"
        sdir = tmp_path / f"sub-{sub}" / f"ses-{ses}"
        sdir.mkdir(parents=True)
        _write_bids_run(sdir, sub, ses, "mvc", 0)

        run = load_run(sub, ses, "mvc", 0, data_root=tmp_path)
        assert run.is_baseline is True
        assert run.phase == "baseline"

    def test_led_sync_correction_applied(self, tmp_path):
        """Camera timestamps are corrected when led_sync.json has passed=True."""
        import json

        sub, ses = "P01", "S01"
        sdir = tmp_path / f"sub-{sub}" / f"ses-{ses}"
        sdir.mkdir(parents=True)
        _write_bids_run(sdir, sub, ses, "powerGrip", 1)

        # Load original (no sync file yet)
        from core.schema import CAMERA_TIMESTAMP
        run_original = load_run(sub, ses, "powerGrip", 1, data_root=tmp_path)
        original_ts = run_original.camera[CAMERA_TIMESTAMP].values.copy()

        # Write a led_sync.json sidecar with correction coefficients
        sync_data = {
            "sync_check_version": "1.0",
            "method": "peaks",
            "skew_ms": 15.0,
            "passed": True,
            "a": 1.0,
            "b": 1_000_000,  # 1 ms offset in ns
            "offset_ms": 15.0,
            "timestamp_utc": "2026-01-01T00:00:00Z",
        }
        sync_path = sdir / f"sub-{sub}_ses-{ses}_led_sync.json"
        with open(sync_path, "w") as f:
            json.dump(sync_data, f)

        # The loader should now apply the correction
        run = load_run(sub, ses, "powerGrip", 1, data_root=tmp_path)
        corrected_ts = run.camera[CAMERA_TIMESTAMP].values

        # t_corrected = a * t + b = 1.0 * t + 1_000_000
        np.testing.assert_array_equal(corrected_ts, original_ts + 1_000_000)

    def test_led_sync_not_applied_when_failed(self, tmp_path):
        """Camera timestamps are NOT corrected when led_sync.json has passed=False."""
        import json

        sub, ses = "P01", "S01"
        sdir = tmp_path / f"sub-{sub}" / f"ses-{ses}"
        sdir.mkdir(parents=True)
        _write_bids_run(sdir, sub, ses, "powerGrip", 1)

        sync_data = {
            "passed": False,
            "a": 2.0,
            "b": 5_000_000,
        }
        sync_path = sdir / f"sub-{sub}_ses-{ses}_led_sync.json"
        with open(sync_path, "w") as f:
            json.dump(sync_data, f)

        from core.schema import CAMERA_TIMESTAMP
        run = load_run(sub, ses, "powerGrip", 1, data_root=tmp_path)
        # Timestamps should NOT be modified
        assert run.camera[CAMERA_TIMESTAMP].iloc[0] == 0  # First timestamp from _make_camera_df

    def test_led_sync_not_applied_when_missing(self, tmp_path):
        """No correction when no led_sync.json exists."""
        sub, ses = "P01", "S01"
        sdir = tmp_path / f"sub-{sub}" / f"ses-{ses}"
        sdir.mkdir(parents=True)
        _write_bids_run(sdir, sub, ses, "powerGrip", 1)

        from core.schema import CAMERA_TIMESTAMP
        run = load_run(sub, ses, "powerGrip", 1, data_root=tmp_path)
        assert run.camera[CAMERA_TIMESTAMP].iloc[0] == 0

    def test_load_run_without_manifest_loads_by_default(self, tmp_path):
        """Without require_manifest, a run with no manifest still loads (backward compat)."""
        sub, ses = "P01", "S01"
        sdir = tmp_path / f"sub-{sub}" / f"ses-{ses}"
        sdir.mkdir(parents=True)
        _write_bids_run(sdir, sub, ses, "powerGrip", 1)
        # No manifest written
        run = load_run(sub, ses, "powerGrip", 1, data_root=tmp_path)
        assert not run.physio.empty

    def test_load_run_skips_partial_when_require_manifest(self, tmp_path):
        """require_manifest=True skips runs with no manifest (crash partial)."""
        sub, ses = "P01", "S01"
        sdir = tmp_path / f"sub-{sub}" / f"ses-{ses}"
        sdir.mkdir(parents=True)
        _write_bids_run(sdir, sub, ses, "powerGrip", 1)
        # No manifest → partial run
        run = load_run(sub, ses, "powerGrip", 1, data_root=tmp_path, require_manifest=True)
        assert run.physio.empty
        assert run.camera.empty
        assert run.targets.empty

    def test_load_run_with_manifest_loads_when_required(self, tmp_path):
        """require_manifest=True loads runs that have a valid complete manifest."""
        sub, ses = "P01", "S01"
        sdir = tmp_path / f"sub-{sub}" / f"ses-{ses}"
        sdir.mkdir(parents=True)
        _write_bids_run(sdir, sub, ses, "powerGrip", 1)
        _write_manifest(sdir, sub, ses, "powerGrip", 1, complete=True)
        run = load_run(sub, ses, "powerGrip", 1, data_root=tmp_path, require_manifest=True)
        assert not run.physio.empty
        assert len(run.physio) == 100

    def test_load_run_skips_incomplete_manifest(self, tmp_path):
        """require_manifest=True skips runs whose manifest has complete=False."""
        sub, ses = "P01", "S01"
        sdir = tmp_path / f"sub-{sub}" / f"ses-{ses}"
        sdir.mkdir(parents=True)
        _write_bids_run(sdir, sub, ses, "powerGrip", 1)
        _write_manifest(sdir, sub, ses, "powerGrip", 1, complete=False)
        run = load_run(sub, ses, "powerGrip", 1, data_root=tmp_path, require_manifest=True)
        assert run.physio.empty


class TestLoadSession:
    def test_load_session_with_multiple_runs(self, tmp_path):
        sub, ses = "P01", "S01"
        sdir = tmp_path / f"sub-{sub}" / f"ses-{ses}"
        sdir.mkdir(parents=True)

        _write_bids_run(sdir, sub, ses, "mvc", 0)
        _write_bids_run(sdir, sub, ses, "thumbCmcIso", 1)
        _write_bids_run(sdir, sub, ses, "powerGrip", 1)
        _write_bids_run(sdir, sub, ses, "freeform", 1)

        session = load_session(sub, ses, data_root=tmp_path)
        assert session.n_runs == 4
        assert "mvc" in session.tasks()
        assert "thumbCmcIso" in session.tasks()
        assert "powerGrip" in session.tasks()
        assert "freeform" in session.tasks()

    def test_load_empty_session(self, tmp_path):
        session = load_session("P99", "S99", data_root=tmp_path)
        assert session.n_runs == 0

    def test_runs_in_phase(self, tmp_path):
        sub, ses = "P01", "S01"
        sdir = tmp_path / f"sub-{sub}" / f"ses-{ses}"
        sdir.mkdir(parents=True)

        _write_bids_run(sdir, sub, ses, "mvc", 0)
        _write_bids_run(sdir, sub, ses, "thumbCmcIso", 1)
        _write_bids_run(sdir, sub, ses, "powerGrip", 1)
        _write_bids_run(sdir, sub, ses, "freeform", 1)

        session = load_session(sub, ses, data_root=tmp_path)
        assert len(session.runs_in_phase("baseline")) == 1
        assert len(session.runs_in_phase("single_dof")) == 1
        assert len(session.runs_in_phase("multi_dof")) == 1
        assert len(session.runs_in_phase("freeform")) == 1


class TestJoinByTimestamp:
    def test_join_aligned(self):
        n = 50
        physio = _make_physio_df(n)
        targets = _make_targets_df(n)
        merged = _join_by_timestamp(physio, targets, PHYSIO_TIMESTAMP, TARGETS_TIMESTAMP)
        assert len(merged) == n
        assert "target_thumb_cmc_flex" in merged.columns

    def test_join_different_rates(self):
        physio = _make_physio_df(100)  # 60Hz
        targets = _make_targets_df(100)
        # Make targets at 30Hz (every other sample)
        targets[TARGETS_TIMESTAMP] = targets[TARGETS_TIMESTAMP] * 2
        merged = _join_by_timestamp(physio, targets, PHYSIO_TIMESTAMP, TARGETS_TIMESTAMP)
        # Should have 100 physio rows, with targets matched where possible
        assert len(merged) == 100

    def test_join_empty(self):
        merged = _join_by_timestamp(pd.DataFrame(), pd.DataFrame(), "ts1", "ts2")
        assert merged.empty


def _make_physio_df_with_phases(n_prep=20, n_record=60, n_rest=20, n_sensors=4,
                                 task="powerGrip", sub="P01", ses="S01", run=1):
    """Create a physio DataFrame with PREP/RECORD/REST phases."""
    n = n_prep + n_record + n_rest
    ts = np.arange(n, dtype=np.int64) * 10_000_000  # 100Hz in ns
    phases = ["PREP"] * n_prep + ["RECORD"] * n_record + ["REST"] * n_rest
    data = {
        PHYSIO_TIMESTAMP: ts,
        "sample_idx": np.arange(n),
        "phase": phases,
        "participant_id": [sub] * n,
        "session_id": [ses] * n,
        "task": [task] * n,
        "run": [run] * n,
    }
    for i in range(n_sensors):
        data[f"fsr{i}"] = np.random.randint(100, 900, size=n)
    data["cue_event"] = ["PREP_START"] + [""] * (n - 3) + ["RECORD_START", ""] * 0 + ["REST_START"]
    data["cue_event"] = [""] * n
    data["cue_event"][0] = "PREP_START"
    data["cue_event"][n_prep] = "RECORD_START"
    data["cue_event"][n_prep + n_record] = "REST_START"
    data["led_fsr"] = [0] * n
    return pd.DataFrame(data)


def _make_targets_df_with_phases(n_prep=20, n_record=60, n_rest=20):
    """Create a targets DataFrame matching the physio phases."""
    n = n_prep + n_record + n_rest
    data = {PHYSIO_TIMESTAMP: np.arange(n, dtype=np.int64) * 10_000_000}
    for col in TARGET_COLUMNS:
        data[col] = np.random.uniform(0, 90, size=n)
    return pd.DataFrame(data)


def _write_bids_run_with_phases(session_dir: Path, sub: str, ses: str, task: str, run: int,
                                 n_prep=20, n_record=60, n_rest=20):
    """Write a BIDS run with PREP/RECORD/REST phases."""
    physio = _make_physio_df_with_phases(n_prep, n_record, n_rest, task=task, sub=sub, ses=ses, run=run)
    targets = _make_targets_df_with_phases(n_prep, n_record, n_rest)
    camera = _make_camera_df(50)

    physio.to_csv(session_dir / build_filename(sub, ses, task, run, "physio"), index=False)
    camera.to_csv(session_dir / build_filename(sub, ses, task, run, "camera"), index=False)
    targets.to_csv(session_dir / build_filename(sub, ses, task, run, "targets"), index=False)


class TestPhaseBasedLabeling:
    """Tests for protocol-phase-based classification label extraction.

    The collection protocol writes PREP/RECORD/REST phase labels.  Classification
    data should use RECORD-phase samples only, labeled with the task name.
    No 'rest' class — PREP and REST are excluded entirely.
    """

    def test_classification_uses_record_phase_only(self, tmp_path):
        """load_classification_data must only return RECORD-phase samples."""
        from core.loader import load_classification_data

        sub, ses = "P01", "S01"
        sdir = tmp_path / f"sub-{sub}" / f"ses-{ses}"
        sdir.mkdir(parents=True)
        _write_bids_run_with_phases(sdir, sub, ses, "powerGrip", 1, n_prep=20, n_record=60, n_rest=20)
        _write_manifest(sdir, sub, ses, "powerGrip", 1, complete=True)

        X, y, meta = load_classification_data(data_root=tmp_path)
        assert X.shape[0] == 60  # only RECORD samples
        assert all(label == "powerGrip" for label in y)
        assert "rest" not in set(y)

    def test_classification_excludes_mvc_and_freeform(self, tmp_path):
        """MVC (baseline) and freeform are excluded from classification."""
        from core.loader import load_classification_data

        sub, ses = "P01", "S01"
        sdir = tmp_path / f"sub-{sub}" / f"ses-{ses}"
        sdir.mkdir(parents=True)
        for task, run in [("mvc", 0), ("powerGrip", 1), ("freeform", 1)]:
            _write_bids_run_with_phases(sdir, sub, ses, task, run)
            _write_manifest(sdir, sub, ses, task, run, complete=True)

        X, y, meta = load_classification_data(data_root=tmp_path)
        labels = set(y)
        assert "powerGrip" in labels
        assert "mvc" not in labels
        assert "freeform" not in labels

    def test_classification_no_rest_class(self, tmp_path):
        """No 'rest' label should appear — RECORD only, no rest class."""
        from core.loader import load_classification_data

        sub, ses = "P01", "S01"
        sdir = tmp_path / f"sub-{sub}" / f"ses-{ses}"
        sdir.mkdir(parents=True)
        _write_bids_run_with_phases(sdir, sub, ses, "powerGrip", 1)
        _write_manifest(sdir, sub, ses, "powerGrip", 1, complete=True)

        X, y, meta = load_classification_data(data_root=tmp_path)
        assert "rest" not in set(y)

    def test_regression_uses_record_phase_only(self, tmp_path):
        """load_regression_data must only return RECORD-phase samples."""
        from core.loader import load_regression_data

        sub, ses = "P01", "S01"
        sdir = tmp_path / f"sub-{sub}" / f"ses-{ses}"
        sdir.mkdir(parents=True)
        _write_bids_run_with_phases(sdir, sub, ses, "powerGrip", 1, n_prep=20, n_record=60, n_rest=20)
        _write_manifest(sdir, sub, ses, "powerGrip", 1, complete=True)

        X, Y, meta = load_regression_data(data_root=tmp_path)
        assert X.shape[0] == 60  # only RECORD samples
        assert Y.shape[1] == 15  # 15 joint angle targets

    def test_regression_includes_mvc_excludes_freeform(self, tmp_path):
        """Regression includes MVC (max force range) but excludes freeform."""
        from core.loader import load_regression_data

        sub, ses = "P01", "S01"
        sdir = tmp_path / f"sub-{sub}" / f"ses-{ses}"
        sdir.mkdir(parents=True)
        for task, run in [("mvc", 0), ("powerGrip", 1), ("freeform", 1)]:
            _write_bids_run_with_phases(sdir, sub, ses, task, run)
            _write_manifest(sdir, sub, ses, task, run, complete=True)

        X, Y, meta = load_regression_data(data_root=tmp_path)
        tasks = set(meta["task"])
        assert "mvc" in tasks  # MVC included for regression
        assert "freeform" not in tasks  # freeform excluded
