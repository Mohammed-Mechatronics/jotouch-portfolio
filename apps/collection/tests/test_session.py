"""Tests for apps/collection/session.py — the generator-based session controller."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.collection.camera import MockCameraReader
from apps.collection.protocol import RunSpec, build_protocol
from apps.collection.sensor_reader import MockSensorReader
from apps.collection.session import (
    CameraEvent,
    FsrEvent,
    QualityEvent,
    RunCompleteEvent,
    SetupEvent,
    StateEvent,
    SummaryEvent,
    TestEvent,
    run_session,
)


def _single_run(
    task: str = "mvc",
    run: int = 0,
    phase: str = "baseline",
    record_duration: float = 0.5,
    prep_duration: float = 0.2,
    rest_duration: float = 0.2,
) -> list[RunSpec]:
    """Return a single-run protocol for fast tests."""
    return [
        RunSpec(
            task=task,
            run=run,
            rep=1,
            phase=phase,
            record_duration=record_duration,
            prep_duration=prep_duration,
            rest_duration=rest_duration,
        )
    ]


def _collect_events(runs, *, skip_precollect: bool = True, precollect_countdown_s: float = 1.0, **kwargs):
    """Helper to drain all events from a short dry-run session."""
    sensor_reader = MockSensorReader(n_sensors=4, seed=42)
    camera_reader = MockCameraReader(seed=42)
    sensor_reader.start()
    camera_reader.start()

    events = list(run_session(
        sensor_reader=sensor_reader,
        camera_reader=camera_reader,
        runs=runs,
        sub="P01",
        ses="S01",
        n_sensors=4,
        skip_precollect=skip_precollect,
        dry_run=True,
        precollect_countdown_s=precollect_countdown_s,
        **kwargs,
    ))

    sensor_reader.stop()
    camera_reader.stop()
    return events


class TestSessionEvents:
    def test_setup_event(self, tmp_path):
        events = _collect_events(
            runs=_single_run(),
            data_root=tmp_path,
        )
        assert isinstance(events[0], SetupEvent)
        assert events[0].sub == "P01"
        assert events[0].ses == "S01"
        assert events[0].n_sensors == 4

    def test_state_events_during_run(self, tmp_path):
        events = _collect_events(
            runs=_single_run(),
            data_root=tmp_path,
        )
        states = [e for e in events if isinstance(e, StateEvent)]
        assert states
        phases = {s.phase for s in states}
        assert phases == {"PREP", "RECORD", "REST"}

    def test_state_events_include_cue_fields(self, tmp_path):
        """StateEvent must carry display_name, description, and instruction
        from the task cue so the UI can show the patient what to do."""
        events = _collect_events(
            runs=_single_run(task="powerGrip"),
            data_root=tmp_path,
        )
        states = [e for e in events if isinstance(e, StateEvent)]
        assert states
        # Every state event for this run should carry the cue fields
        for s in states:
            assert s.display_name, f"StateEvent phase={s.phase} has empty display_name"
            assert s.description, f"StateEvent phase={s.phase} has empty description"
            assert s.instruction, f"StateEvent phase={s.phase} has empty instruction"
        # The first state (PREP) should have the powerGrip cue
        prep = next(s for s in states if s.phase == "PREP")
        assert prep.display_name == "Power Grip"
        assert "cylindrical" in prep.description.lower()
        assert "fist" in prep.instruction.lower()
        # REST phase should have a relax instruction
        rest = next(s for s in states if s.phase == "REST")
        assert "relax" in rest.instruction.lower()

    def test_fsr_and_camera_events(self, tmp_path):
        events = _collect_events(
            runs=_single_run(record_duration=1.0),
            data_root=tmp_path,
        )
        fsr_events = [e for e in events if isinstance(e, FsrEvent)]
        cam_events = [e for e in events if isinstance(e, CameraEvent)]
        assert len(fsr_events) >= 10
        assert len(cam_events) >= 1
        assert len(fsr_events[0].values) == 4

    def test_quality_events(self, tmp_path):
        events = _collect_events(
            runs=_single_run(),
            data_root=tmp_path,
        )
        quality = [e for e in events if isinstance(e, QualityEvent)]
        assert quality
        assert quality[0].level in ("green", "yellow", "red")

    def test_run_complete_and_summary(self, tmp_path):
        events = _collect_events(
            runs=_single_run(),
            data_root=tmp_path,
        )
        run_completes = [e for e in events if isinstance(e, RunCompleteEvent)]
        summaries = [e for e in events if isinstance(e, SummaryEvent)]
        assert run_completes
        assert summaries
        assert summaries[0].completed_runs == len(run_completes)

    def test_multi_run_protocol(self, tmp_path):
        events = _collect_events(
            runs=build_protocol(
                n_reps=1,
                include_freeform=False,
                record_duration=0.2,
                prep_duration=0.1,
                rest_duration=0.1,
            ),
            data_root=tmp_path,
        )
        run_completes = [e for e in events if isinstance(e, RunCompleteEvent)]
        assert len(run_completes) == 25  # 1 baseline + 15 single + 9 multi

    def test_precollect_events(self, tmp_path, monkeypatch):
        # Speed up the interactive precollect tests so this unit test completes quickly.
        from apps.collection import precollect
        fast_meta = {k: {**v, "duration_s": 0.1} for k, v in precollect.TEST_META.items()}
        monkeypatch.setattr(precollect, "TEST_META", fast_meta)

        events = _collect_events(
            runs=_single_run(record_duration=0.2),
            skip_precollect=False,
            precollect_countdown_s=0.1,
            data_root=tmp_path,
        )
        tests = [e for e in events if isinstance(e, TestEvent)]
        assert len(tests) == 8  # exactly 8 pre-collection tests


class TestSessionData:
    def test_physio_json_sampling_frequency(self, tmp_path):
        _collect_events(
            runs=_single_run(),
            data_root=tmp_path,
        )
        physio_json_path = tmp_path / "sub-P01" / "ses-S01" / "sub-P01_ses-S01_physio.json"
        assert physio_json_path.exists()
        with open(physio_json_path, encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["SamplingFrequency"] == 100.0

    def test_physio_sample_rate_is_100hz(self, tmp_path):
        """RECORD-phase physio samples at ~100 Hz (±3% tolerance)."""
        _collect_events(
            runs=_single_run(task="mvc", record_duration=2.0, prep_duration=0.1, rest_duration=0.1),
            data_root=tmp_path,
        )
        physio_path = tmp_path / "sub-P01" / "ses-S01" / "sub-P01_ses-S01_task-mvc_run-00_physio.csv"
        assert physio_path.exists()
        import csv as _csv
        with open(physio_path, encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
        # Filter to RECORD-phase rows only
        record_rows = [r for r in rows if r["phase"] == "RECORD"]
        # 2-second RECORD at 100 Hz = ~200 samples (±3%)
        assert 194 <= len(record_rows) <= 206, \
            f"Expected ~200 RECORD rows, got {len(record_rows)}"
        # Check median inter-sample interval is roughly 10 ms (allow overhead
        # from camera reads / quality checks in the mock loop).
        ts = [int(r["t_monotonic_ns"]) for r in record_rows]
        diffs = [ts[i+1] - ts[i] for i in range(len(ts)-1)]
        if diffs:
            diffs.sort()
            median_ms = diffs[len(diffs)//2] / 1e6
            assert 7.0 <= median_ms <= 16.0, f"Median interval {median_ms:.1f}ms, expected ~10ms"

    def test_led_fsr_column_present(self, tmp_path):
        _collect_events(
            runs=_single_run(),
            data_root=tmp_path,
        )
        physio_path = tmp_path / "sub-P01" / "ses-S01" / "sub-P01_ses-S01_task-mvc_run-00_physio.csv"
        with open(physio_path, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        assert "led_fsr" in header

    def test_camera_csv_columns(self, tmp_path):
        _collect_events(
            runs=_single_run(record_duration=1.0),
            data_root=tmp_path,
        )
        camera_path = tmp_path / "sub-P01" / "ses-S01" / "sub-P01_ses-S01_task-mvc_run-00_camera.csv"
        assert camera_path.exists()
        with open(camera_path, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        assert "cam_ts_ns" in header
        assert "led_cam" in header

    def test_targets_csv_columns(self, tmp_path):
        _collect_events(
            runs=_single_run(),
            data_root=tmp_path,
        )
        targets_path = tmp_path / "sub-P01" / "ses-S01" / "sub-P01_ses-S01_task-mvc_run-00_targets.csv"
        assert targets_path.exists()
        with open(targets_path, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        assert "target_thumb_cmc_flex" in header

    def test_physio_has_all_three_phases(self, tmp_path):
        """Physio CSV must contain PREP, RECORD, and REST phase rows."""
        _collect_events(
            runs=_single_run(prep_duration=0.3, record_duration=0.5, rest_duration=0.3),
            data_root=tmp_path,
        )
        physio_path = tmp_path / "sub-P01" / "ses-S01" / "sub-P01_ses-S01_task-mvc_run-00_physio.csv"
        import csv as _csv
        with open(physio_path, encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
        phases = {r["phase"] for r in rows}
        assert phases == {"PREP", "RECORD", "REST"}, f"Got phases: {phases}"

    def test_cue_event_markers_present(self, tmp_path):
        """Phase transitions must write cue_event markers (PREP_START, RECORD_START, REST_START)."""
        _collect_events(
            runs=_single_run(prep_duration=0.2, record_duration=0.3, rest_duration=0.2),
            data_root=tmp_path,
        )
        physio_path = tmp_path / "sub-P01" / "ses-S01" / "sub-P01_ses-S01_task-mvc_run-00_physio.csv"
        import csv as _csv
        with open(physio_path, encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
        cue_events = {r["cue_event"] for r in rows if r["cue_event"]}
        assert "PREP_START" in cue_events
        assert "RECORD_START" in cue_events
        assert "REST_START" in cue_events

    def test_physio_has_quality_flag_column(self, tmp_path):
        _collect_events(
            runs=_single_run(),
            data_root=tmp_path,
        )
        physio_path = tmp_path / "sub-P01" / "ses-S01" / "sub-P01_ses-S01_task-mvc_run-00_physio.csv"
        with open(physio_path, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        assert "quality_flag" in header

    def test_manifest_written_for_completed_run(self, tmp_path):
        """Each completed run must have a manifest.json with complete=True."""
        _collect_events(
            runs=_single_run(),
            data_root=tmp_path,
        )
        manifest_path = (
            tmp_path / "sub-P01" / "ses-S01"
            / "sub-P01_ses-S01_task-mvc_run-00_manifest.json"
        )
        assert manifest_path.exists()
        import json as _json
        with open(manifest_path, encoding="utf-8") as f:
            m = _json.load(f)
        assert m["complete"] is True
        assert m["physio_rows"] > 0

    def test_complete_sentinel_written(self, tmp_path):
        _collect_events(
            runs=_single_run(),
            data_root=tmp_path,
        )
        sentinel = (
            tmp_path / "sub-P01" / "ses-S01"
            / "sub-P01_ses-S01_task-mvc_run-00.complete"
        )
        assert sentinel.exists()

    def test_stop_mid_run_quarantines_partial(self, tmp_path):
        """When the operator stops mid-run, the partial run is quarantined to _partial/."""
        import threading
        stop_event = threading.Event()
        sensor = MockSensorReader(n_sensors=4)
        camera = MockCameraReader()
        sensor.start()
        camera.start()

        runs = _single_run(prep_duration=0.1, record_duration=2.0, rest_duration=0.1)

        events = []
        gen = run_session(
            sub="P01", ses="S01", sensor_reader=sensor, camera_reader=camera,
            runs=runs, n_sensors=4, dry_run=True, skip_precollect=True,
            data_root=tmp_path, stop_event=stop_event,
        )
        # Consume events until we see RECORD phase, then stop
        for event in gen:
            events.append(event)
            if isinstance(event, StateEvent) and event.phase == "RECORD":
                stop_event.set()
                break
        # Drain remaining events
        for event in gen:
            events.append(event)

        # The partial run should be in _partial/, not in the session dir
        sdir = tmp_path / "sub-P01" / "ses-S01"
        partial_dir = sdir / "_partial"
        assert partial_dir.exists()
        # No .complete sentinel in the session dir
        sentinels = list(sdir.glob("*.complete"))
        assert len(sentinels) == 0
        # No physio CSV in the session dir (moved to _partial/)
        physio_in_main = list(sdir.glob("*_physio.csv"))
        assert len(physio_in_main) == 0
        # CSVs are in _partial/
        partial_csvs = list(partial_dir.rglob("*.csv"))
        assert len(partial_csvs) == 3

    def test_reuse_run_number_after_quarantine(self, tmp_path):
        """After a run is quarantined, the next session can reuse the same run number."""
        # First session: complete a run at run-00
        _collect_events(
            runs=_single_run(task="mvc", record_duration=0.3, prep_duration=0.1, rest_duration=0.1),
            data_root=tmp_path,
        )
        # Verify run-00 completed
        sdir = tmp_path / "sub-P01" / "ses-S01"
        assert (sdir / "sub-P01_ses-S01_task-mvc_run-00.complete").exists()

        # Now simulate a second session that tries to re-run run-00
        # The existing run-00 is complete, so it should auto-increment
        import threading
        stop_event = threading.Event()
        sensor = MockSensorReader(n_sensors=4)
        camera = MockCameraReader()
        sensor.start()
        camera.start()
        runs = _single_run(task="mvc", record_duration=0.3, prep_duration=0.1, rest_duration=0.1)
        events = list(run_session(
            sub="P01", ses="S01", sensor_reader=sensor, camera_reader=camera,
            runs=runs, n_sensors=4, dry_run=True, skip_precollect=True,
            data_root=tmp_path, stop_event=stop_event,
        ))
        # The second run should have auto-incremented to run-01
        run01_sentinel = sdir / "sub-P01_ses-S01_task-mvc_run-01.complete"
        assert run01_sentinel.exists()
        # Original run-00 is still intact
        assert (sdir / "sub-P01_ses-S01_task-mvc_run-00.complete").exists()

    def test_summary_includes_quarantined_count(self, tmp_path):
        """SummaryEvent must include quarantined_runs count when a run is aborted."""
        import threading
        stop_event = threading.Event()
        sensor = MockSensorReader(n_sensors=4)
        camera = MockCameraReader()
        sensor.start()
        camera.start()

        runs = _single_run(prep_duration=0.1, record_duration=2.0, rest_duration=0.1)
        events = []
        gen = run_session(
            sub="P01", ses="S01", sensor_reader=sensor, camera_reader=camera,
            runs=runs, n_sensors=4, dry_run=True, skip_precollect=True,
            data_root=tmp_path, stop_event=stop_event,
        )
        for event in gen:
            events.append(event)
            if isinstance(event, StateEvent) and event.phase == "RECORD":
                stop_event.set()

        summary = next(e for e in events if isinstance(e, SummaryEvent))
        # The run was stopped mid-RECORD, so it was quarantined
        assert summary.quarantined_runs >= 1
        assert summary.completed_runs == 0

    def test_run_skipped_event_emitted_on_abort(self, tmp_path):
        """A run_skipped event must be emitted when a run is quarantined due to stop."""
        import threading
        stop_event = threading.Event()
        sensor = MockSensorReader(n_sensors=4)
        camera = MockCameraReader()
        sensor.start()
        camera.start()

        runs = _single_run(prep_duration=0.1, record_duration=2.0, rest_duration=0.1)
        events = []
        gen = run_session(
            sub="P01", ses="S01", sensor_reader=sensor, camera_reader=camera,
            runs=runs, n_sensors=4, dry_run=True, skip_precollect=True,
            data_root=tmp_path, stop_event=stop_event,
        )
        for event in gen:
            events.append(event)
            if isinstance(event, StateEvent) and event.phase == "RECORD":
                stop_event.set()

        # There must be a run_skipped event (dict with type="run_skipped")
        skipped = [e for e in events if isinstance(e, dict) and e.get("type") == "run_skipped"]
        assert len(skipped) >= 1
        assert "task" in skipped[0]
        assert "run" in skipped[0]

    def test_summary_quarantined_zero_on_clean_completion(self, tmp_path):
        """SummaryEvent must report quarantined_runs=0 when all runs complete."""
        events = _collect_events(
            runs=_single_run(record_duration=0.2, prep_duration=0.1, rest_duration=0.1),
            data_root=tmp_path,
        )
        summary = next(e for e in events if isinstance(e, SummaryEvent))
        assert summary.quarantined_runs == 0
        assert summary.completed_runs == 1


# ── trigger_sync() session wiring (ADR 003) ─────────────────────────────────


class _RecordingMockSensorReader(MockSensorReader):
    """MockSensorReader that records trigger_sync() calls for inspection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trigger_calls = 0

    def trigger_sync(self) -> None:
        self.trigger_calls += 1
        super().trigger_sync()


class TestSessionTriggerSync:
    """The session loop must call sensor_reader.trigger_sync() once, before
    the first RECORD phase, so the PRBS preamble is captured at the start
    of the recorded CSV (ADR 003).
    """

    def test_trigger_called_once_for_single_run(self, tmp_path):
        sensor = _RecordingMockSensorReader(n_sensors=4, seed=1)
        camera = MockCameraReader(seed=1)
        sensor.start()
        camera.start()
        try:
            events = list(run_session(
                sub="P01", ses="S01",
                sensor_reader=sensor, camera_reader=camera,
                runs=_single_run(record_duration=0.2, prep_duration=0.1, rest_duration=0.1),
                n_sensors=4, dry_run=True, skip_precollect=True,
                data_root=tmp_path,
            ))
        finally:
            sensor.stop()
            camera.stop()

        assert sensor.trigger_calls == 1, (
            f"trigger_sync() must be called exactly once for a single-run session, "
            f"got {sensor.trigger_calls}"
        )
        # Verify the trigger happened before the first RECORD StateEvent
        trigger_index = None
        first_record_index = None
        for i, e in enumerate(events):
            if isinstance(e, dict) and e.get("type") == "trigger_sync":
                trigger_index = i
            if isinstance(e, StateEvent) and e.phase == "RECORD" and first_record_index is None:
                first_record_index = i
        # The session doesn't emit a "trigger_sync" event — it just calls the
        # method. So we can't find it in events. Instead, assert it was called
        # exactly once (above) and that a RECORD phase actually happened.
        assert first_record_index is not None, "No RECORD phase was emitted"

    def test_trigger_called_once_for_multi_run_session(self, tmp_path):
        """trigger_sync() must be called only ONCE for the first run of a
        multi-run session. Subsequent runs reuse the same PRBS preamble
        (Stage 2 NAd tracks drift across the whole session).
        """
        sensor = _RecordingMockSensorReader(n_sensors=4, seed=2)
        camera = MockCameraReader(seed=2)
        sensor.start()
        camera.start()
        try:
            runs = [
                RunSpec(task="mvc", run=0, rep=1, phase="baseline",
                        record_duration=0.2, prep_duration=0.1, rest_duration=0.1),
                RunSpec(task="powerGrip", run=1, rep=1, phase="train",
                        record_duration=0.2, prep_duration=0.1, rest_duration=0.1),
                RunSpec(task="powerGrip", run=2, rep=1, phase="train",
                        record_duration=0.2, prep_duration=0.1, rest_duration=0.1),
            ]
            list(run_session(
                sub="P02", ses="S01",
                sensor_reader=sensor, camera_reader=camera,
                runs=runs, n_sensors=4, dry_run=True, skip_precollect=True,
                data_root=tmp_path,
            ))
        finally:
            sensor.stop()
            camera.stop()

        assert sensor.trigger_calls == 1, (
            f"trigger_sync() must be called exactly once across 3 runs, "
            f"got {sensor.trigger_calls}"
        )
