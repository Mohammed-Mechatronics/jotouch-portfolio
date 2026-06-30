"""Session controller — generator-based collection loop.

Replaces the blocking ``run_single_run()`` in ``apps/collection.__main__`` with a
non-blocking generator that yields structured events while writing BIDS data to
disk in real time.

Usage:
    from apps.collection.session import run_session
    from apps.collection.sensor_reader import create_sensor_reader
    from apps.collection.camera import create_camera_reader
    from apps.collection.protocol import build_protocol

    sensor_reader = create_sensor_reader(dry_run=True, n_sensors=4)
    camera_reader = create_camera_reader(dry_run=True)
    sensor_reader.start()
    camera_reader.start()

    for event in run_session(
        sub="P01", ses="S01",
        sensor_reader=sensor_reader,
        camera_reader=camera_reader,
        runs=build_protocol(),
        n_sensors=4,
        dry_run=True,
    ):
        print(event)

    sensor_reader.stop()
    camera_reader.stop()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any, Callable, Iterator, Protocol

logger = logging.getLogger(__name__)

from apps.collection.timer import now_ns
from apps.collection.bids_writer import (
    BIDSRunWriter,
    append_participants_tsv,
    append_sessions_tsv,
    quarantine_partial_run,
    sweep_orphan_partials,
    update_mvc_calibration,
    write_dataset_description,
    write_session_metadata,
)
from apps.collection.cues import get_cue, format_cue_for_display, format_countdown
from apps.collection.precollect import run_all_tests_interactive, PrecollectResults, TestResult
from apps.collection.protocol import RunSpec
from apps.collection.quality import QualityMonitor
from core import naming, paths, schema
from core.joint_angles import landmarks_to_joint_angles


# ── Events ────────────────────────────────────────────────────────────────────


@dataclass
class SetupEvent:
    """Emitted once at the start of a session."""

    type: str = "setup"
    sub: str = ""
    ses: str = ""
    n_sensors: int = 4
    total_runs: int = 0
    dry_run: bool = False


@dataclass
class TestEvent:
    """Emitted for each pre-collection test."""

    __test__ = False  # prevent pytest from collecting this dataclass
    type: str = "test"
    name: str = ""
    passed: bool = False
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestInstructionEvent:
    """Emitted before a test starts — tells the UI what to show the operator."""

    __test__ = False
    type: str = "test_instruction"
    name: str = ""          # machine name (e.g. "channel_activation")
    label: str = ""         # human label (e.g. "Channel Activation")
    instruction: str = ""   # what the subject should do
    duration_s: float = 0.0
    test_index: int = 0     # 0-based index of this test
    total_tests: int = 8


@dataclass
class TestReadyEvent:
    """Emitted when the backend is waiting for the operator to press 'Ready'."""

    __test__ = False
    type: str = "test_ready"
    name: str = ""


@dataclass
class TestCountdownEvent:
    """Emitted during the 3-2-1 countdown before sampling starts."""

    __test__ = False
    type: str = "test_countdown"
    name: str = ""
    countdown: int = 3  # 3, 2, 1


@dataclass
class TestRunningEvent:
    """Emitted when sampling starts ('GO')."""

    __test__ = False
    type: str = "test_running"
    name: str = ""
    elapsed_s: float = 0.0
    duration_s: float = 0.0


@dataclass
class CollectionReadyEvent:
    """Emitted after tests complete — backend waits for 'begin_collection'."""

    type: str = "collection_ready"
    total_runs: int = 0
    tests_passed: int = 0
    tests_total: int = 0
    run_summary: list[dict] | None = None  # [{task, run, rep, phase, duration_s}, ...]


@dataclass
class MockModeEvent:
    """Emitted at session start to indicate which streams are using mock data."""

    type: str = "mock_mode"
    sensor_mock: bool = False
    camera_mock: bool = False


@dataclass
class StateEvent:
    """Emitted when the session state changes (PREP, RECORD, REST, etc.)."""

    type: str = "state"
    phase: str = ""  # PREP, RECORD, REST, DONE
    run: int = 0
    task: str = ""           # machine name (e.g. "thumbCmcIso")
    display_name: str = ""   # human-readable name (e.g. "Thumb CMC Flexion")
    description: str = ""    # short description of the movement
    instruction: str = ""    # what the patient should do
    rep: int = 1
    remaining_s: float = 0.0
    elapsed_session_s: float = 0.0


@dataclass
class FsrEvent:
    """Emitted (throttled) for UI FSR plots."""

    type: str = "fsr"
    t_ns: int = 0
    sample_idx: int = 0
    values: list[int] | None = None


@dataclass
class CameraEvent:
    """Emitted (throttled) for UI camera preview metadata."""

    type: str = "camera"
    t_ns: int = 0
    valid: bool = False
    confidence: float = 0.0


@dataclass
class QualityEvent:
    """Emitted when data quality changes."""

    type: str = "quality"
    level: str = "green"  # green | yellow | red
    reason: str = ""
    per_sensor: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunCompleteEvent:
    """Emitted when a single run finishes."""

    type: str = "run_complete"
    task: str = ""
    run: int = 0
    rep: int = 1
    physio_samples: int = 0
    camera_frames: int = 0
    duration_s: float = 0.0


@dataclass
class SummaryEvent:
    """Emitted once at the end of the session."""

    type: str = "summary"
    completed_runs: int = 0
    total_runs: int = 0
    total_physio_samples: int = 0
    total_camera_frames: int = 0
    quarantined_runs: int = 0
    session_dir: Path | None = None


SessionEvent = (
    SetupEvent | TestEvent | TestInstructionEvent | TestReadyEvent
    | TestCountdownEvent | TestRunningEvent | CollectionReadyEvent
    | MockModeEvent | StateEvent | FsrEvent | CameraEvent | QualityEvent
    | RunCompleteEvent | SummaryEvent
)


# ── Sensor / camera reader protocol (duck typing) ─────────────────────────────


class _SensorReader(Protocol):
    def start(self) -> bool:
        ...

    def stop(self) -> None:
        ...

    def read(self) -> Any:
        ...


class _CameraReader(Protocol):
    def start(self) -> bool:
        ...

    def stop(self) -> None:
        ...

    def get_frame(self) -> dict[str, Any]:
        ...


# ── Session generator ─────────────────────────────────────────────────────────


@dataclass
class _PhaseState:
    """Mutable sampling state shared across PREP / RECORD / REST phases."""
    sample_idx: int = 0
    camera_frame: int = 0
    last_cam_ts_ns: int = 0
    last_landmarks: list[float] | None = None
    cam_data: dict[str, Any] | None = None
    next_ui_fsr_s: float = 0.0
    next_ui_camera_s: float = 0.0
    next_sample_ns: int = 0
    physio_count: int = 0
    camera_count: int = 0


def _sample_phase(
    state: _PhaseState,
    writer: BIDSRunWriter,
    sensor_reader: _SensorReader,
    camera_reader: _CameraReader | None,
    quality_monitor: QualityMonitor,
    run_spec: RunSpec,
    actual_run: int,
    phase: str,
    cue_event: str,
    duration_s: float,
    session_start_s: float,
    stop_event: threading.Event | None,
    dry_run: bool,
    n_sensors: int,
    state_throttle_s: float,
    quality_active: bool,
    cue_display_name: str = "",
    cue_description: str = "",
    cue_instruction: str = "",
) -> Iterator[SessionEvent]:
    """Sample sensors for ``duration_s`` and write rows tagged with ``phase``.

    The first sample of the phase gets ``cue_event`` as its event marker.
    Quality monitoring runs only when ``quality_active`` is True (RECORD only).
    Mutates ``state`` in place so counters persist across phases.
    """
    sample_period_ns = 10_000_000  # 100 Hz = 10 ms
    ui_throttle_fsr_s = 0.02  # 50 Hz UI updates
    ui_throttle_camera_s = 0.033  # ~30 Hz UI updates
    phase_start_s = time.monotonic()
    next_state_s = phase_start_s
    first_sample = True

    while time.monotonic() - phase_start_s < duration_s:
        if stop_event is not None and stop_event.is_set():
            yield QualityEvent(level="red", reason="Session stopped by user")
            return
        now_s = time.monotonic()
        ts_ns = now_ns()

        # Pace at 100 Hz for dry-run (mock reader returns instantly).
        # For live serial, sensor_reader.read() blocks until a line arrives.
        if dry_run:
            if ts_ns < state.next_sample_ns:
                sleep_s = (state.next_sample_ns - ts_ns) / 1e9
                if sleep_s > 0:
                    time.sleep(sleep_s)
            state.next_sample_ns += sample_period_ns

        sample = sensor_reader.read()
        t_ns = sample.t_ns
        fsr_values = sample.fsr
        led_state = sample.led

        # Ensure FSR values are padded to n_sensors
        while len(fsr_values) < n_sensors:
            fsr_values.append(0)
        fsr_values = fsr_values[:n_sensors]

        # Write physio sample
        writer.write_physio(
            t_monotonic_ns=t_ns,
            sample_idx=state.sample_idx,
            fsr_values=fsr_values,
            phase=phase,
            cue_event=cue_event if first_sample else "",
            led_fsr=led_state,
        )
        state.physio_count += 1
        first_sample = False

        # Camera at ~30 Hz
        cam_dt_ns = t_ns - state.last_cam_ts_ns
        if cam_dt_ns >= 33_333_333 and camera_reader is not None:
            state.cam_data = camera_reader.get_frame()
            cam_ts = state.cam_data.get("t_ns", t_ns)
            writer.write_camera(
                cam_ts_ns=cam_ts,
                landmarks=state.cam_data.get("landmarks"),
                valid=state.cam_data.get("valid", False),
                confidence=state.cam_data.get("confidence", 0.0),
                handedness=state.cam_data.get("handedness", "Right"),
                led_cam=int(state.cam_data.get("led_brightness") or 0),
            )
            state.camera_frame += 1
            state.camera_count += 1
            state.last_cam_ts_ns = t_ns
            if state.cam_data.get("valid") and state.cam_data.get("landmarks"):
                state.last_landmarks = state.cam_data["landmarks"]

            # Throttled camera UI event
            if now_s - state.next_ui_camera_s >= ui_throttle_camera_s:
                yield CameraEvent(
                    t_ns=cam_ts,
                    valid=state.cam_data.get("valid", False),
                    confidence=state.cam_data.get("confidence", 0.0),
                )
                state.next_ui_camera_s = now_s

        # Write targets (joint angles) at 100 Hz using latest landmarks
        if state.last_landmarks:
            joint_angles = landmarks_to_joint_angles(state.last_landmarks)
            writer.write_targets(t_monotonic_ns=t_ns, joint_angles=joint_angles)
        else:
            writer.write_targets(t_monotonic_ns=t_ns, joint_angles=[0.0] * 15)

        # Quality check (RECORD phase only)
        if quality_active:
            quality_update = quality_monitor.update(
                t_ns=t_ns,
                fsr_values=fsr_values,
                led_state=led_state,
                camera_valid=state.cam_data.get("valid", False) if state.cam_data else False,
                camera_confidence=state.cam_data.get("confidence", 0.0) if state.cam_data else 0.0,
            )
            if quality_update:
                level, reason, per_sensor = quality_update
                yield QualityEvent(level=level, reason=reason, per_sensor=per_sensor)

        # Throttled FSR UI event
        if now_s - state.next_ui_fsr_s >= ui_throttle_fsr_s:
            yield FsrEvent(t_ns=t_ns, sample_idx=state.sample_idx, values=list(fsr_values))
            state.next_ui_fsr_s = now_s

        # Update progress state every 1 s
        remaining = duration_s - (now_s - phase_start_s)
        if now_s - next_state_s >= state_throttle_s:
            yield StateEvent(
                phase=phase,
                run=actual_run,
                task=run_spec.task,
                display_name=cue_display_name,
                description=cue_description,
                instruction=cue_instruction,
                rep=run_spec.rep,
                remaining_s=max(0.0, remaining),
                elapsed_session_s=now_s - session_start_s,
            )
            next_state_s = now_s

        state.sample_idx += 1


def run_session(
    sub: str,
    ses: str,
    sensor_reader: _SensorReader,
    camera_reader: _CameraReader,
    runs: list[RunSpec],
    *,
    n_sensors: int = 4,
    skip_precollect: bool = False,
    dry_run: bool = False,
    data_root: Path | None = None,
    stop_event: threading.Event | None = None,
    ready_event: threading.Event | None = None,
    begin_event: threading.Event | None = None,
    retry_event: threading.Event | None = None,
    retry_test_name: str | None = None,
    get_retry_test_name: Callable[[], str | None] | None = None,
    broadcast_fn: Callable[[dict], None] | None = None,
    precollect_countdown_s: float = 1.0,
    quality_config: Any | None = None,
    duration_overrides: dict[str, float] | None = None,
) -> Iterator[SessionEvent]:
    """Run a full collection session as a generator of structured events.

    This function writes BIDS CSVs to disk in real time while yielding events
    that can be consumed by a UI, a CLI printer, or a test harness.

    If ``stop_event`` is provided, it is checked periodically and the session
    is terminated early when the event is set.

    Parameters
    ----------
    sub, ses : str
        Subject and session IDs.
    sensor_reader : object with ``start()``, ``stop()``, ``read()``
        Returns a sample with ``fsr`` (list[int]), ``led`` (int), and ``t_ns`` (int).
    camera_reader : object with ``start()``, ``stop()``, ``get_frame()``
        Returns a dict with ``t_ns``, ``valid``, ``confidence``, ``landmarks``.
    runs : list[RunSpec]
        Protocol runs to execute.
    n_sensors : int
        Number of FSR channels.
    skip_precollect : bool
        Skip the mandatory pre-collection tests.
    dry_run : bool
        Whether this is a hardware-free dry run.
    data_root : Path | None
        Root data directory (defaults to ``core.paths.RAW_DIR``).

    Yields
    ------
    SessionEvent
        One of the event dataclasses defined above.
    """
    root = data_root or paths.RAW_DIR
    session_start_s = time.monotonic()

    # Common timing constants
    state_throttle_s = 1.0  # 1 Hz state updates

    # ── Setup event ───────────────────────────────────────────────────────────
    yield SetupEvent(
        sub=sub, ses=ses, n_sensors=n_sensors,
        total_runs=len(runs), dry_run=dry_run,
    )

    # ── Mock mode event ──────────────────────────────────────────────────────
    # Tell the UI which streams are using mock data so it can show a banner.
    from apps.collection.sensor_reader import MockSensorReader
    from apps.collection.camera import MockCameraReader
    sensor_mock = isinstance(sensor_reader, MockSensorReader)
    camera_mock = isinstance(camera_reader, MockCameraReader)
    yield MockModeEvent(sensor_mock=sensor_mock, camera_mock=camera_mock)

    # ── Dataset-level BIDS metadata ─────────────────────────────────────────
    write_dataset_description(data_root=root)

    # ── Session metadata ────────────────────────────────────────────────────
    # Note: sampling_frequency is updated to 100 Hz to match the Arduino firmware.
    # Derive the chosen protocol config (n_reps, include_freeform) from the
    # runs list so it is persisted to physio.json and the post-session sweep
    # can check completeness against the CHOSEN protocol, not the 76-run default.
    _chosen_n_reps = max({r.rep for r in runs}, default=1)
    _chosen_include_freeform = any(r.phase == "freeform" for r in runs)
    write_session_metadata(
        sub, ses,
        sampling_frequency=100.0,
        sensor_count=n_sensors,
        camera_fps=30.0,
        n_reps=_chosen_n_reps,
        include_freeform=_chosen_include_freeform,
        data_root=root,
    )
    append_participants_tsv(sub, data_root=root)
    append_sessions_tsv(sub, ses, sensor_count=n_sensors, sampling_frequency_hz=100.0, data_root=root)

    # ── Sweep orphan/partial runs from a previous crashed session ──────────
    # If the previous session was killed (SIGKILL/power loss), CSVs may exist
    # without a manifest. Quarantine them so the session dir is clean and
    # next_run_number doesn't count orphan files.
    try:
        swept = sweep_orphan_partials(sub, ses, data_root=root)
        if swept > 0:
            yield QualityEvent(
                level="yellow",
                reason=f"Swept {swept} orphan/partial run(s) from previous session",
            )
    except Exception as exc:
        logger.warning("Orphan sweep failed: %s", exc)

    # ── Pre-collection tests ───────────────────────────────────────────────
    if not skip_precollect:
        test_results = PrecollectResults()
        try:
            # Signal that the session thread is now the queue consumer so the
            # reader thread stops discarding samples silently.
            if hasattr(sensor_reader, 'attach_consumer'):
                sensor_reader.attach_consumer()
            # Flush any stale data in the serial buffer before starting tests
            if hasattr(sensor_reader, 'flush_buffer'):
                sensor_reader.flush_buffer()
                logger.info("Flushed serial buffer before pre-collection tests")
            logger.info("Starting pre-collection tests")
            # If no camera is available, pass None — camera tests will fail
            # with a clear message instead of crashing
            camera_callback = camera_reader.get_frame if camera_reader is not None else None
            # Provide a non-blocking preview reader for FSR bar display while
            # the operator reads the instruction and waits to press Ready.
            # read_sensors() blocks on queue.get() which stalls the generator;
            # read_sensors_preview() returns the latest atomic sample instantly.
            _preview = getattr(sensor_reader, 'read_sensors_preview', None)
            for event in run_all_tests_interactive(
                sensor_reader.read_sensors,
                camera_callback,
                read_sensors_preview=_preview,
                ready_event=ready_event,
                stop_event=stop_event,
                broadcast_fn=broadcast_fn,
                countdown_s=precollect_countdown_s,
                duration_overrides=duration_overrides,
            ):
                if isinstance(event, TestResult):
                    test_results.results.append(event)
                    logger.info("Test %s: %s — %s", event.name, "PASS" if event.passed else "FAIL", event.message)
                    yield TestEvent(name=event.name, passed=event.passed, message=event.message, details=event.details)
                elif isinstance(event, dict):
                    # Forward instruction/ready/countdown/running events to the UI
                    yield event
                if stop_event is not None and stop_event.is_set():
                    logger.info("Pre-collection tests stopped by operator")
                    break
        except Exception as exc:
            logger.error("Pre-collection tests crashed: %s", exc, exc_info=True)
            yield QualityEvent(level="red", reason=f"Pre-collection tests crashed: {exc}")
        try:
            test_results.save_to_bids(sub, ses, data_root=root)
        except Exception as exc:
            yield QualityEvent(level="yellow", reason=f"Pre-collection metadata not saved: {exc}")
        yield QualityEvent(
            level="green" if test_results.all_passed else "yellow",
            reason=f"Pre-collection tests: {test_results.n_passed}/{test_results.n_passed + test_results.n_failed} passed",
        )
        # Surface advisory warnings from warning-only tests (e.g. response_linearity)
        warnings = [
            f"{r.name}: {r.message}"
            for r in test_results.results
            if r.details.get("warning")
        ]
        if warnings:
            yield QualityEvent(
                level="yellow",
                reason="Advisory warnings — " + "; ".join(warnings),
            )
    else:
        # skip_precollect path: attach consumer here since tests were skipped
        if hasattr(sensor_reader, 'attach_consumer'):
            sensor_reader.attach_consumer()
        yield QualityEvent(level="green", reason="Pre-collection tests skipped")

    # ── Collection gate ─────────────────────────────────────────────────────
    # Wait for the operator to press "Begin Collection" before starting run 1.
    # Also supports "Retry Failed Tests" — if retry_event is set, re-run failed tests.
    if begin_event is not None:
        _emit_collection_ready = True  # False while retry tests are running
        while True:
            if _emit_collection_ready:
                run_summary = [
                    {"task": r.task, "run": r.run, "rep": r.rep, "phase": r.phase, "duration_s": r.record_duration}
                    for r in runs
                ]
                yield CollectionReadyEvent(
                    total_runs=len(runs),
                    tests_passed=test_results.n_passed if not skip_precollect else 0,
                    tests_total=len(test_results.results) if not skip_precollect else 0,
                    run_summary=run_summary,
                )

            # Wait for operator to press "Begin Collection" or "Retry",
            # with a 5-minute timeout.  Poll every 200 ms so retry_event
            # is also noticed quickly.
            begin_wait_start = time.monotonic()
            begin_timeout_s = 300.0  # 5 minutes
            while not begin_event.is_set() and (retry_event is None or not retry_event.is_set()):
                if stop_event is not None and stop_event.is_set():
                    if hasattr(sensor_reader, 'detach_consumer'):
                        sensor_reader.detach_consumer()
                    yield QualityEvent(level="red", reason="Session stopped by user")
                    return
                if time.monotonic() - begin_wait_start > begin_timeout_s:
                    yield QualityEvent(
                        level="red",
                        reason="Collection gate timed out after 5 minutes of inactivity",
                    )
                    return
                begin_event.wait(timeout=0.2)

            # Check if retry was pressed
            if retry_event is not None and retry_event.is_set():
                retry_event.clear()
                # Determine which tests to retry.  Read the current test name
                # via getter (set at the time the user pressed Retry) so we
                # see the live value, not the snapshot from session start.
                current_retry_name = (
                    get_retry_test_name() if get_retry_test_name is not None
                    else retry_test_name
                )
                failed_names = [r.name for r in test_results.results if not r.passed]
                if current_retry_name is not None and current_retry_name in failed_names:
                    target_names = [current_retry_name]
                else:
                    target_names = failed_names
                if not target_names:
                    yield QualityEvent(level="green", reason="No failed tests to retry")
                    _emit_collection_ready = True
                    continue
                logger.info("Retrying tests: %s", target_names)
                # Signal the UI to go back to the tests screen before emitting
                # test_instruction events — otherwise they land on the hidden screen.
                yield {"type": "retry_started", "tests": target_names}
                yield QualityEvent(level="yellow", reason=f"Retrying {len(target_names)} test(s)...")
                # Remove the target tests from results so they can be re-added
                target_set = set(target_names)
                test_results.results = [r for r in test_results.results if r.name not in target_set or r.passed]
                camera_callback = camera_reader.get_frame if camera_reader is not None else None
                _preview = getattr(sensor_reader, 'read_sensors_preview', None)
                for target_name in target_names:
                    for event in run_all_tests_interactive(
                        sensor_reader.read_sensors,
                        camera_callback,
                        read_sensors_preview=_preview,
                        ready_event=ready_event,
                        stop_event=stop_event,
                        retry_test=target_name,
                        broadcast_fn=broadcast_fn,
                        duration_overrides=duration_overrides,
                    ):
                        if isinstance(event, TestResult):
                            test_results.results.append(event)
                            logger.info("Retry test %s: %s — %s", event.name, "PASS" if event.passed else "FAIL", event.message)
                            yield TestEvent(name=event.name, passed=event.passed, message=event.message, details=event.details)
                        elif isinstance(event, dict):
                            yield event
                        if stop_event is not None and stop_event.is_set():
                            break
                # Save updated results
                try:
                    test_results.save_to_bids(sub, ses, data_root=root)
                except Exception:
                    pass
                yield QualityEvent(
                    level="green" if test_results.all_passed else "yellow",
                    reason=f"Tests after retry: {test_results.n_passed}/{test_results.n_passed + test_results.n_failed} passed",
                )
                # Loop back to collection gate — emit collection_ready again
                # so the UI re-renders the begin gate with updated pass counts.
                _emit_collection_ready = True
                continue
            # begin_event was set — proceed to collection
            break

    # ── Run collection ──────────────────────────────────────────────────────
    completed_runs = 0
    quarantined_runs = 0
    total_physio = 0
    total_camera = 0

    for i, run_spec in enumerate(runs):
        elapsed_s = time.monotonic() - session_start_s
        cue = get_cue(run_spec.task)
        display_text = format_cue_for_display(cue, run_spec.run, run_spec.rep, 3)

        yield StateEvent(
            phase="PREP",
            run=run_spec.run,
            task=run_spec.task,
            display_name=cue.display_name,
            description=cue.description,
            instruction=cue.instruction,
            rep=run_spec.rep,
            remaining_s=run_spec.prep_duration,
            elapsed_session_s=elapsed_s,
        )

        # Open writer — use the protocol run number, but if that run already
        # exists (crash-recovery re-run), quarantine the existing partial first
        # so the run number is freed, then reuse it. Completed runs (with a
        # valid manifest) are NOT quarantined — they are preserved and the
        # writer auto-increments to the next free run number instead.
        actual_run = run_spec.run
        writer = BIDSRunWriter(
            sub=sub, ses=ses,
            task=run_spec.task, run=actual_run,
            n_sensors=n_sensors,
            data_root=root,
        )
        try:
            writer.open()
        except FileExistsError:
            from apps.collection.bids_writer import manifest_path as _mp
            mpath = _mp(sub, ses, run_spec.task, actual_run, data_root=root)
            is_complete = False
            if mpath.exists():
                try:
                    import json as _json
                    with open(mpath, encoding="utf-8") as f:
                        m = _json.load(f)
                    is_complete = m.get("complete", False)
                except Exception:
                    pass
            if is_complete:
                # Completed run — don't touch it, auto-increment instead.
                actual_run = naming.next_run_number(sub, ses, run_spec.task, data_root=root)
                logger.warning(
                    "Run %02d for %s already completed — re-running as run %02d",
                    run_spec.run, run_spec.task, actual_run,
                )
            else:
                # Partial run (crash or previous abort) — quarantine it and
                # reuse the same run number.
                logger.warning(
                    "Run %02d for %s has partial data — quarantining and reusing run number",
                    run_spec.run, run_spec.task,
                )
                quarantine_partial_run(
                    sub, ses, run_spec.task, actual_run, data_root=root
                )
            writer = BIDSRunWriter(
                sub=sub, ses=ses,
                task=run_spec.task, run=actual_run,
                n_sensors=n_sensors,
                data_root=root,
            )
            writer.open()

        try:
            # ── Sampling state shared across all phases ──────────────────────
            pstate = _PhaseState(next_sample_ns=now_ns())
            quality_monitor = QualityMonitor(
                n_sensors=n_sensors, target_hz=100.0, quality_config=quality_config
            )

            # ── PREP phase: sample + write with phase="PREP" ─────────────────
            yield StateEvent(
                phase="PREP", run=actual_run, task=run_spec.task, rep=run_spec.rep,
                display_name=cue.display_name,
                description=cue.description,
                instruction=cue.instruction,
                remaining_s=run_spec.prep_duration,
                elapsed_session_s=time.monotonic() - session_start_s,
            )
            yield from _sample_phase(
                pstate, writer, sensor_reader, camera_reader, quality_monitor,
                run_spec, actual_run, "PREP", "PREP_START",
                run_spec.prep_duration, session_start_s, stop_event,
                dry_run, n_sensors, state_throttle_s, quality_active=False,
                cue_display_name=cue.display_name,
                cue_description=cue.description,
                cue_instruction=cue.instruction,
            )

            if stop_event is not None and stop_event.is_set():
                writer.close()
                break

            # Flush stale serial data before RECORD so first sample is fresh.
            if hasattr(sensor_reader, "flush_buffer"):
                sensor_reader.flush_buffer()

            # Arm the PRBS preamble on the FIRST run only, immediately before
            # RECORD starts.  The firmware sits idle (LED OFF) until it
            # receives 'S', then begins the 6.3 s PRBS preamble that
            # led_sync.py Stage 1 uses for unambiguous coarse acquisition.
            # See ADR 003 for the rationale.
            if i == 0 and hasattr(sensor_reader, "trigger_sync"):
                sensor_reader.trigger_sync()

            # ── RECORD phase: sample + write with phase="RECORD" ─────────────
            yield StateEvent(
                phase="RECORD", run=actual_run, task=run_spec.task, rep=run_spec.rep,
                display_name=cue.display_name,
                description=cue.description,
                instruction=cue.instruction,
                remaining_s=run_spec.record_duration,
                elapsed_session_s=time.monotonic() - session_start_s,
            )
            yield from _sample_phase(
                pstate, writer, sensor_reader, camera_reader, quality_monitor,
                run_spec, actual_run, "RECORD", "RECORD_START",
                run_spec.record_duration, session_start_s, stop_event,
                dry_run, n_sensors, state_throttle_s, quality_active=True,
                cue_display_name=cue.display_name,
                cue_description=cue.description,
                cue_instruction=cue.instruction,
            )

            if stop_event is not None and stop_event.is_set():
                writer.close(aborted=True)
                quarantine_partial_run(
                    sub, ses, run_spec.task, actual_run, data_root=root
                )
                quarantined_runs += 1
                yield {"type": "run_skipped", "task": run_spec.task, "run": actual_run, "reason": "stopped"}
                break

            # ── REST phase: sample + write with phase="REST" ─────────────────
            yield StateEvent(
                phase="REST", run=actual_run, task=run_spec.task, rep=run_spec.rep,
                display_name=cue.display_name,
                description=cue.description,
                instruction="Relax your hand and rest.",
                remaining_s=run_spec.rest_duration,
                elapsed_session_s=time.monotonic() - session_start_s,
            )
            yield from _sample_phase(
                pstate, writer, sensor_reader, camera_reader, quality_monitor,
                run_spec, actual_run, "REST", "REST_START",
                run_spec.rest_duration, session_start_s, stop_event,
                dry_run, n_sensors, state_throttle_s, quality_active=False,
                cue_display_name=cue.display_name,
                cue_description=cue.description,
                cue_instruction="Relax your hand and rest.",
            )

            if stop_event is not None and stop_event.is_set():
                writer.close(aborted=True)
                quarantine_partial_run(
                    sub, ses, run_spec.task, actual_run, data_root=root
                )
                quarantined_runs += 1
                yield {"type": "run_skipped", "task": run_spec.task, "run": actual_run, "reason": "stopped"}
                break

            result = writer.close()
            completed_runs += 1
            total_physio += result["physio_rows"]
            total_camera += result["camera_rows"]

            yield RunCompleteEvent(
                task=run_spec.task,
                run=actual_run,
                rep=run_spec.rep,
                physio_samples=pstate.physio_count,
                camera_frames=pstate.camera_count,
                duration_s=run_spec.record_duration,
            )

        except KeyboardInterrupt:
            writer.close(aborted=True)
            quarantine_partial_run(
                sub, ses, run_spec.task, actual_run, data_root=root
            )
            quarantined_runs += 1
            yield {"type": "run_skipped", "task": run_spec.task, "run": actual_run, "reason": "interrupted"}
            yield QualityEvent(level="red", reason="Session interrupted by user")
            break
        finally:
            if writer.is_open:
                writer.close(aborted=True)
                quarantine_partial_run(
                    sub, ses, run_spec.task, actual_run, data_root=root
                )
                quarantined_runs += 1
                yield {"type": "run_skipped", "task": run_spec.task, "run": actual_run, "reason": "crashed"}

    # Compute MVC calibration from the baseline MVC run and update physio.json
    try:
        update_mvc_calibration(sub, ses, data_root=root)
    except Exception as exc:
        logger.warning("MVC calibration update failed: %s", exc)

    # Run offline LED sync and write led_sync.json
    try:
        from apps.collection.led_sync import write_led_sync
        write_led_sync(sub, ses, data_root=root)
    except Exception as exc:
        logger.warning("LED sync failed: %s", exc)

    # Release the queue consumer flag so the preview reader resumes
    # non-blocking mode (in case the user returns to the Setup screen).
    if hasattr(sensor_reader, 'detach_consumer'):
        sensor_reader.detach_consumer()

    yield SummaryEvent(
        completed_runs=completed_runs,
        total_runs=len(runs),
        total_physio_samples=total_physio,
        total_camera_frames=total_camera,
        quarantined_runs=quarantined_runs,
        session_dir=paths.session_dir(sub, ses, data_root=root),
    )
