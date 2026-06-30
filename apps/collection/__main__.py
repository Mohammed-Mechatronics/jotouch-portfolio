"""Entry point for the JoTouch BIDS collection app.

Runs the full 3-phase regression protocol with pre-collection tests.

Usage:
    python -m apps.collection --sub P01 --ses S01
    python -m apps.collection --sub P01 --ses S01 --port COM3 --baud 115200
    python -m apps.collection --sub P01 --ses S01 --skip-precollect
    python -m apps.collection --sub P01 --ses S01 --dry-run  (no hardware)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.collection.camera import create_camera_reader
from apps.collection.config import QualityConfig, TestDurationConfig
from apps.collection.protocol import build_protocol, protocol_summary
from apps.collection.sensor_reader import create_sensor_reader
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


def _format_event(event) -> str | None:
    """Format a session event for terminal output. Returns None if no output."""
    if isinstance(event, SetupEvent):
        lines = [
            "JoTouch BIDS Collection App",
            "=" * 60,
            f"Subject: {event.sub}",
            f"Session: {event.ses}",
            f"Sensors: {event.n_sensors}",
            f"Mode: {'DRY RUN (mock)' if event.dry_run else 'LIVE (hardware)'}",
        ]
        return "\n".join(lines)

    if isinstance(event, StateEvent):
        if event.phase == "PREP":
            return f"  {event.task} (run {event.run:02d}, rep {event.rep}) - PREP {event.remaining_s:.1f}s"
        if event.phase == "RECORD":
            return f"  {event.task} (run {event.run:02d}, rep {event.rep}) - RECORD {event.remaining_s:.1f}s"
        if event.phase == "REST":
            return f"  REST ({event.remaining_s:.0f}s)..."
        return None

    if isinstance(event, TestEvent):
        status = "PASS" if event.passed else "FAIL"
        return f"  [{status}] {event.name}: {event.message}"

    if isinstance(event, QualityEvent):
        prefix = {"green": "[OK]", "yellow": "[WARN]", "red": "[ERROR]"}.get(event.level, "[INFO]")
        return f"  {prefix} {event.reason}"

    if isinstance(event, RunCompleteEvent):
        return (
            f"  Saved: {event.physio_samples} physio, "
            f"{event.camera_frames} camera for {event.task} run {event.run:02d}"
        )

    if isinstance(event, SummaryEvent):
        lines = [
            "=" * 60,
            f"Collection complete: {event.completed_runs}/{event.total_runs} runs completed",
            f"Total samples: {event.total_physio_samples} physio, {event.total_camera_frames} camera",
            f"Data saved to: {event.session_dir}",
        ]
        return "\n".join(lines)

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="JoTouch BIDS collection app")
    parser.add_argument("--sub", type=str, default="P01", help="Subject ID (e.g. P01)")
    parser.add_argument("--ses", type=str, default="S01", help="Session ID (e.g. S01)")
    parser.add_argument("--port", type=str, default=None, help="Serial port (e.g. COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--n-sensors", type=int, default=4, help="Number of FSR sensors")
    parser.add_argument("--skip-precollect", action="store_true", help="Skip pre-collection tests")
    parser.add_argument("--dry-run", action="store_true", help="No hardware (mock sensors + camera)")
    parser.add_argument("--allow-mock-camera", action="store_true",
                        help="If a real camera cannot be opened, fall back to the mock camera reader instead of failing")
    parser.add_argument("--phase", type=str, default=None,
                        help="Run only one phase (baseline, single_dof, multi_dof, freeform)")
    parser.add_argument("--task", type=str, default=None,
                        help="Run only one task (e.g. powerGrip)")
    parser.add_argument("--quality-config", type=str, default=None,
                        help="JSON file with QualityMonitor thresholds (e.g. {'fsr_red_ratio': 0.4})")
    parser.add_argument("--test-config", type=str, default=None,
                        help="JSON file with pre-collection test durations in seconds")
    args = parser.parse_args()

    # Load optional quality/test config files
    quality_config = None
    if args.quality_config:
        try:
            with open(args.quality_config, encoding="utf-8") as f:
                quality_config = QualityConfig.from_dict(json.load(f))
        except Exception as exc:
            print(f"ERROR: Could not load quality config: {exc}")
            return 1

    duration_overrides = None
    if args.test_config:
        try:
            with open(args.test_config, encoding="utf-8") as f:
                duration_overrides = TestDurationConfig.from_dict(json.load(f)).__dict__
        except Exception as exc:
            print(f"ERROR: Could not load test config: {exc}")
            return 1

    # Build protocol
    runs = build_protocol()
    if args.phase:
        runs = [r for r in runs if r.phase == args.phase]
    if args.task:
        runs = [r for r in runs if r.task == args.task]

    summary = protocol_summary(runs)

    # Set up sensor/camera readers
    sensor_reader = create_sensor_reader(
        port=args.port,
        baud=args.baud,
        n_sensors=args.n_sensors,
        dry_run=args.dry_run,
    )
    camera_reader = create_camera_reader(dry_run=True) if args.dry_run else create_camera_reader(
        dry_run=False, camera_index=0
    )

    sensor_reader.start()
    camera_started = camera_reader.start()
    if not camera_started and not args.dry_run:
        if args.allow_mock_camera:
            print("WARNING: Camera not available. Using mock camera reader (--allow-mock-camera).")
            camera_reader = create_camera_reader(dry_run=True)
            camera_reader.start()
        else:
            print("ERROR: Camera not available. Connect a webcam, enable --dry-run, or pass --allow-mock-camera.")
            sensor_reader.stop()
            return 1

    # Print header and protocol summary before the session starts
    print(f"Protocol: {summary['total_runs']} runs, ~{summary['total_duration_s']:.0f}s total")
    for phase, data in summary["phases"].items():
        print(f"  {phase}: {data['n_tasks']} tasks, {data['n_runs']} runs, {data['duration_s']:.0f}s")
    print()

    if not args.skip_precollect:
        print("\n--- Pre-collection Tests ---")

    if not args.dry_run and not args.port:
        print("ERROR: --port required for live mode (or use --dry-run)")
        return 1

    exit_code = 0
    try:
        for event in run_session(
            sub=args.sub,
            ses=args.ses,
            sensor_reader=sensor_reader,
            camera_reader=camera_reader,
            runs=runs,
            n_sensors=args.n_sensors,
            skip_precollect=args.skip_precollect,
            dry_run=args.dry_run,
            quality_config=quality_config,
            duration_overrides=duration_overrides,
        ):
            output = _format_event(event)
            if output is not None:
                print(output)
    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user.")
        exit_code = 1
    finally:
        sensor_reader.stop()
        camera_reader.stop()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
