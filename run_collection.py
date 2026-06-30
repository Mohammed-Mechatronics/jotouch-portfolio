"""Launch the JoTouch BIDS collection application.

The collection app runs the 3-phase regression protocol:
  Phase 1: Single-DOF isolation (15 tasks × 3 reps)
  Phase 2: Multi-DOF combinations (9 tasks × 3 reps)
  Phase 3: Freeform (3 × 60s runs)

Data is written as BIDS CSVs to ``data/raw/sub-PXX/ses-SXX/``:
  - ``_physio.csv``  (FSR signals)
  - ``_camera.csv``  (MediaPipe hand landmarks)
  - ``_targets.csv`` (derived 15 joint angles)

Usage:
    python run_collection.py
    python run_collection.py --port COM3 --baud 115200
    python run_collection.py --sub P01 --ses S01
    python run_collection.py --dry-run          # mock hardware, no camera/serial
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="JoTouch BIDS collection app")
    parser.add_argument("--port", type=str, default=None, help="Serial port (e.g. COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--sub", type=str, default="P01", help="Subject ID (e.g. P01)")
    parser.add_argument("--ses", type=str, default="S01", help="Session ID (e.g. S01)")
    args = parser.parse_args()

    print("JoTouch Collection App")
    print("=" * 60)
    print(f"Subject: {args.sub}, Session: {args.ses}")
    print()

    bids_app = Path(__file__).parent / "apps" / "collection" / "__main__.py"
    if not bids_app.exists():
        print(f"BIDS collection app not found at {bids_app}")
        return 1

    print("Starting BIDS collection app...")
    cmd = [sys.executable, str(bids_app), "--sub", args.sub, "--ses", args.ses]
    if args.port:
        cmd += ["--port", args.port, "--baud", str(args.baud)]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
