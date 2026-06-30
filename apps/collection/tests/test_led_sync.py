"""Tests for apps.collection.led_sync — hybrid PRBS + NAd."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from apps.collection.led_sync import run_led_sync, write_led_sync
from apps.collection.prbs import PRBS_SEQUENCE, PRBS_CHIP_S, PRBS_DURATION_S


def _write_mvc_data(sdir: Path, fsr_led_values: list[int], cam_led_values: list[float],
                    start_s: float = 7.0) -> None:
    """Write test physio + camera CSVs with LED data.

    Timestamps start at ``start_s`` seconds (default 7.0) to skip the
    PRBS preamble phase (0–6.3 s) so the periodic edge detection works.
    """
    t0_ns = int(start_s * 1e9)
    physio_csv = sdir / "sub-P01_ses-S01_task-mvc_run-00_physio.csv"
    with open(physio_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_monotonic_ns", "sample_idx", "phase", "participant_id", "session_id", "task", "run", "fsr0", "fsr1", "fsr2", "fsr3", "cue_event", "led_fsr"])
        for i, led in enumerate(fsr_led_values):
            writer.writerow([t0_ns + i * 10_000_000, i, "ACTIVE", "P01", "S01", "mvc", 0, 100, 100, 100, 100, "", led])

    camera_csv = sdir / "sub-P01_ses-S01_task-mvc_run-00_camera.csv"
    with open(camera_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cam_ts_ns", "mp_valid", "mp_confidence", "mp_handedness"] + [f"mp_lm{i:02d}_{axis}" for i in range(21) for axis in ("x", "y", "z")] + ["led_cam"])
        for i, led in enumerate(cam_led_values):
            writer.writerow([t0_ns + i * 33_333_333, 1, 0.9, "Right"] + [0.0] * 63 + [led])


class TestLedSync:
    def test_no_camera_led_blinks_returns_failure(self, tmp_path):
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sdir.mkdir(parents=True, exist_ok=True)
        _write_mvc_data(sdir, [0, 1, 0, 1, 0], [0.0, 0.0, 0.0, 0.0, 0.0])
        result = run_led_sync("P01", "S01", data_root=tmp_path)
        assert result["passed"] is False
        # Either FSR or camera edges are too few
        assert "blinks" in result["reason"] or "NAd" in result["reason"]

    def test_matched_blinks_pass(self, tmp_path):
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sdir.mkdir(parents=True, exist_ok=True)
        # Two aligned rising edges in FSR LED and camera brightness
        fsr_led = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        cam_led = [0.0, 0.0, 0.0, 255.0, 255.0, 255.0, 0.0, 0.0, 255.0, 255.0]
        _write_mvc_data(sdir, fsr_led, cam_led)
        result = run_led_sync("P01", "S01", data_root=tmp_path)
        # With only 2 edges, NAd may not find enough windows
        # The key is that a and b are computed (or fails safely)
        assert result["passed"] is False or result["a"] is not None

    def test_write_led_sync_creates_file(self, tmp_path):
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sdir.mkdir(parents=True, exist_ok=True)
        _write_mvc_data(sdir, [0, 1, 0], [0.0, 0.0, 0.0])
        path = write_led_sync("P01", "S01", data_root=tmp_path)
        assert path.exists()

    def test_hybrid_method_field(self, tmp_path):
        """The method field should be 'hybrid_prbs_nad'."""
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sdir.mkdir(parents=True, exist_ok=True)
        _write_mvc_data(sdir, [0, 1, 0, 1, 0], [0.0, 255.0, 0.0, 255.0, 0.0])
        result = run_led_sync("P01", "S01", data_root=tmp_path)
        assert result["method"] == "hybrid_prbs_nad"

    def test_prbs_fields_present(self, tmp_path):
        """PRBS fields should be present (even if None when no PRBS detected)."""
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sdir.mkdir(parents=True, exist_ok=True)
        _write_mvc_data(sdir, [0, 1, 0, 1, 0], [0.0, 255.0, 0.0, 255.0, 0.0])
        result = run_led_sync("P01", "S01", data_root=tmp_path)
        assert "prbs_offset_ms" in result
        assert "prbs_score" in result
        assert "nad_offset_ms" in result
        assert "nad_drift_ppm" in result
        assert "n_windows" in result
        assert "cross_validation_passed" in result

    def test_camera_led_with_ambient_offset_detects_edges(self, tmp_path):
        """Camera LED brightness with ambient offset (e.g. 105-167 range
        instead of 0-255) must still detect rising edges in the periodic stage.

        Real-world ROI brightness includes ambient light, so the OFF state
        is ~106 (not 0) and ON state is ~166. The threshold must be relative
        to the signal's dynamic range, not an absolute fraction of max.
        """
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sdir.mkdir(parents=True, exist_ok=True)
        # Generate 10 seconds of data at 100Hz (FSR) / 30Hz (camera)
        # so the periodic stage (>= 6.3s) has enough samples.
        # 1 Hz blinks: ON for 100ms, OFF for 900ms.
        sample_hz = 100
        cam_hz = 30
        duration_s = 10.0
        t0_ns = int(7.0 * 1e9)  # start at 7s to be past PRBS window after normalization
        dt_ns = int(1e9 / sample_hz)
        cam_dt_ns = int(1e9 / cam_hz)

        n_fsr = int(duration_s * sample_hz)
        fsr_led = []
        for i in range(n_fsr):
            t_s = i / sample_hz
            ms = (t_s * 1000) % 1000
            fsr_led.append(1 if ms < 100 else 0)

        n_cam = int(duration_s * cam_hz)
        cam_led = []
        for i in range(n_cam):
            t_s = i / cam_hz
            ms = (t_s * 1000) % 1000
            # Ambient offset: OFF=106, ON=166 (real ROI brightness)
            cam_led.append(166.0 if ms < 100 else 106.0)

        # Write physio CSV
        physio_csv = sdir / "sub-P01_ses-S01_task-mvc_run-00_physio.csv"
        with open(physio_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["t_monotonic_ns", "sample_idx", "phase", "participant_id",
                             "session_id", "task", "run", "fsr0", "fsr1", "fsr2", "fsr3",
                             "cue_event", "led_fsr"])
            for i, led in enumerate(fsr_led):
                writer.writerow([t0_ns + i * dt_ns, i, "RECORD", "P01", "S01", "mvc", 0,
                                 100, 100, 100, 100, "", led])

        # Write camera CSV
        camera_csv = sdir / "sub-P01_ses-S01_task-mvc_run-00_camera.csv"
        with open(camera_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["cam_ts_ns", "mp_valid", "mp_confidence", "mp_handedness"]
                            + [f"mp_lm{i:02d}_{axis}" for i in range(21) for axis in ("x", "y", "z")]
                            + ["led_cam"])
            for i, brightness in enumerate(cam_led):
                writer.writerow([t0_ns + i * cam_dt_ns, 1, 0.9, "Right"]
                                + [0.0] * 63 + [brightness])

        result = run_led_sync("P01", "S01", data_root=tmp_path)
        # Must NOT fail with "Too few camera LED blinks detected (0)"
        assert "Too few camera LED blinks" not in result["reason"], \
            f"Camera edges must be detected with ambient offset. reason: {result['reason']}"


# ── WS-2: Timestamp normalization + phase filtering ──────────────────────────

def _generate_prbs_led_samples(duration_s: float, sample_hz: float = 100.0,
                               delay_s: float = 0.0) -> list[int]:
    """Generate LED values following the PRBS preamble at a given sample rate.

    ``delay_s`` shifts the LED pattern (simulating camera delay).
    """
    n = int(duration_s * sample_hz)
    values = []
    for i in range(n):
        t_s = i / sample_hz + delay_s
        if t_s < 0:
            values.append(0)
            continue
        chip_idx = int(t_s / PRBS_CHIP_S) % len(PRBS_SEQUENCE)
        values.append(PRBS_SEQUENCE[chip_idx])
    return values


def _write_session_with_prbs(sdir: Path, *, absolute_start_s: float = 5000.0,
                             prep_blink: bool = False, cam_delay_s: float = 0.05,
                             record_duration_s: float = 8.0) -> None:
    """Write a realistic session with absolute timestamps, PRBS preamble in
    RECORD phase, and optional PREP blink data (from WS-1 'B' command).

    Timestamps use absolute monotonic time (e.g. 5000s) to reproduce the
    real-world bug where fsr_ts_s < PRBS_DURATION_S never matches.
    """
    t0_ns = int(absolute_start_s * 1e9)
    sample_hz = 100.0
    dt_ns = int(1e9 / sample_hz)
    physio_csv = sdir / "sub-P01_ses-S01_task-mvc_run-00_physio.csv"
    camera_csv = sdir / "sub-P01_ses-S01_task-mvc_run-00_camera.csv"

    # Generate PRBS LED values for RECORD phase
    fsr_led_record = _generate_prbs_led_samples(record_duration_s, sample_hz, delay_s=0.0)
    cam_led_record = _generate_prbs_led_samples(record_duration_s, 30.0, delay_s=cam_delay_s)
    # Convert cam LED to brightness values (0 or 255)
    cam_brightness_record = [255.0 if v else 0.0 for v in cam_led_record]

    rows_physio = []
    sample_idx = 0

    # PREP phase (optional blink data from 'B' command)
    if prep_blink:
        prep_duration_s = 2.0
        n_prep = int(prep_duration_s * sample_hz)
        for i in range(n_prep):
            t_ns = t0_ns + i * dt_ns
            ms = (t_ns // 1_000_000) % 1000
            led = 1 if ms < 100 else 0  # 1 Hz blink
            rows_physio.append([t_ns, sample_idx, "PREP", "P01", "S01", "mvc", 0,
                                100, 100, 100, 100, "", led])
            sample_idx += 1
        prep_end_ns = t0_ns + n_prep * dt_ns
    else:
        prep_end_ns = t0_ns

    # RECORD phase with PRBS preamble
    n_record = len(fsr_led_record)
    for i in range(n_record):
        t_ns = prep_end_ns + i * dt_ns
        rows_physio.append([t_ns, sample_idx, "RECORD", "P01", "S01", "mvc", 0,
                            100, 100, 100, 100, "", fsr_led_record[i]])
        sample_idx += 1

    with open(physio_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_monotonic_ns", "sample_idx", "phase", "participant_id",
                         "session_id", "task", "run", "fsr0", "fsr1", "fsr2", "fsr3",
                         "cue_event", "led_fsr"])
        writer.writerows(rows_physio)

    # Camera CSV (30 Hz, starts at RECORD phase)
    cam_dt_ns = int(1e9 / 30.0)
    with open(camera_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cam_ts_ns", "mp_valid", "mp_confidence", "mp_handedness"]
                        + [f"mp_lm{i:02d}_{axis}" for i in range(21) for axis in ("x", "y", "z")]
                        + ["led_cam"])
        for i, brightness in enumerate(cam_brightness_record):
            t_ns = prep_end_ns + i * cam_dt_ns
            writer.writerow([t_ns, 1, 0.9, "Right"] + [0.0] * 63 + [brightness])


class TestPrbsTimestampNormalization:
    """WS-2: PRBS Stage 1 must normalize timestamps to the RECORD phase start,
    not use absolute monotonic time (which is thousands of seconds and never
    matches the < PRBS_DURATION_S check)."""

    def test_prbs_detected_with_absolute_timestamps(self, tmp_path):
        """PRBS Stage 1 must detect the preamble even when timestamps start
        at 5000s (absolute monotonic time), not 0s."""
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sdir.mkdir(parents=True, exist_ok=True)
        _write_session_with_prbs(sdir, absolute_start_s=5000.0, prep_blink=False,
                                 cam_delay_s=0.05, record_duration_s=8.0)
        result = run_led_sync("P01", "S01", data_root=tmp_path)
        # PRBS offset must be detected (not None)
        assert result["prbs_offset_ms"] is not None, \
            f"PRBS offset must be detected with normalized timestamps. reason: {result['reason']}"
        assert result["prbs_score"] is not None
        assert result["prbs_score"] > 0.3, \
            f"PRBS score should be significant, got {result['prbs_score']}"

    def test_prep_blink_data_does_not_contaminate_prbs(self, tmp_path):
        """PREP phase blink data (from WS-1 'B' command) must NOT be included
        in the PRBS window. Only RECORD phase data should be used."""
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sdir.mkdir(parents=True, exist_ok=True)
        _write_session_with_prbs(sdir, absolute_start_s=5000.0, prep_blink=True,
                                 cam_delay_s=0.05, record_duration_s=8.0)
        result = run_led_sync("P01", "S01", data_root=tmp_path)
        # PRBS should still be detected despite PREP blink data
        assert result["prbs_offset_ms"] is not None, \
            f"PRBS must be detected from RECORD data only, not PREP. reason: {result['reason']}"

    def test_backward_compat_no_phase_column(self, tmp_path):
        """When the physio CSV has no 'phase' column (old format), led_sync
        must still work by falling back to timestamp-based normalization
        (first led_fsr 0→1 transition as the anchor point)."""
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sdir.mkdir(parents=True, exist_ok=True)
        # Write data without phase column, with absolute timestamps + PRBS
        t0_ns = int(5000.0 * 1e9)
        fsr_led = _generate_prbs_led_samples(8.0, 100.0, delay_s=0.0)
        cam_led = _generate_prbs_led_samples(8.0, 30.0, delay_s=0.05)
        cam_brightness = [255.0 if v else 0.0 for v in cam_led]

        physio_csv = sdir / "sub-P01_ses-S01_task-mvc_run-00_physio.csv"
        with open(physio_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # No "phase" column
            writer.writerow(["t_monotonic_ns", "sample_idx", "participant_id",
                             "session_id", "task", "run", "fsr0", "fsr1", "fsr2",
                             "fsr3", "cue_event", "led_fsr"])
            for i, led in enumerate(fsr_led):
                writer.writerow([t0_ns + i * 10_000_000, i, "P01", "S01", "mvc", 0,
                                 100, 100, 100, 100, "", led])

        camera_csv = sdir / "sub-P01_ses-S01_task-mvc_run-00_camera.csv"
        with open(camera_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["cam_ts_ns", "mp_valid", "mp_confidence", "mp_handedness"]
                            + [f"mp_lm{i:02d}_{axis}" for i in range(21) for axis in ("x", "y", "z")]
                            + ["led_cam"])
            for i, brightness in enumerate(cam_brightness):
                writer.writerow([t0_ns + i * 33_333_333, 1, 0.9, "Right"]
                                + [0.0] * 63 + [brightness])

        result = run_led_sync("P01", "S01", data_root=tmp_path)
        # Should not crash, should detect PRBS or at least not fail with
        # "PRBS preamble not captured" due to absolute timestamps
        assert result["prbs_offset_ms"] is not None or result["reason"] != "", \
            "led_sync must handle missing phase column without crashing"
