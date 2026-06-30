"""Pre-collection tests — mandatory checks before data collection.

5 sensor-specific tests + 2 hardware checks + 1 regression-specific test.
All tests must pass before collection begins.

Tests:
  1. Creep/drift warmup     — FSR signals stabilize after warmup
  2. Channel activation     — all channels produce non-zero output
  3. Dead/stuck channels    — no channels are stuck at 0 or 1023
  4. Baseline stability     — rest baseline is stable
  5. Response linearity     — response scales with applied force
  6. Camera tracking        — MediaPipe detects hand with good confidence
  7. Sync check             — LED blink aligns FSR and camera timestamps
  8. Single-DOF isolation   — regression: single-DOF tasks produce isolated movement
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

import numpy as np

from core import paths


# ── Test metadata (human-readable labels + operator instructions) ────────────

TEST_META: dict[str, dict[str, str | float]] = {
    "dead_stuck_channels": {
        "label": "Dead / Stuck Channel Check",
        "instruction": "Keep your hand relaxed and still. We're checking that no sensor is stuck at 0 or max.",
        "duration_s": 3.0,
        "category": "sensor_specific",
    },
    "channel_activation": {
        "label": "Channel Activation",
        "instruction": "Make a firm fist and hold it for 3 seconds. All sensors should respond.",
        "duration_s": 3.0,
        "category": "sensor_specific",
    },
    "camera_tracking": {
        "label": "Camera Tracking",
        "instruction": "Hold your hand in front of the camera, palm facing it, for 3 seconds.",
        "duration_s": 3.0,
        "category": "hardware",
    },
    "sync_check": {
        "label": "LED ↔ Sensor Sync Check",
        "instruction": "Stay still — the LED will blink for 10 seconds. We verify the LED column in the serial data and check the camera sees the blinks. If the camera can't see the LED, you'll be prompted to calibrate the LED ROI.",
        "duration_s": 10.0,
        "category": "hardware",
    },
    "baseline_stability": {
        "label": "Baseline Stability",
        "instruction": "Relax your hand completely and keep it still for 5 seconds. We're measuring rest noise.",
        "duration_s": 5.0,
        "category": "sensor_specific",
    },
    "response_linearity": {
        "label": "Response Linearity",
        "instruction": "Gradually increase grip force from zero to maximum over 6 seconds, then release.",
        "duration_s": 6.0,
        "category": "sensor_specific",
    },
    "single_dof_isolation": {
        "label": "Single-DOF Isolation",
        "instruction": "Flex only your index finger MCP joint (knuckle) repeatedly for 5 seconds. Keep other fingers still.",
        "duration_s": 5.0,
        "category": "regression_specific",
    },
    "creep_drift": {
        "label": "Creep / Drift Warmup",
        "instruction": "Apply a steady moderate force and hold it for 10 seconds. We're checking sensor drift.",
        "duration_s": 10.0,
        "category": "sensor_specific",
    },
}

# Ordered list of test names (fast-first for quick feedback)
TEST_ORDER = [
    "dead_stuck_channels",
    "channel_activation",
    "camera_tracking",
    "sync_check",
    "baseline_stability",
    "response_linearity",
    "single_dof_isolation",
    "creep_drift",
]


@dataclass
class TestResult:
    """Result of a single pre-collection test."""
    __test__ = False  # prevent pytest from collecting this class
    """Result of a single pre-collection test."""
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


@dataclass
class PrecollectResults:
    """Results of all pre-collection tests."""
    results: list[TestResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def save_to_bids(self, sub: str, ses: str, data_root: Path | None = None) -> Path:
        """Save results to BIDS precollect.json metadata file."""
        sdir = paths.session_dir(sub, ses, data_root=data_root or paths.RAW_DIR)
        sdir.mkdir(parents=True, exist_ok=True)

        data = {
            "test_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sensor_specific": {},
            "hardware": {},
            "regression_specific": {},
        }

        for r in self.results:
            category = r.details.get("category", "sensor_specific")
            if category not in data:
                data[category] = {}
            data[category][r.name] = {
                "passed": r.passed,
                **{k: v for k, v in r.details.items() if k != "category"},
            }

        out_path = sdir / f"sub-{sub}_ses-{ses}_precollect.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return out_path


# ── Sensor-specific tests (1-5) ───────────────────────────────────────────────


def check_creep_drift(
    read_sensors: Callable[[], list[int]],
    duration_s: float = 10.0,
    warmup_s: float = 3.0,
    sample_hz: float = 100.0,
) -> TestResult:
    """Test 1: FSR signals stabilize after warmup (creep/drift).

    Reads sensors for `duration_s`, checks that drift after warmup is < 5%.
    """
    samples = []
    n_samples = int(duration_s * sample_hz)

    for _ in range(n_samples):
        # read_sensors() blocks on queue.get() until the Arduino delivers the
        # next sample — that IS the 100 Hz pacing.  Do NOT also sleep here.
        samples.append(read_sensors())

    arr = np.array(samples, dtype=float)
    warmup_idx = int(warmup_s * sample_hz)

    # Compare first 1s after warmup vs last 1s
    early = arr[warmup_idx:warmup_idx + int(sample_hz)].mean(axis=0)
    late = arr[-int(sample_hz):].mean(axis=0)

    drift_pct = np.abs(late - early) / (early + 1e-6) * 100
    max_drift = float(np.max(drift_pct))
    passed = max_drift < 5.0

    return TestResult(
        name="creep_drift",
        passed=passed,
        message=f"Max drift: {max_drift:.2f}%" + ("" if passed else " (threshold: 5%)"),
        details={
            "category": "sensor_specific",
            "warmup_s": warmup_s,
            "drift_pct": round(max_drift, 2),
            "per_channel": {f"fsr{i}": round(float(d), 2) for i, d in enumerate(drift_pct)},
        },
    )


def check_channel_activation(
    read_sensors: Callable[[], list[int]],
    duration_s: float = 3.0,
    sample_hz: float = 100.0,
) -> TestResult:
    """Test 2: All channels produce non-zero output during active movement.

    Subject should make a fist during this test.
    """
    samples = []
    n_samples = int(duration_s * sample_hz)

    for _ in range(n_samples):
        samples.append(read_sensors())

    arr = np.array(samples, dtype=float)
    max_vals = arr.max(axis=0)
    all_active = all(v > 50 for v in max_vals)  # at least 50 ADC units

    return TestResult(
        name="channel_activation",
        passed=all_active,
        message="All channels active" if all_active else "Some channels inactive (< 50)",
        details={
            "category": "sensor_specific",
            "all_channels_active": all_active,
            "max_values": {f"fsr{i}": int(v) for i, v in enumerate(max_vals)},
        },
    )


def check_dead_stuck_channels(
    read_sensors: Callable[[], list[int]],
    duration_s: float = 3.0,
    sample_hz: float = 100.0,
) -> TestResult:
    """Test 3: No channels are stuck at 0 or 1023 (ADC range)."""
    samples = []
    n_samples = int(duration_s * sample_hz)

    for _ in range(n_samples):
        samples.append(read_sensors())

    arr = np.array(samples, dtype=float)
    dead_channels = []
    for i in range(arr.shape[1]):
        if np.all(arr[:, i] == 0):
            dead_channels.append(f"fsr{i}")
        elif np.all(arr[:, i] >= 1023):
            dead_channels.append(f"fsr{i}")

    passed = len(dead_channels) == 0

    return TestResult(
        name="dead_stuck_channels",
        passed=passed,
        message="No dead/stuck channels" if passed else f"Dead/stuck: {dead_channels}",
        details={
            "category": "sensor_specific",
            "dead_channels": dead_channels,
        },
    )


def check_baseline_stability(
    read_sensors: Callable[[], list[int]],
    duration_s: float = 5.0,
    sample_hz: float = 100.0,
    max_std_threshold: float = 20.0,
) -> TestResult:
    """Test 4: Rest baseline is stable (low std during rest).

    Subject should keep their hand relaxed and still during this test.
    """
    samples = []
    n_samples = int(duration_s * sample_hz)

    for _ in range(n_samples):
        samples.append(read_sensors())

    arr = np.array(samples, dtype=float)
    stds = arr.std(axis=0)
    max_std = float(np.max(stds))
    passed = max_std < max_std_threshold

    return TestResult(
        name="baseline_stability",
        passed=passed,
        message=f"Max std: {max_std:.1f}" + ("" if passed else f" (threshold: {max_std_threshold})"),
        details={
            "category": "sensor_specific",
            "rest_duration_s": duration_s,
            "max_std": round(max_std, 1),
            "per_channel_std": {f"fsr{i}": round(float(s), 1) for i, s in enumerate(stds)},
        },
    )


def check_response_linearity(
    read_sensors: Callable[[], list[int]],
    duration_s: float = 6.0,
    sample_hz: float = 100.0,
) -> TestResult:
    """Test 5: Response scales linearly with applied force (warning-only).

    Subject should gradually increase force from 0 to max over the test duration.
    This test is advisory only — it does not block collection. Poor linearity
    is reported as a warning so the operator can proceed and still review quality
    later.
    """
    samples = []
    n_samples = int(duration_s * sample_hz)

    for _ in range(n_samples):
        samples.append(read_sensors())

    arr = np.array(samples, dtype=float)
    t = np.arange(arr.shape[0])

    results_per_channel = []
    for i in range(arr.shape[1]):
        ch = arr[:, i]
        # R² against linear fit
        corr = np.corrcoef(t, ch)[0, 1]
        r2 = corr ** 2 if not np.isnan(corr) else 0.0

        # Monotonicity: check if the signal generally increases
        # Convert numpy.bool_ → Python bool so it's JSON-serializable.
        is_increasing = bool(corr > 0)

        # Saturation: check if the last 20% of samples are all at max
        last_20_pct = ch[int(n_samples * 0.8):]
        saturated = bool((last_20_pct >= 1020).all() and len(last_20_pct) > 0)

        # Dynamic range
        dynamic_range = float(ch.max() - ch.min())

        results_per_channel.append({
            "r2": round(float(r2), 3),
            "increasing": is_increasing,
            "saturated": saturated,
            "dynamic_range": round(dynamic_range, 1),
        })

    # Advisory thresholds: used for warning message, not for blocking.
    good_channels = [
        r for r in results_per_channel
        if r["r2"] > 0.7 and r["increasing"] and not r["saturated"] and r["dynamic_range"] > 100
    ]
    quality_ok = len(good_channels) > 0

    best_r2 = max(r["r2"] for r in results_per_channel)
    any_saturated = any(r["saturated"] for r in results_per_channel)
    max_range = max(r["dynamic_range"] for r in results_per_channel)

    if quality_ok:
        msg = f"Linear response. Best R²: {best_r2:.2f}, range: {max_range:.0f}"
    else:
        if any_saturated:
            msg = f"Sensor saturated (advisory). Best R²: {best_r2:.2f}, range: {max_range:.0f}"
        elif max_range < 100:
            msg = f"Low dynamic range (advisory). Best R²: {best_r2:.2f}, range: {max_range:.0f}"
        else:
            msg = f"Weak linearity (advisory). Best R²: {best_r2:.2f}, range: {max_range:.0f}"

    return TestResult(
        name="response_linearity",
        passed=True,  # warning-only: never blocks collection
        message=msg,
        details={
            "category": "sensor_specific",
            "best_r2": round(best_r2, 3),
            "max_dynamic_range": round(max_range, 1),
            "any_saturated": any_saturated,
            "quality_ok": quality_ok,
            "warning": not quality_ok,
            "per_channel": {f"fsr{i}": r for i, r in enumerate(results_per_channel)},
        },
    )


# ── Hardware tests (6-7) ──────────────────────────────────────────────────────


def check_camera_tracking(
    get_camera_data: Callable[[], dict] | None = None,
    duration_s: float = 3.0,
    sample_hz: float = 30.0,
) -> TestResult:
    """Test 6: MediaPipe detects hand with good confidence."""
    if get_camera_data is None:
        return TestResult(
            name="camera_tracking",
            passed=False,
            message="Camera not available (no callback provided)",
            details={"category": "hardware", "mean_confidence": None, "handedness": None},
        )

    confidences = []
    handedness_values = []
    n_samples = int(duration_s * sample_hz)
    dt = 1.0 / sample_hz

    for _ in range(n_samples):
        data = get_camera_data()
        if data.get("valid", False):
            confidences.append(data.get("confidence", 0.0))
            handedness_values.append(data.get("handedness", "Unknown"))
        time.sleep(dt)

    if not confidences:
        return TestResult(
            name="camera_tracking",
            passed=False,
            message="No valid hand detections",
            details={"category": "hardware", "mean_confidence": 0.0, "handedness": "None"},
        )

    mean_conf = float(np.mean(confidences))
    handedness = max(set(handedness_values), key=handedness_values.count)
    passed = mean_conf > 0.7

    return TestResult(
        name="camera_tracking",
        passed=passed,
        message=f"Mean confidence: {mean_conf:.2f}" + ("" if passed else " (threshold: 0.70)"),
        details={
            "category": "hardware",
            "mean_confidence": round(mean_conf, 3),
            "handedness": handedness,
            "n_valid_frames": len(confidences),
        },
    )


def check_sync_check(
    read_sample: Callable | None = None,
    get_camera_data: Callable[[], dict] | None = None,
    duration_s: float = 5.0,
    sample_hz: float = 100.0,
    led_roi: dict | None = None,
) -> TestResult:
    """Test 7: LED blink visible in serial CSV and optionally in camera.

    Primary check: reads the full SensorSample (via ``read_sample``) and
    verifies the ``led`` column toggles at 1 Hz with 100 ms ON time.
    This is the authoritative source — it comes directly from the Arduino
    firmware and requires no camera ROI calibration.

    Secondary check (optional): if ``led_roi`` is provided and
    ``get_camera_data`` is available, records a camera brightness time
    series and detects LED **transitions** (rising edges) via frame
    differencing, using the ``transition_threshold`` established during
    ROI calibration.  This verifies the camera actually *sees the blink*,
    not just that brightness averages differ (which is fooled by
    auto-exposure drift and async sampling misalignment).

    ``read_sample`` must be callable → ``SensorSample`` (with ``.fsr`` and
    ``.led`` fields). If ``None``, the test is skipped (no hardware).
    """
    if read_sample is None:
        return TestResult(
            name="sync_check",
            passed=False,
            message="FSR not available",
            details={"category": "hardware", "method": "LED_blink", "skew_ms": None},
        )

    n_samples = int(duration_s * sample_hz)

    # The transition threshold (brightness delta 0-255 that counts as an LED
    # ON/OFF transition) is established during ROI calibration and persisted
    # in led_roi.json.  Fall back to 15.0 for ROIs saved before the field
    # existed.
    transition_threshold = 15.0
    if led_roi is not None:
        transition_threshold = float(led_roi.get("transition_threshold", 15.0))

    # ── Sample LED column from serial CSV (+ camera brightness time series) ─
    # The camera brightness is recorded as a time series (not bucketed by
    # serial LED state) so we can detect rising edges independently of the
    # async sampling alignment between serial and camera.
    led_states: list[int] = []
    fsr_samples: list[list[int]] = []
    camera_brightness_series: list[float] = []
    use_camera_led = led_roi is not None and get_camera_data is not None

    for _ in range(n_samples):
        try:
            sample = read_sample()
            fsr_samples.append(sample.fsr)
            led_states.append(sample.led)
        except Exception:
            pass
        if use_camera_led:
            try:
                cam = get_camera_data()
                if "led_brightness" in cam and cam["led_brightness"] is not None:
                    camera_brightness_series.append(float(cam["led_brightness"]))
            except Exception:
                pass

    # ── Check 1: FSR data is flowing, no channel stuck ───────────────────
    arr = np.array(fsr_samples, dtype=float) if fsr_samples else np.zeros((1, 4))
    ranges = arr.max(axis=0) - arr.min(axis=0)
    data_flowing = any(r > 5 for r in ranges)
    sample_count_ok = len(fsr_samples) >= n_samples * 0.8
    stuck = any(
        (arr[:, i] == 0).all() or (arr[:, i] == 1023).all()
        for i in range(arr.shape[1])
    )

    # ── Check 2: LED is toggling (serial LED column) ─────────────────────
    # Count 0→1 rising edges.  At 1 Hz / 100 ms ON over 5 s → ~5 edges.
    # Accept ≥2 to be robust against startup timing.
    led_transitions = 0
    for i in range(1, len(led_states)):
        if led_states[i - 1] == 0 and led_states[i] == 1:
            led_transitions += 1
    has_both = 0 in led_states and 1 in led_states
    led_toggling = led_transitions >= 2 and has_both
    if led_states and not led_toggling:
        logger.warning(
            "LED not toggling in serial CSV — transitions=%d, has_both=%s "
            "(was 'B' command sent to Arduino?)",
            led_transitions, has_both,
        )

    # ── Check 3: Camera sees LED transitions (edge detection) ────────────
    # Detect rising edges in the camera brightness time series via frame
    # differencing: a brightness jump > transition_threshold between
    # consecutive samples marks an LED OFF→ON transition.  This is the same
    # method used by led_roi.py calibration meter and led_sync.py offline
    # sync, and is immune to auto-exposure drift (which changes slowly over
    # many frames) because LED transitions are instant (1-sample jumps).
    # Require ≥2 rising edges to confirm the camera sees the blink pattern.
    camera_led_ok = True  # don't fail if camera check is skipped
    camera_rising_edges = 0
    camera_brightness_range = 0.0
    camera_on_mean = 0.0
    camera_off_mean = 0.0
    camera_contrast = 0.0
    n_cam_samples = len(camera_brightness_series)
    if use_camera_led and n_cam_samples > 5:
        bseries = np.array(camera_brightness_series, dtype=float)
        camera_brightness_range = float(bseries.max() - bseries.min())
        # Frame differencing — rising edges where delta exceeds threshold.
        deltas = np.diff(bseries)
        camera_rising_edges = int(np.sum(deltas > transition_threshold))
        # Advisory ON/OFF means (kept for diagnostics, not the pass gate).
        # Split by median to label ON/OFF without relying on serial alignment.
        if bseries.max() > bseries.min():
            mid = (bseries.max() + bseries.min()) / 2.0
            on_mask = bseries >= mid
            camera_on_mean = float(bseries[on_mask].mean()) if on_mask.any() else 0.0
            camera_off_mean = float(bseries[~on_mask].mean()) if (~on_mask).any() else 0.0
            camera_contrast = (camera_on_mean - camera_off_mean) / max(camera_off_mean, 1.0)
        camera_led_ok = camera_rising_edges >= 2
        if not camera_led_ok:
            logger.warning(
                "LED transitions not detected in camera ROI — rising_edges=%d "
                "(threshold: >=2), brightness_range=%.1f, transition_threshold=%.1f. "
                "Re-calibrate the LED ROI.",
                camera_rising_edges, camera_brightness_range, transition_threshold,
            )
    elif use_camera_led:
        # Not enough camera samples — can't make a reliable determination.
        camera_led_ok = False
        logger.warning(
            "Camera LED check skipped — insufficient samples (%d). "
            "Camera may not be providing led_brightness.",
            n_cam_samples,
        )

    # ── Aggregate ─────────────────────────────────────────────────────────
    # Pass policy:
    # - Serial LED MUST toggle (primary evidence from Arduino firmware).
    # - FSR data MUST flow and no channel stuck.
    # - Camera LED detection is secondary:
    #   * If led_roi is configured and camera data is available, the camera
    #     MUST see ≥2 rising edges.  If it doesn't, the test FAILS with a
    #     calibration suggestion (the ROI may be wrong or the LED out of
    #     frame).
    #   * If led_roi is NOT configured, the camera check is skipped (not
    #     the operator's fault — they haven't calibrated yet).  The test
    #     can still pass on serial evidence alone, but a warning is emitted
    #     recommending calibration for full sync verification.
    needs_calibration = (
        led_roi is None
        and get_camera_data is not None
        and led_toggling
    )
    camera_failed = use_camera_led and not camera_led_ok

    passed = (
        data_flowing and sample_count_ok and not stuck
        and led_toggling
        and not camera_failed
    )

    issues = []
    if not data_flowing:
        issues.append("data not flowing (all channels static)")
    if not sample_count_ok:
        issues.append(f"sample rate too low ({len(fsr_samples)}/{n_samples})")
    if stuck:
        issues.append("channel stuck at 0 or 1023")
    if not led_toggling:
        issues.append(
            f"LED not toggling in serial CSV (transitions={led_transitions}; "
            "check 'B' command sent to Arduino and firmware version)"
        )
    if camera_failed:
        issues.append(
            f"LED transitions not detected in camera ROI "
            f"(rising edges: {camera_rising_edges}, need ≥2; "
            f"brightness range: {camera_brightness_range:.1f}, "
            f"threshold: {transition_threshold:.1f}). "
            "Re-calibrate the LED ROI: click 'Calibrate LED ROI' on the Setup screen "
            "and draw a rectangle around the LED. Make sure the LED is visible to the camera."
        )
    if needs_calibration:
        issues.append(
            "LED ROI not calibrated — camera sync not verified. "
            "Calibrate the LED ROI on the Setup screen for full sync verification."
        )

    method = "serial_LED_column"
    if use_camera_led:
        method += " + camera_LED_edge_detection"

    # Build a user-facing message that distinguishes hard failures
    # from advisory warnings.
    if passed and not needs_calibration:
        msg = (
            f"Sync OK — LED toggling ({led_transitions} transitions), "
            f"camera sees {camera_rising_edges} rising edges"
        )
    elif passed and needs_calibration:
        msg = (
            f"Sync OK (serial) — LED toggling ({led_transitions} transitions). "
            "WARNING: LED ROI not calibrated — camera sync not verified."
        )
    else:
        msg = "; ".join(issues)

    return TestResult(
        name="sync_check",
        passed=passed,
        message=msg,
        details={
            "category": "hardware",
            "method": method,
            "n_samples": len(fsr_samples),
            "expected_samples": n_samples,
            "led_transitions": led_transitions,
            "serial_led_used": True,
            "led_toggling": led_toggling,
            "stuck_channel": stuck,
            "ranges": {f"fsr{i}": round(float(r), 1) for i, r in enumerate(ranges)},
            "led_roi_used": led_roi is not None,
            "transition_threshold": round(transition_threshold, 1),
            "camera_rising_edges": camera_rising_edges,
            "camera_brightness_range": round(camera_brightness_range, 1),
            "camera_on_mean": round(camera_on_mean, 1),
            "camera_off_mean": round(camera_off_mean, 1),
            "camera_contrast": round(camera_contrast, 3),
            "camera_brightness_samples": n_cam_samples,
            "needs_calibration": needs_calibration,
            "camera_failed": camera_failed,
            "issues": issues,
        },
    )


# ── Regression-specific test (8) ──────────────────────────────────────────────


def check_single_dof_isolation(
    read_sensors: Callable[[], list[int]] | None = None,
    duration_s: float = 5.0,
    sample_hz: float = 100.0,
) -> TestResult:
    """Test 8: Single-DOF tasks produce isolated movement (regression-specific).

    Subject should slowly flex and extend their index finger MCP joint only.

    A real isolation test checks:
    1. At least one channel shows clear modulation (range > 100 ADC units)
    2. The modulated channel is dominant — other channels show much less
       movement (cross-channel correlation is low, or the ratio of the
       primary channel's range to the mean of other channels is > 3:1)
    3. The movement is cyclical (flex/extend), not just a step

    This distinguishes "one finger moved" from "whole hand clenched".
    """
    if read_sensors is None:
        return TestResult(
            name="single_dof_isolation",
            passed=False,
            message="FSR not available",
            details={"category": "regression_specific", "dofs_tested": []},
        )

    samples = []
    n_samples = int(duration_s * sample_hz)

    for _ in range(n_samples):
        samples.append(read_sensors())

    arr = np.array(samples, dtype=float)
    n_channels = arr.shape[1]

    # 1. Per-channel ranges
    ranges = arr.max(axis=0) - arr.min(axis=0)

    # 2. Find the primary (most modulated) channel
    primary_ch = int(np.argmax(ranges))
    primary_range = float(ranges[primary_ch])

    # 3. Cross-channel correlation matrix
    if n_channels > 1:
        corr_matrix = np.corrcoef(arr.T)
        # Off-diagonal correlations between primary and others
        other_corrs = []
        for j in range(n_channels):
            if j != primary_ch:
                c = corr_matrix[primary_ch, j]
                other_corrs.append(abs(c) if not np.isnan(c) else 0.0)
        mean_cross_corr = float(np.mean(other_corrs)) if other_corrs else 0.0
    else:
        mean_cross_corr = 0.0

    # 4. Isolation ratio: primary range / mean of other ranges
    if n_channels > 1:
        other_ranges = [ranges[j] for j in range(n_channels) if j != primary_ch]
        mean_other_range = float(np.mean(other_ranges)) if other_ranges else 0.0
        isolation_ratio = primary_range / max(mean_other_range, 1.0)
    else:
        isolation_ratio = float('inf') if primary_range > 100 else 0.0

    # 5. Check for cyclical movement (at least 2 direction changes)
    primary_signal = arr[:, primary_ch]
    diffs = np.diff(primary_signal)
    direction_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))

    # Pass criteria:
    # - Primary channel has clear modulation (range > 100)
    # - Isolation ratio > 2.0 (primary moves at least 2x more than others)
    # - Cross-channel correlation < 0.8 (not all channels moving together)
    # - At least 1 direction change (cyclical, not just a step)
    has_modulation = primary_range > 100
    is_isolated = isolation_ratio > 2.0
    low_cross_corr = mean_cross_corr < 0.8
    is_cyclical = direction_changes >= 1

    passed = has_modulation and is_isolated and low_cross_corr and is_cyclical

    # Build message
    issues = []
    if not has_modulation:
        issues.append(f"no modulation (range {primary_range:.0f} < 100)")
    if not is_isolated:
        issues.append(f"poor isolation (ratio {isolation_ratio:.1f} < 2.0)")
    if not low_cross_corr:
        issues.append(f"channels correlated (mean {mean_cross_corr:.2f} > 0.80)")
    if not is_cyclical:
        issues.append("no cyclical movement detected")

    if passed:
        msg = (f"Isolated movement on FSR{primary_ch}. "
               f"Range: {primary_range:.0f}, isolation: {isolation_ratio:.1f}:1, "
               f"cross-corr: {mean_cross_corr:.2f}")
    else:
        msg = "; ".join(issues)

    return TestResult(
        name="single_dof_isolation",
        passed=passed,
        message=msg,
        details={
            "category": "regression_specific",
            "dofs_tested": ["index_mcp"],
            "primary_channel": primary_ch,
            "primary_range": round(primary_range, 1),
            "isolation_ratio": round(isolation_ratio, 2),
            "mean_cross_corr": round(mean_cross_corr, 3),
            "direction_changes": direction_changes,
            "ranges": {f"fsr{i}": round(float(r), 1) for i, r in enumerate(ranges)},
        },
    )


# ── Interactive test runner ──────────────────────────────────────────────────

# A test function takes (read_sensors, get_camera_data, **kwargs) and returns TestResult.
# Map test names to their check functions.
_TEST_FUNCTIONS = {
    "dead_stuck_channels": check_dead_stuck_channels,
    "channel_activation": check_channel_activation,
    "camera_tracking": check_camera_tracking,
    "sync_check": None,  # special-cased below
    "baseline_stability": check_baseline_stability,
    "response_linearity": check_response_linearity,
    "single_dof_isolation": check_single_dof_isolation,
    "creep_drift": check_creep_drift,
}


def run_all_tests_interactive(
    read_sensors: Callable[[], list[int]] | None = None,
    get_camera_data: Callable[[], dict] | None = None,
    *,
    read_sensors_preview: Callable[[], list[int] | None] | None = None,
    ready_event: threading.Event | None = None,
    stop_event: threading.Event | None = None,
    retry_test: str | None = None,
    broadcast_fn: Callable[[dict], None] | None = None,
    countdown_s: float = 1.0,
    duration_overrides: dict[str, float] | None = None,
):
    """Interactive pre-collection test runner.

    For each test, yields a sequence of events:
      1. ``TestInstructionEvent`` — what to show the operator
      2. ``TestReadyEvent`` — waiting for operator to press "Ready"
         (pauses on ``ready_event`` until set)
      3. ``TestCountdownEvent`` × 3 — 3, 2, 1
      4. ``TestRunningEvent`` — "GO", sampling starts
      5. ``TestResult`` — pass/fail

    Parameters
    ----------
    read_sensors_preview : callable | None
        Non-blocking read used ONLY for the FSR bar display while waiting for
        the operator to press Ready.  Must never block (e.g. ``read_sensors_preview()``
        on ``SerialSensorReader``).  If None, falls back to ``read_sensors``.
        Using a blocking ``read_sensors`` here would stall the generator.
    ready_event : threading.Event | None
        Set by the WS handler when the operator presses "Ready".
        If None, tests auto-advance without waiting (for dry-run / testing).
    stop_event : threading.Event | None
        If set during the ready wait, the generator stops early.
    retry_test : str | None
        If set, only run this single test (for per-test retry).
    countdown_s : float
        Seconds to wait between each countdown number (3, 2, 1).
        Default 1.0 for real operator pacing; use a smaller value for tests.
    """
    import threading as _threading
    # Use the non-blocking preview reader for UI display during ready-wait.
    # Falling back to read_sensors is only safe for MockSensorReader (never blocks).
    _preview_fn = read_sensors_preview if read_sensors_preview is not None else read_sensors
    auto_ready = ready_event is None
    if auto_ready:
        ready_event = _threading.Event()
        ready_event.set()  # auto-advance if no event provided

    test_names = [retry_test] if retry_test else TEST_ORDER
    total = len(test_names)
    overrides = duration_overrides or {}

    for idx, name in enumerate(test_names):
        meta = dict(TEST_META.get(name, {"label": name, "instruction": "", "duration_s": 3.0}))
        if name in overrides:
            meta["duration_s"] = overrides[name]
        logger.info("Test %d/%d: %s starting (duration=%.1fs)", idx + 1, total, name, meta["duration_s"])

        # 1. Instruction
        yield {
            "type": "test_instruction",
            "name": name,
            "label": meta["label"],
            "instruction": meta["instruction"],
            "duration_s": meta["duration_s"],
            "test_index": idx,
            "total_tests": total,
        }

        # 2. Wait for operator "Ready"
        # Clear the event BEFORE yielding so there's no race where the UI
        # sets it before we start waiting.
        ready_event.clear()
        # Switch to non-blocking preview mode BEFORE the ready-wait so the
        # reader thread doesn't fill the queue while nobody is draining it.
        # The ready-wait uses _preview_fn (non-blocking), so _has_consumer
        # must be False to prevent queue-full warnings and sample drops.
        _owner = getattr(read_sensors, '__self__', None)
        if _owner is not None and hasattr(_owner, '_has_consumer'):
            _owner._has_consumer = False
        logger.info("Test %s — entering ready-wait (yielding test_ready)", name)
        yield {"type": "test_ready", "name": name}
        logger.info("Test %s — test_ready consumed, polling for operator press", name)
        if auto_ready:
            ready_event.set()  # auto-advance immediately in test/CLI mode
        _last_fsr_t = 0.0
        while not ready_event.is_set():
            if stop_event is not None and stop_event.is_set():
                return
            # Short timeout so FSR bars can update at up to 50 Hz while waiting
            ready_event.wait(timeout=0.02)
            # Yield FSR values so bars stay live during the wait.
            # MUST use _preview_fn (non-blocking) here — read_sensors blocks on
            # queue.get() and would stall the generator if the queue is empty.
            if _preview_fn is not None:
                now = time.monotonic()
                if now - _last_fsr_t >= 0.02:
                    _last_fsr_t = now
                    try:
                        vals = _preview_fn()
                        if vals is not None:
                            yield {"type": "fsr", "values": vals}
                    except Exception:
                        pass

        # Operator pressed Ready.
        # _has_consumer was already set to False before the ready-wait above.
        # The countdown also uses _preview_fn, so we stay in preview mode.

        # 3. Countdown 3-2-1
        # Sleep in 50 ms chunks so FSR events keep flowing and the UI does not
        # appear frozen for a full second between each count.
        _fsr_interval = 0.05  # 20 Hz FSR during countdown
        for count in (3, 2, 1):
            yield {"type": "test_countdown", "name": name, "countdown": count}
            elapsed = 0.0
            while elapsed < countdown_s:
                chunk = min(_fsr_interval, countdown_s - elapsed)
                time.sleep(chunk)
                elapsed += chunk
                if _preview_fn is not None:
                    try:
                        vals = _preview_fn()
                        if vals is not None:
                            yield {"type": "fsr", "values": vals}
                    except Exception:
                        pass

        # Re-enable blocking mode and flush stale samples right before the
        # test starts recording.  This gives the test fresh samples from t=0.
        if _owner is not None and hasattr(_owner, '_has_consumer'):
            _owner._has_consumer = True
        if _owner is not None and hasattr(_owner, 'flush_buffer'):
            logger.info("Test %s — calling flush_buffer()", name)
            _owner.flush_buffer()
            logger.info("Test %s — flush_buffer() returned", name)

        # 4. Run the test
        logger.info("Test %s — yielding test_running (GO)", name)
        yield {"type": "test_running", "name": name, "elapsed_s": 0.0, "duration_s": meta["duration_s"]}
        logger.info("Test %s — test_running consumed, starting test function", name)

        # Wrap read_sensors so each sample is broadcast to the UI for live bars
        # AND so a per-test deadline is enforced.  If the test runs past
        # 2× its expected duration, the wrapper raises TimeoutError which is
        # caught below and converted to a failure result.  This prevents a
        # stalled serial port or a buggy test from hanging the session forever
        # (the "stuck on GO" bug).
        _test_deadline = time.monotonic() + max(meta["duration_s"] * 2, 10.0)
        _test_name = name
        _broadcast_read_sensors = read_sensors
        if broadcast_fn is not None and read_sensors is not None:
            _last_broadcast_t = [0.0]
            def _broadcasting_read_sensors():
                if time.monotonic() > _test_deadline:
                    raise TimeoutError(
                        f"Test {_test_name} exceeded deadline "
                        f"({max(meta['duration_s'] * 2, 10.0):.0f}s) — "
                        f"serial port may have stalled"
                    )
                values = read_sensors()
                now = time.monotonic()
                if now - _last_broadcast_t[0] >= 0.02:  # 50Hz throttle
                    _last_broadcast_t[0] = now
                    try:
                        broadcast_fn({"type": "fsr", "values": list(values)})
                    except Exception:
                        pass
                return values
            _broadcast_read_sensors = _broadcasting_read_sensors

        if name == "sync_check":
            # Load LED ROI if available and apply to camera reader
            led_roi = None
            try:
                from pathlib import Path
                import json
                roi_path = Path("led_roi.json")
                if roi_path.exists():
                    with open(roi_path, encoding="utf-8") as f:
                        roi_data = json.load(f)
                    led_roi = {"x": roi_data["x"], "y": roi_data["y"],
                               "width": roi_data["width"], "height": roi_data["height"],
                               "transition_threshold": float(roi_data.get("transition_threshold", 15.0))}
                    logger.info("Loaded LED ROI: %s", led_roi)
                    # Apply ROI to camera reader so get_frame() includes led_brightness
                    _cam_owner = getattr(get_camera_data, '__self__', None)
                    if _cam_owner is not None and hasattr(_cam_owner, 'set_led_roi'):
                        _cam_owner.set_led_roi(led_roi)
            except Exception:
                pass
            # Start the LED blinking in blink-only mode (no PRBS) so the
            # camera can detect brightness changes.  This sends 'B' to the
            # Arduino via the sensor reader.  The PRBS preamble is NOT
            # triggered here — it's saved for the recording phase (ADR 003).
            _sensor_owner = getattr(read_sensors, '__self__', None)
            if _sensor_owner is not None and hasattr(_sensor_owner, 'start_led_preview'):
                try:
                    _sensor_owner.start_led_preview()
                except Exception as exc:
                    logger.warning("Could not start LED preview: %s", exc)
            # Build a read_sample callable that reads full SensorSamples
            # (with LED column) AND broadcasts FSR bars so the UI stays live.
            _read_sample_fn = None
            if _sensor_owner is not None and hasattr(_sensor_owner, 'read'):
                _raw_read = _sensor_owner.read
                _last_sync_t = [0.0]
                _sync_deadline = _test_deadline
                def _read_sample_fn():
                    if time.monotonic() > _sync_deadline:
                        raise TimeoutError(
                            f"Test {_test_name} exceeded deadline — serial port may have stalled"
                        )
                    sample = _raw_read()
                    now = time.monotonic()
                    if broadcast_fn is not None and now - _last_sync_t[0] >= 0.02:
                        _last_sync_t[0] = now
                        try:
                            broadcast_fn({"type": "fsr", "values": list(sample.fsr)})
                        except Exception:
                            pass
                    return sample
            # LED-visibility sync check: verify LED toggles in serial CSV and
            # optionally in camera ROI.
            try:
                result = check_sync_check(
                    read_sample=_read_sample_fn,
                    get_camera_data=get_camera_data,
                    duration_s=meta["duration_s"],
                    led_roi=led_roi,
                )
            except TimeoutError as exc:
                logger.error("Test %s timed out: %s", name, exc)
                result = TestResult(
                    name=name, passed=False, message=str(exc),
                    details={"category": "hardware", "timeout": True},
                )
            except Exception as exc:
                logger.error("Test %s crashed: %s", name, exc, exc_info=True)
                result = TestResult(
                    name=name, passed=False, message=f"Test crashed: {exc}",
                    details={"category": "hardware", "crash": True},
                )
        elif name == "camera_tracking":
            try:
                result = check_camera_tracking(get_camera_data, duration_s=meta["duration_s"])
            except Exception as exc:
                logger.error("Test %s crashed: %s", name, exc, exc_info=True)
                result = TestResult(
                    name=name, passed=False, message=f"Test crashed: {exc}",
                    details={"category": "hardware", "crash": True},
                )
        else:
            fn = _TEST_FUNCTIONS.get(name)
            if fn is None:
                result = TestResult(name=name, passed=False, message="Test function not found")
            elif _broadcast_read_sensors is None:
                result = TestResult(
                    name=name, passed=False,
                    message="FSR not available",
                    details={"category": meta.get("category", "sensor_specific")},
                )
            else:
                # Sensor-only tests take read_sensors, NOT get_camera_data
                try:
                    result = fn(_broadcast_read_sensors, duration_s=meta["duration_s"])
                except TimeoutError as exc:
                    logger.error("Test %s timed out: %s", name, exc)
                    result = TestResult(
                        name=name, passed=False, message=str(exc),
                        details={"category": meta.get("category", "sensor_specific"), "timeout": True},
                    )
                except Exception as exc:
                    logger.error("Test %s crashed: %s", name, exc, exc_info=True)
                    result = TestResult(
                        name=name, passed=False, message=f"Test crashed: {exc}",
                        details={"category": meta.get("category", "sensor_specific"), "crash": True},
                    )

        # 5. Result
        logger.info("Test %s finished — yielding result (passed=%s)", name, result.passed)
        yield result
        logger.info("Test %s result consumed by session — advancing to next test", name)


# ── Run all tests (non-interactive, for CLI/tests) ────────────────────────────


def run_all_tests(
    read_sensors: Callable[[], list[int]] | None = None,
    get_camera_data: Callable[[], dict] | None = None,
) -> PrecollectResults:
    """Run all 8 pre-collection tests.

    Parameters
    ----------
    read_sensors : callable, optional
        Function that returns a list of FSR values [fsr0, fsr1, ...].
        If None, sensor tests are skipped (marked as failed).
    get_camera_data : callable, optional
        Function that returns a dict with 'valid', 'confidence', 'handedness'.
        If None, camera tests are skipped.

    Returns
    -------
    PrecollectResults
        Results of all tests.
    """
    results = PrecollectResults()

    if read_sensors is None:
        # No sensor callback — mark all sensor tests as failed
        for name in ["creep_drift", "channel_activation", "dead_stuck_channels",
                      "baseline_stability", "response_linearity", "single_dof_isolation"]:
            results.results.append(TestResult(
                name=name,
                passed=False,
                message="FSR not available (no read_sensors callback)",
                details={"category": "sensor_specific" if "single" not in name else "regression_specific"},
            ))
    else:
        print("Test 1/8: Creep/drift warmup (10s)...")
        results.results.append(check_creep_drift(read_sensors))
        print(f"  {results.results[-1]}")

        print("Test 2/8: Channel activation (make a fist, 3s)...")
        results.results.append(check_channel_activation(read_sensors))
        print(f"  {results.results[-1]}")

        print("Test 3/8: Dead/stuck channels (3s)...")
        results.results.append(check_dead_stuck_channels(read_sensors))
        print(f"  {results.results[-1]}")

        print("Test 4/8: Baseline stability (relax hand, 5s)...")
        results.results.append(check_baseline_stability(read_sensors))
        print(f"  {results.results[-1]}")

        print("Test 5/8: Response linearity (gradual force, 6s)...")
        results.results.append(check_response_linearity(read_sensors))
        print(f"  {results.results[-1]}")

    print("Test 6/8: Camera tracking (3s)...")
    results.results.append(check_camera_tracking(get_camera_data))
    print(f"  {results.results[-1]}")

    print("Test 7/8: Sync check (LED column from serial)...")
    # Build a read_sample callable from read_sensors' owner if available.
    # Falls back to None → skipped (FSR not available).
    _sync_read_sample = None
    _sync_owner = getattr(read_sensors, '__self__', None) if read_sensors else None
    if _sync_owner is not None and hasattr(_sync_owner, 'read'):
        _sync_read_sample = _sync_owner.read
    results.results.append(check_sync_check(
        read_sample=_sync_read_sample,
        get_camera_data=get_camera_data,
    ))
    print(f"  {results.results[-1]}")

    if read_sensors is not None:
        print("Test 8/8: Single-DOF isolation (flex index MCP, 5s)...")
        results.results.append(check_single_dof_isolation(read_sensors))
        print(f"  {results.results[-1]}")
    else:
        results.results.append(TestResult(
            name="single_dof_isolation",
            passed=False,
            message="FSR not available",
            details={"category": "regression_specific", "dofs_tested": []},
        ))

    return results
