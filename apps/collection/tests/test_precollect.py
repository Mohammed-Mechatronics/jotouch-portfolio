"""Tests for apps.collection.precollect — pre-collection tests."""

from __future__ import annotations

import json
import threading
from unittest.mock import patch

import numpy as np
import pytest

# Patch time.sleep in precollect for the entire module so every check_*
# sampling loop completes instantly.  Without this the suite takes 40+ seconds
# because each test runs real wall-clock sleeps at 1/sample_hz intervals.
pytestmark = pytest.mark.usefixtures("fast_sleep")

from apps.collection.precollect import (
    TestResult,
    PrecollectResults,
    check_creep_drift,
    check_channel_activation,
    check_dead_stuck_channels,
    check_baseline_stability,
    check_response_linearity,
    check_camera_tracking,
    check_single_dof_isolation,
    check_sync_check,
    run_all_tests,
    run_all_tests_interactive,
)


# ── Mock sensor readers ───────────────────────────────────────────────────────


def make_stable_reader(values: list[int] = None):
    """Return a reader that always returns the same values."""
    values = values or [300, 400, 500, 600]
    return lambda: list(values)


def make_drifting_reader(start: list[int], drift: float):
    """Return a reader whose values drift over time."""
    call_count = [0]
    def reader():
        vals = [int(s + drift * call_count[0]) for s in start]
        call_count[0] += 1
        return vals
    return reader


def make_fist_reader():
    """Return a reader that simulates a fist (high values)."""
    return lambda: [700, 800, 750, 850]


def make_dead_channel_reader(dead_idx: int = 0):
    """Return a reader with one dead channel (always 0)."""
    def reader():
        vals = [300, 400, 500, 600]
        vals[dead_idx] = 0
        return vals
    return reader


def make_stuck_channel_reader(stuck_idx: int = 0, stuck_val: int = 1023):
    """Return a reader with one stuck channel (always 1023)."""
    def reader():
        vals = [300, 400, 500, 600]
        vals[stuck_idx] = stuck_val
        return vals
    return reader


def make_ramp_reader(start: list[int], end: list[int], n_samples: int):
    """Return a reader that ramps from start to end values."""
    call_count = [0]
    def reader():
        t = call_count[0] / n_samples
        vals = [int(s + (e - s) * t) for s, e in zip(start, end)]
        call_count[0] += 1
        return vals
    return reader


def make_noisy_reader(base: list[int], std: float = 5.0):
    """Return a reader with small noise around base values."""
    return lambda: [int(b + np.random.normal(0, std)) for b in base]


def make_camera_reader(confidence: float = 0.95, valid: bool = True):
    """Return a mock camera data reader."""
    return lambda: {
        "valid": valid,
        "confidence": confidence,
        "handedness": "Right",
        "landmarks": [0.1] * 63,
    }


class TestCreepDrift:
    def test_stable_passes(self):
        reader = make_stable_reader([300, 400, 500, 600])
        result = check_creep_drift(reader, duration_s=2.0, warmup_s=0.5, sample_hz=100.0)
        assert result.passed

    def test_drifting_fails(self):
        reader = make_drifting_reader([300, 400, 500, 600], drift=10.0)
        result = check_creep_drift(reader, duration_s=2.0, warmup_s=0.5, sample_hz=100.0)
        assert not result.passed


class TestChannelActivation:
    def test_all_active_passes(self):
        reader = make_fist_reader()
        result = check_channel_activation(reader, duration_s=0.2, sample_hz=100.0)
        assert result.passed

    def test_low_values_fails(self):
        reader = make_stable_reader([10, 20, 30, 40])
        result = check_channel_activation(reader, duration_s=0.2, sample_hz=100.0)
        assert not result.passed


class TestDeadStuckChannels:
    def test_all_good_passes(self):
        reader = make_stable_reader()
        result = check_dead_stuck_channels(reader, duration_s=0.2, sample_hz=100.0)
        assert result.passed

    def test_dead_channel_fails(self):
        reader = make_dead_channel_reader(0)
        result = check_dead_stuck_channels(reader, duration_s=0.2, sample_hz=100.0)
        assert not result.passed
        assert "fsr0" in result.details["dead_channels"]

    def test_stuck_channel_fails(self):
        reader = make_stuck_channel_reader(1, 1023)
        result = check_dead_stuck_channels(reader, duration_s=0.2, sample_hz=100.0)
        assert not result.passed


class TestBaselineStability:
    def test_stable_passes(self):
        reader = make_noisy_reader([300, 400, 500, 600], std=2.0)
        result = check_baseline_stability(reader, duration_s=0.3, sample_hz=100.0)
        assert result.passed

    def test_noisy_fails(self):
        reader = make_noisy_reader([300, 400, 500, 600], std=100.0)
        result = check_baseline_stability(reader, duration_s=0.3, sample_hz=100.0)
        assert not result.passed


class TestResponseLinearity:
    def test_linear_ramp_passes(self):
        reader = make_ramp_reader([100, 100, 100, 100], [800, 800, 800, 800], 100)
        result = check_response_linearity(reader, duration_s=0.3, sample_hz=100.0)
        assert result.passed

    def test_constant_warns_but_passes(self):
        reader = make_stable_reader([400, 400, 400, 400])
        result = check_response_linearity(reader, duration_s=0.3, sample_hz=100.0)
        assert result.passed
        assert result.details["warning"] is True
        assert result.details["quality_ok"] is False


class TestCameraTracking:
    def test_good_camera_passes(self):
        reader = make_camera_reader(confidence=0.95)
        result = check_camera_tracking(reader, duration_s=0.2, sample_hz=30.0)
        assert result.passed

    def test_low_confidence_fails(self):
        reader = make_camera_reader(confidence=0.3)
        result = check_camera_tracking(reader, duration_s=0.2, sample_hz=30.0)
        assert not result.passed

    def test_no_camera_fails(self):
        result = check_camera_tracking(None, duration_s=0.2)
        assert not result.passed


class TestSingleDOFIsolation:
    def test_modulation_passes(self):
        # Only channel 0 modulates cyclically (flex/extend pattern)
        call_count = [0]
        def reader():
            t = call_count[0] / 20  # 2 cycles over 40 samples
            val = int(400 + 300 * np.sin(t * np.pi))
            call_count[0] += 1
            return [val, 400, 400, 400]
        result = check_single_dof_isolation(reader, duration_s=0.7, sample_hz=100.0)
        assert result.passed

    def test_all_channels_move_fails(self):
        # All channels move together — not isolated
        reader = make_ramp_reader([100, 100, 100, 100], [800, 800, 800, 800], 100)
        result = check_single_dof_isolation(reader, duration_s=0.3, sample_hz=100.0)
        assert not result.passed

    def test_no_modulation_fails(self):
        reader = make_stable_reader([400, 400, 400, 400])
        result = check_single_dof_isolation(reader, duration_s=0.3, sample_hz=100.0)
        assert not result.passed


class TestRunAllTests:
    def test_no_hardware(self):
        results = run_all_tests(read_sensors=None, get_camera_data=None)
        assert results.n_failed > 0
        assert not results.all_passed

    def test_with_mock_hardware(self):
        results = run_all_tests(
            read_sensors=make_stable_reader([300, 400, 500, 600]),
            get_camera_data=make_camera_reader(),
        )
        # Some tests may fail with constant values (linearity, isolation)
        # but at least the tests should run
        assert len(results.results) == 8


class TestPrecollectResults:
    def test_save_to_bids(self, tmp_path):
        results = PrecollectResults()
        results.results.append(TestResult(
            name="creep_drift",
            passed=True,
            details={"category": "sensor_specific", "drift_pct": 1.5},
        ))
        path = results.save_to_bids("P01", "S01", data_root=tmp_path)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "sensor_specific" in data
        assert data["sensor_specific"]["creep_drift"]["passed"] is True


class TestRunAllTestsInteractive:
    def test_broadcasts_fsr_events_during_tests(self):
        """FSR events are emitted during the test running phase when broadcast_fn is set."""
        ready_event = threading.Event()
        stop_event = threading.Event()

        fsr_events = []
        def broadcast_fn(event):
            if event.get("type") == "fsr":
                fsr_events.append(event)

        generator = run_all_tests_interactive(
            read_sensors=make_stable_reader([300, 400, 500, 600]),
            get_camera_data=make_camera_reader(),
            ready_event=ready_event,
            stop_event=stop_event,
            broadcast_fn=broadcast_fn,
        )

        # Drive the generator manually
        events_seen = []
        for event in generator:
            events_seen.append(event)
            if isinstance(event, dict) and event.get("type") == "test_ready":
                # Simulate operator pressing Ready
                ready_event.set()
            if isinstance(event, TestResult):
                # Stop after first test result (don't run all 8)
                stop_event.set()
                break

        # We should have seen at least one FSR event broadcast during the test
        assert len(fsr_events) > 0, "Expected FSR events to be broadcast during test execution"
        for e in fsr_events:
            assert "values" in e
            assert len(e["values"]) == 4

    def test_duration_overrides_apply(self):
        """duration_overrides changes the duration_s reported in test_instruction."""
        ready_event = threading.Event()
        stop_event = threading.Event()
        generator = run_all_tests_interactive(
            read_sensors=make_stable_reader([300, 400, 500, 600]),
            get_camera_data=make_camera_reader(),
            ready_event=ready_event,
            stop_event=stop_event,
            duration_overrides={"dead_stuck_channels": 7.5},
        )

        for event in generator:
            if isinstance(event, dict) and event.get("type") == "test_instruction":
                if event["name"] == "dead_stuck_channels":
                    assert event["duration_s"] == 7.5
                    stop_event.set()
                    break
            if isinstance(event, TestResult):
                stop_event.set()
                break


# ── check_sync_check ──────────────────────────────────────────────────────────


class TestSyncCheck:
    """check_sync_check uses the serial LED column as primary truth.

    ``read_sample`` → SensorSample is the only required input.
    Camera ROI check is an optional secondary gate.
    """

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _make_sample_reader(led_pattern: list[int], fsr_start=100, fsr_end=800):
        """Return a read_sample callable yielding SensorSamples with given LED pattern."""
        from apps.collection.sensor_reader import SensorSample
        call_count = [0]
        n = len(led_pattern)
        def read_sample():
            idx = call_count[0]
            led = led_pattern[idx % n]
            t = idx / 100.0
            fsr = [int(fsr_start + (fsr_end - fsr_start) * t / 5.0)] * 4
            call_count[0] += 1
            return SensorSample(fsr=fsr, led=led, t_ns=idx * 10_000_000)
        return read_sample

    @staticmethod
    def _toggling_led_pattern(total: int, hz: float = 100.0, blink_hz: float = 1.0, on_ms: float = 100.0) -> list[int]:
        """Generate a 1 Hz LED pattern at the given sample rate."""
        period = int(hz / blink_hz)
        on_samples = int(on_ms / 1000.0 * hz)
        return [1 if (i % period) < on_samples else 0 for i in range(total)]

    # ── serial LED column tests ───────────────────────────────────────────

    def test_serial_led_toggling_passes(self):
        """Toggling LED column → sync_check passes with correct details."""
        n = 500
        pattern = self._toggling_led_pattern(n)
        read_sample = self._make_sample_reader(pattern)
        result = check_sync_check(read_sample=read_sample, duration_s=5.0, sample_hz=100.0)
        assert result.passed, result.message
        assert result.details["serial_led_used"] is True
        assert result.details["led_toggling"] is True
        assert result.details["led_transitions"] >= 2
        assert result.name == "sync_check"

    def test_no_hardware_fails(self):
        """None read_sample → failed result immediately."""
        result = check_sync_check(read_sample=None)
        assert not result.passed
        assert "FSR not available" in result.message

    def test_serial_led_stuck_high_fails(self):
        """LED stuck HIGH (never LOW) → sync_check fails."""
        from apps.collection.sensor_reader import SensorSample
        call_count = [0]
        def read_sample():
            fsr = [300 + call_count[0] % 200] * 4
            call_count[0] += 1
            return SensorSample(fsr=fsr, led=1, t_ns=call_count[0] * 10_000_000)
        result = check_sync_check(read_sample=read_sample, duration_s=5.0, sample_hz=100.0)
        assert not result.passed
        assert "not toggling" in result.message

    def test_serial_led_stuck_low_fails(self):
        """LED stuck LOW (never HIGH) → sync_check fails."""
        from apps.collection.sensor_reader import SensorSample
        call_count = [0]
        def read_sample():
            fsr = [300 + call_count[0] % 200] * 4
            call_count[0] += 1
            return SensorSample(fsr=fsr, led=0, t_ns=call_count[0] * 10_000_000)
        result = check_sync_check(read_sample=read_sample, duration_s=5.0, sample_hz=100.0)
        assert not result.passed
        assert "not toggling" in result.message

    def test_static_fsr_fails(self):
        """All FSR channels static (no pressure variation) → data not flowing."""
        from apps.collection.sensor_reader import SensorSample
        n, count = 500, [0]
        pattern = self._toggling_led_pattern(n)
        def read_sample():
            idx = count[0]; count[0] += 1
            return SensorSample(fsr=[400, 400, 400, 400], led=pattern[idx % n],
                                t_ns=idx * 10_000_000)
        result = check_sync_check(read_sample=read_sample, duration_s=5.0, sample_hz=100.0)
        assert not result.passed
        assert "data not flowing" in result.message

    def test_stuck_channel_fails(self):
        """A channel stuck at 0 throughout → sync_check fails."""
        from apps.collection.sensor_reader import SensorSample
        n, count = 500, [0]
        pattern = self._toggling_led_pattern(n)
        def read_sample():
            idx = count[0]; count[0] += 1
            fsr = [100 + idx % 200, 200 + idx % 200, 0, 300 + idx % 200]  # ch2 stuck at 0
            return SensorSample(fsr=fsr, led=pattern[idx % n], t_ns=idx * 10_000_000)
        result = check_sync_check(read_sample=read_sample, duration_s=5.0, sample_hz=100.0)
        assert not result.passed
        assert "stuck" in result.message

    def test_method_label_is_serial_led(self):
        """method detail always includes 'serial_LED_column'."""
        n = 500
        read_sample = self._make_sample_reader(self._toggling_led_pattern(n))
        result = check_sync_check(read_sample=read_sample, duration_s=5.0, sample_hz=100.0)
        assert "serial_LED_column" in result.details["method"]

    # ── camera ROI path (secondary) ───────────────────────────────────────

    def test_camera_led_roi_passes_when_brightness_correlates(self):
        """LED toggles in serial + camera brightness shows rising edges → passes.

        The camera brightness steps up sharply each time the LED turns ON,
        producing rising edges detectable by frame differencing.
        """
        n = 500
        pattern = self._toggling_led_pattern(n)
        read_sample = self._make_sample_reader(pattern)
        call_count = [0]

        def camera_with_led():
            idx = call_count[0]
            call_count[0] += 1
            led = pattern[idx % n]
            # Brightness high when LED ON (200), low when OFF (20)
            brightness = 200.0 if led else 20.0
            return {"valid": True, "confidence": 0.9, "landmarks": [0.1] * 63,
                    "led_brightness": brightness}

        roi = {"x": 10, "y": 10, "width": 20, "height": 20,
               "transition_threshold": 15.0}
        result = check_sync_check(
            read_sample=read_sample, get_camera_data=camera_with_led,
            duration_s=5.0, sample_hz=100.0, led_roi=roi,
        )
        assert result.passed, result.message
        assert result.details["led_roi_used"] is True
        assert "camera_LED_edge_detection" in result.details["method"]
        assert result.details["camera_rising_edges"] >= 2

    def test_camera_led_roi_fails_when_brightness_flat(self):
        """LED toggles in serial but camera brightness is flat (no edges) → fails."""
        n = 500
        read_sample = self._make_sample_reader(self._toggling_led_pattern(n))

        def camera_flat_led():
            return {"valid": True, "confidence": 0.9, "landmarks": [0.1] * 63,
                    "led_brightness": 128.0}

        roi = {"x": 10, "y": 10, "width": 20, "height": 20,
               "transition_threshold": 15.0}
        result = check_sync_check(
            read_sample=read_sample, get_camera_data=camera_flat_led,
            duration_s=5.0, sample_hz=100.0, led_roi=roi,
        )
        assert not result.passed
        assert "transitions not detected" in result.message
        assert "Calibrate" in result.message or "calibrate" in result.message
        assert result.details["camera_failed"] is True
        assert result.details["camera_rising_edges"] == 0

    def test_camera_led_roi_fails_when_brightness_uncorrelated(self):
        """LED toggles in serial but camera brightness drifts slowly (no sharp
        rising edges) → fails.  Slow ambient/auto-exposure drift produces
        small per-sample deltas that never exceed the transition threshold,
        so frame differencing correctly rejects it.
        """
        n = 500
        pattern = self._toggling_led_pattern(n)
        read_sample = self._make_sample_reader(pattern)
        call_count = [0]

        def camera_random_brightness():
            idx = call_count[0]
            call_count[0] += 1
            # Slow sinusoidal drift — wide range but tiny per-sample deltas
            # (max delta = 50 * 0.3 ≈ 15, right at threshold; use smaller
            # step to stay safely below).
            import math
            brightness = 100.0 + 50.0 * math.sin(idx * 0.1)
            return {"valid": True, "confidence": 0.9, "landmarks": [0.1] * 63,
                    "led_brightness": brightness}

        roi = {"x": 10, "y": 10, "width": 20, "height": 20,
               "transition_threshold": 15.0}
        result = check_sync_check(
            read_sample=read_sample, get_camera_data=camera_random_brightness,
            duration_s=5.0, sample_hz=100.0, led_roi=roi,
        )
        assert not result.passed, (
            f"Should fail — slow drift has no sharp rising edges. "
            f"rising_edges={result.details.get('camera_rising_edges')}"
        )
        assert result.details["camera_failed"] is True

    def test_passes_with_calibration_warning_when_no_roi(self):
        """Serial LED toggling + camera available but no ROI → passes with calibration warning."""
        n = 500
        read_sample = self._make_sample_reader(self._toggling_led_pattern(n))

        def camera_available():
            return {"valid": True, "confidence": 0.9, "landmarks": [0.1] * 63}

        result = check_sync_check(
            read_sample=read_sample, get_camera_data=camera_available,
            duration_s=5.0, sample_hz=100.0, led_roi=None,
        )
        assert result.passed, result.message
        assert result.details["needs_calibration"] is True
        assert "not calibrated" in result.message.lower()


# ── PrecollectResults properties ──────────────────────────────────────────────


class TestPrecollectResultsProperties:
    def test_n_passed_and_n_failed(self):
        """n_passed and n_failed count correctly; all_passed requires every result."""
        results = PrecollectResults()
        results.results.append(TestResult(name="a", passed=True))
        results.results.append(TestResult(name="b", passed=True))
        results.results.append(TestResult(name="c", passed=False))

        assert results.n_passed == 2
        assert results.n_failed == 1
        assert results.all_passed is False

    def test_all_passed_true_when_all_pass(self):
        results = PrecollectResults()
        results.results.append(TestResult(name="a", passed=True))
        results.results.append(TestResult(name="b", passed=True))
        assert results.all_passed is True
        assert results.n_failed == 0

    def test_empty_results_all_passed(self):
        """Empty result list: all_passed is vacuously True, counts are 0."""
        results = PrecollectResults()
        assert results.all_passed is True
        assert results.n_passed == 0
        assert results.n_failed == 0


# ── run_all_tests expected pass/fail pattern ─────────────────────────────────


class TestRunAllTestsExpectedPattern:
    def test_constant_reader_expected_failures(self):
        """With constant sensor data [300,400,500,600]:
        - response_linearity PASSES (warning-only) but reports a warning
        - single_dof_isolation FAILS: no modulation (range=0)
        - sync_check FAILS: check_sync_check requires range>5 to confirm
          data is flowing — constant data looks like a frozen/dead signal
        All other tests pass because mid-range stable values satisfy their
        specific criteria (no stuck channels, reasonable baseline, etc.)
        """
        results = run_all_tests(
            read_sensors=make_stable_reader([300, 400, 500, 600]),
            get_camera_data=make_camera_reader(),
        )
        assert len(results.results) == 8

        result_map = {r.name: r for r in results.results}
        # Hard failures: tests that require dynamic signal and still block collection
        expected_failures = {"single_dof_isolation", "sync_check"}
        for name in expected_failures:
            assert result_map[name].passed is False, \
                f"{name} should fail with constant static data"
        # response_linearity is warning-only: it passes but carries a warning
        assert result_map["response_linearity"].passed is True
        assert result_map["response_linearity"].details["warning"] is True
        # Everything else should pass
        for name, result in result_map.items():
            if name not in expected_failures and name != "response_linearity":
                assert result.passed is True, f"{name} should pass with stable mid-range data"


# ── _owner / flush_buffer wiring in run_all_tests_interactive ─────────────────


class TestInteractiveOwnerWiring:
    def test_flush_buffer_called_on_mock_reader_before_test(self):
        """run_all_tests_interactive calls flush_buffer() on the sensor reader
        before each test run when the reader exposes that method.

        MockSensorReader has flush_buffer but NOT _has_consumer (that is only
        on SerialSensorReader).  The _owner path checks hasattr guards so it
        only sets _has_consumer when present — but always calls flush_buffer
        when present.  This test verifies that basic wiring.
        """
        from apps.collection.sensor_reader import MockSensorReader

        sensor_reader = MockSensorReader(n_sensors=4, seed=1)
        sensor_reader.start()

        # Spy on flush_buffer
        flush_calls = [0]
        orig_flush = sensor_reader.flush_buffer
        def spy_flush():
            flush_calls[0] += 1
            orig_flush()
        sensor_reader.flush_buffer = spy_flush

        ready_event = threading.Event()
        stop_event = threading.Event()

        # Use the bound method so getattr(fn, '__self__') returns sensor_reader
        generator = run_all_tests_interactive(
            read_sensors=sensor_reader.read_sensors,
            get_camera_data=make_camera_reader(),
            ready_event=ready_event,
            stop_event=stop_event,
        )

        for event in generator:
            if isinstance(event, dict) and event.get("type") == "test_ready":
                ready_event.set()
            if isinstance(event, TestResult):
                stop_event.set()
                break

        sensor_reader.stop()

        assert flush_calls[0] >= 1, \
            "flush_buffer() must be called at least once before a test runs"

    def test_has_consumer_toggled_on_serial_reader(self):
        """For SerialSensorReader (which has _has_consumer), run_all_tests_interactive
        must set _has_consumer=False during ready-wait (preview mode) and
        _has_consumer=True before the actual test sampling begins.
        """
        from apps.collection.sensor_reader import SerialSensorReader

        # Build a fake serial reader without real hardware
        reader = SerialSensorReader(port="COM_FAKE", n_sensors=4)

        # Inject a fake _ser so start() succeeds without a real COM port
        class _FakeSerial:
            def readline(self):
                import time as _t; _t.sleep(0.001)
                return b"300,400,500,600,0\n"
            def reset_input_buffer(self): pass
            def close(self): pass

        reader._open_serial = lambda: setattr(reader, "_ser", _FakeSerial())
        reader.start()

        has_consumer_at_ready = [None]
        ready_event = threading.Event()
        stop_event = threading.Event()

        generator = run_all_tests_interactive(
            read_sensors=reader.read_sensors,
            get_camera_data=make_camera_reader(),
            ready_event=ready_event,
            stop_event=stop_event,
        )

        for event in generator:
            if isinstance(event, dict) and event.get("type") == "test_ready":
                # During ready-wait, _has_consumer must be False so preview can read
                has_consumer_at_ready[0] = reader._has_consumer
                ready_event.set()
            if isinstance(event, TestResult):
                stop_event.set()
                break

        reader.stop()

        assert has_consumer_at_ready[0] is False, \
            "_has_consumer must be False during ready-wait phase"
