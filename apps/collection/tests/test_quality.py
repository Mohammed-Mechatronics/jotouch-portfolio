"""Tests for apps/collection/quality.py — real-time quality monitoring."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.collection.config import QualityConfig
from apps.collection.quality import QualityMonitor


class TestQualityMonitor:
    def test_initial_window_returns_none(self):
        monitor = QualityMonitor(n_sensors=4, target_hz=100.0)
        for i in range(10):
            result = monitor.update(
                t_ns=i * 10_000_000,
                fsr_values=[100, 200, 300, 400],
                led_state=i % 2,
                camera_valid=True,
                camera_confidence=0.9,
            )
        assert result is None

    def test_low_rate_yellow(self):
        monitor = QualityMonitor(n_sensors=4, target_hz=100.0)
        # Feed 80 samples over ~1.01 seconds => ~79 Hz (between 50% and 80% threshold)
        for i in range(80):
            result = monitor.update(
                t_ns=i * 12_658_228,
                fsr_values=[100 + i, 200 + i, 300 + i, 400 + i],
                led_state=i % 2,
                camera_valid=True,
                camera_confidence=0.9,
            )
        assert result is not None
        level, reason, per_sensor = result
        assert level == "yellow"
        assert "FSR rate" in reason

    def test_stuck_sensor_red(self):
        monitor = QualityMonitor(n_sensors=4, target_hz=100.0)
        # Feed 101 identical samples spanning 1 second
        for i in range(101):
            result = monitor.update(
                t_ns=i * 10_000_000,
                fsr_values=[100, 200, 300, 400],
                led_state=0,
                camera_valid=True,
                camera_confidence=0.9,
            )
        assert result is not None
        level, reason, per_sensor = result
        assert level == "red"
        assert "stuck" in reason.lower()

    def test_no_led_blink_yellow(self):
        monitor = QualityMonitor(n_sensors=4, target_hz=100.0)
        # Feed samples at 100 Hz with LED always off and changing FSR
        for i in range(101):
            result = monitor.update(
                t_ns=i * 10_000_000,
                fsr_values=[100 + i, 200 + i, 300 + i, 400 + i],
                led_state=0,
                camera_valid=True,
                camera_confidence=0.9,
            )
        assert result is not None
        level, reason, per_sensor = result
        assert level == "yellow"
        assert "LED" in reason

    def test_poor_camera_tracking_yellow(self):
        monitor = QualityMonitor(n_sensors=4, target_hz=100.0)
        # 50% valid frames is between 30% (red) and 70% (yellow) thresholds
        for i in range(101):
            result = monitor.update(
                t_ns=i * 10_000_000,
                fsr_values=[100 + i, 200 + i, 300 + i, 400 + i],
                led_state=i % 2,
                camera_valid=(i % 2 == 0),
                camera_confidence=0.9 if (i % 2 == 0) else 0.0,
            )
        assert result is not None
        level, reason, per_sensor = result
        assert level == "yellow"
        assert "Camera" in reason

    def test_good_quality_no_event(self):
        monitor = QualityMonitor(n_sensors=4, target_hz=100.0)
        # First window should report green
        for i in range(101):
            result = monitor.update(
                t_ns=i * 10_000_000,
                fsr_values=[100 + i, 200 + i, 300 + i, 400 + i],
                led_state=i % 2,
                camera_valid=True,
                camera_confidence=0.9,
            )
        assert result is not None
        level, reason, per_sensor = result
        assert level == "green"
        assert len(per_sensor["flat_pct"]) == 4
        assert len(per_sensor["zero_pct"]) == 4

        # Second identical window should not emit a duplicate event
        for i in range(101, 202):
            result2 = monitor.update(
                t_ns=i * 10_000_000,
                fsr_values=[100 + i, 200 + i, 300 + i, 400 + i],
                led_state=i % 2,
                camera_valid=True,
                camera_confidence=0.9,
            )
        assert result2 is None

    def test_custom_quality_config(self):
        # With relaxed thresholds, a rate that would normally be yellow stays green
        cfg = QualityConfig(fsr_yellow_ratio=0.75)
        monitor = QualityMonitor(n_sensors=4, target_hz=100.0, quality_config=cfg)
        # Feed ~79 Hz, which is below the default 0.8 threshold but above the custom 0.75 threshold
        for i in range(80):
            result = monitor.update(
                t_ns=i * 12_658_228,
                fsr_values=[100 + i, 200 + i, 300 + i, 400 + i],
                led_state=i % 2,
                camera_valid=True,
                camera_confidence=0.9,
            )
        assert result is not None
        level, reason, per_sensor = result
        assert level == "green"

    def test_per_sensor_zero_reporting(self):
        """Per-sensor zero percentages reflect which channel is dead."""
        monitor = QualityMonitor(n_sensors=4, target_hz=100.0)
        # Channel 0 always zero; others vary
        for i in range(101):
            result = monitor.update(
                t_ns=i * 10_000_000,
                fsr_values=[0, 100 + i, 200 + i, 300 + i],
                led_state=i % 2,
                camera_valid=True,
                camera_confidence=0.9,
            )
        assert result is not None
        level, reason, per_sensor = result
        zero_pct = per_sensor["zero_pct"]
        assert zero_pct[0] > 95
        assert zero_pct[1] < 5
        assert zero_pct[2] < 5
        assert zero_pct[3] < 5
