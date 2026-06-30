"""Real-time quality monitoring for the collection session.

Tracks FSR sample rate, stuck sensors, zero readings, LED blink presence, and
camera hand-detection quality. Emits a ``QualityEvent`` only when the overall
level changes so the UI is not flooded with updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from apps.collection.config import QualityConfig, DEFAULT_QUALITY_CONFIG


@dataclass
class QualitySnapshot:
    """Quality metrics computed over a 1-second window."""

    fsr_hz: float = 0.0
    flat_pct: float = 0.0
    zero_pct: float = 0.0
    led_blinks: int = 0
    camera_valid_pct: float = 0.0
    camera_mean_confidence: float = 0.0
    per_sensor_flat_pct: list[float] = field(default_factory=list)
    per_sensor_zero_pct: list[float] = field(default_factory=list)


class QualityMonitor:
    """Track data quality in real time during a collection session.

    Parameters
    ----------
    n_sensors : int
        Number of FSR channels.
    target_hz : float
        Expected FSR sampling rate.
    camera_fps : float
        Expected camera frame rate.
    """

    def __init__(
        self,
        n_sensors: int = 4,
        target_hz: float = 100.0,
        camera_fps: float = 30.0,
        *,
        quality_config: QualityConfig | None = None,
    ) -> None:
        self.n_sensors = n_sensors
        self.target_hz = target_hz
        self.camera_fps = camera_fps

        cfg = quality_config if quality_config is not None else DEFAULT_QUALITY_CONFIG
        self._fsr_red_ratio = cfg.fsr_red_ratio
        self._fsr_yellow_ratio = cfg.fsr_yellow_ratio
        self._flat_red_pct = cfg.flat_red_pct
        self._flat_yellow_pct = cfg.flat_yellow_pct
        self._zero_red_pct = cfg.zero_red_pct
        self._zero_yellow_pct = cfg.zero_yellow_pct
        self._camera_red_pct = cfg.camera_red_pct
        self._camera_yellow_pct = cfg.camera_yellow_pct

        self._last_fsr: list[int] | None = None
        self._last_led = 0
        self._window_start_ns: int | None = None
        self._samples = 0
        self._flat = 0
        self._zero = 0
        self._per_sensor_flat: list[int] | None = None
        self._per_sensor_zero: list[int] | None = None
        self._led_transitions = 0
        self._camera_valid = 0
        self._camera_invalid = 0
        self._camera_confidence_sum = 0.0
        self._camera_count = 0
        self._last_level = "green"
        self._last_reason = ""

    def _current_level(self, snap: QualitySnapshot) -> tuple[str, list[str]]:
        """Return (level, reasons) for the current snapshot."""
        reasons: list[str] = []
        level = "green"

        if snap.fsr_hz < self.target_hz * self._fsr_red_ratio:
            reasons.append(f"FSR rate critically low ({snap.fsr_hz:.1f} Hz)")
            level = "red"
        elif snap.fsr_hz < self.target_hz * self._fsr_yellow_ratio:
            reasons.append(f"FSR rate low ({snap.fsr_hz:.1f} Hz)")
            if level == "green":
                level = "yellow"

        if snap.flat_pct > self._flat_red_pct:
            reasons.append(f"Sensors stuck ({snap.flat_pct:.0f}% flat)")
            level = "red"
        elif snap.flat_pct > self._flat_yellow_pct:
            reasons.append(f"Sensors often flat ({snap.flat_pct:.0f}%)")
            if level == "green":
                level = "yellow"

        if snap.zero_pct > self._zero_red_pct:
            reasons.append(f"Many zero readings ({snap.zero_pct:.0f}%)")
            level = "red"
        elif snap.zero_pct > self._zero_yellow_pct:
            reasons.append(f"Some zero readings ({snap.zero_pct:.0f}%)")
            if level == "green":
                level = "yellow"

        if snap.led_blinks == 0:
            reasons.append("LED not blinking")
            if level == "green":
                level = "yellow"

        if snap.camera_valid_pct < self._camera_red_pct:
            reasons.append("Camera not tracking hand")
            level = "red"
        elif snap.camera_valid_pct < self._camera_yellow_pct:
            reasons.append("Camera tracking weak")
            if level == "green":
                level = "yellow"

        return level, reasons

    def update(
        self,
        t_ns: int,
        fsr_values: list[int],
        led_state: int,
        camera_valid: bool,
        camera_confidence: float,
    ) -> tuple[str, str] | None:
        """Ingest one sample and return (level, reason) if the level changed.

        The monitor computes a 1-second rolling window based on the sample
        timestamps. The tuple is only returned when the level or reason changes
        to avoid UI spam.
        """
        if self._window_start_ns is None:
            self._window_start_ns = t_ns
        self._samples += 1

        if self._last_fsr is not None and all(
            v == self._last_fsr[i] for i, v in enumerate(fsr_values)
        ):
            self._flat += 1
        if any(v == 0 for v in fsr_values):
            self._zero += 1

        # Per-sensor flat / zero tracking
        if self._per_sensor_flat is None:
            self._per_sensor_flat = [0] * len(fsr_values)
        if self._per_sensor_zero is None:
            self._per_sensor_zero = [0] * len(fsr_values)
        if self._last_fsr is not None:
            for i, v in enumerate(fsr_values):
                if v == self._last_fsr[i]:
                    self._per_sensor_flat[i] += 1
                if v == 0:
                    self._per_sensor_zero[i] += 1
        self._last_fsr = list(fsr_values)

        if led_state != self._last_led:
            self._led_transitions += 1
        self._last_led = led_state

        if camera_valid:
            self._camera_valid += 1
        else:
            self._camera_invalid += 1
        self._camera_confidence_sum += camera_confidence
        self._camera_count += 1

        elapsed_ns = t_ns - self._window_start_ns
        if elapsed_ns < 1_000_000_000:
            return None

        elapsed_s = elapsed_ns / 1e9
        total_camera = self._camera_valid + self._camera_invalid
        snap = QualitySnapshot(
            fsr_hz=self._samples / elapsed_s,
            flat_pct=100.0 * self._flat / max(1, self._samples),
            zero_pct=100.0 * self._zero / max(1, self._samples),
            led_blinks=self._led_transitions,
            camera_valid_pct=100.0 * self._camera_valid / max(1, total_camera),
            camera_mean_confidence=(
                self._camera_confidence_sum / max(1, self._camera_count)
            ),
            per_sensor_flat_pct=[
                100.0 * c / max(1, self._samples) for c in (self._per_sensor_flat or [])
            ],
            per_sensor_zero_pct=[
                100.0 * c / max(1, self._samples) for c in (self._per_sensor_zero or [])
            ],
        )

        level, reasons = self._current_level(snap)
        reason = "; ".join(reasons) if reasons else "Quality good"

        # Reset window
        self._window_start_ns = t_ns
        self._samples = 0
        self._flat = 0
        self._zero = 0
        self._per_sensor_flat = None
        self._per_sensor_zero = None
        self._led_transitions = 0
        self._camera_valid = 0
        self._camera_invalid = 0
        self._camera_confidence_sum = 0.0
        self._camera_count = 0

        per_sensor = {
            "flat_pct": snap.per_sensor_flat_pct,
            "zero_pct": snap.per_sensor_zero_pct,
        }

        if level == self._last_level and reason == self._last_reason:
            return None
        self._last_level = level
        self._last_reason = reason
        return level, reason, per_sensor
