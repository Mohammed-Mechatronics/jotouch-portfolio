"""Runtime configuration for the collection session.

Defaults are tuned for the 100 Hz FSR + 30 Hz camera setup used in the JoTouch
band. Values can be overridden at session start via the WebSocket ``start``
command or the CLI ``--quality-config`` / ``--test-config`` options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualityConfig:
    """Thresholds for the real-time QualityMonitor."""

    # FSR rate relative to target_hz
    fsr_red_ratio: float = 0.5      # < 50% of target -> red
    fsr_yellow_ratio: float = 0.8   # < 80% of target -> yellow

    # Percentage of samples where a sensor is stuck at the same value
    flat_red_pct: float = 30.0      # > 30% flat -> red
    flat_yellow_pct: float = 10.0   # > 10% flat -> yellow

    # Percentage of samples that are exactly zero
    zero_red_pct: float = 50.0      # > 50% zero -> red
    zero_yellow_pct: float = 20.0   # > 20% zero -> yellow

    # Percentage of camera frames with a valid hand detection
    camera_red_pct: float = 30.0    # < 30% valid -> red
    camera_yellow_pct: float = 70.0  # < 70% valid -> yellow

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "QualityConfig":
        """Build a config from a dict, ignoring unknown keys."""
        if data is None:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TestDurationConfig:
    """Durations (in seconds) for each pre-collection test.

    These defaults MUST match the ``duration_s`` values in ``TEST_META``
    (precollect.py).  The API always sends a ``TestDurationConfig`` (even
    when the UI doesn't provide one), so these values override TEST_META
    at runtime.  If you change a duration in TEST_META, change it here too.
    """

    dead_stuck_channels: float = 3.0
    channel_activation: float = 3.0
    camera_tracking: float = 3.0
    sync_check: float = 10.0
    baseline_stability: float = 5.0
    response_linearity: float = 6.0
    single_dof_isolation: float = 5.0
    creep_drift: float = 10.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TestDurationConfig":
        """Build a config from a dict, ignoring unknown keys."""
        if data is None:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_test_meta(self, base_meta: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Return a copy of ``base_meta`` with durations replaced by this config."""
        meta = {name: dict(info) for name, info in base_meta.items()}
        for name, duration in self.__dict__.items():
            if name in meta:
                meta[name]["duration_s"] = duration
        return meta


DEFAULT_QUALITY_CONFIG = QualityConfig()
DEFAULT_TEST_DURATION_CONFIG = TestDurationConfig()


@dataclass
class CameraTrackingConfig:
    """Camera + hand-tracking settings.

    Controls the MediaPipe HandLandmarker detection/tracking thresholds,
    camera resolution, and exposure.  Higher confidence thresholds = more
    accurate but may lose tracking when the hand moves fast or is partially
    occluded.  Lower thresholds = more responsive but may produce false
    detections.

    Fields
    ----------
    resolution : str
        Camera capture resolution.  Options: "640x480", "1280x720", "1920x1080".
        Higher resolution = more detail but slower processing.
    min_detection_confidence : float
        Minimum confidence for the palm detection model (0.0–1.0).
        Raise to reduce false detections; lower if the hand isn't detected.
    min_presence_confidence : float
        Minimum confidence for hand presence in the landmark model (0.0–1.0).
        If below this, the landmarker re-triggers palm detection.
    min_tracking_confidence : float
        Minimum IoU threshold for frame-to-frame tracking (0.0–1.0).
        Raise for stricter tracking; lower if tracking is lost frequently.
    auto_exposure : bool
        If True, let the camera auto-adjust exposure.  If False, use manual
        exposure (useful when ambient lighting is stable).
    """

    resolution: str = "640x480"
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    auto_exposure: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CameraTrackingConfig":
        """Build a config from a dict, ignoring unknown keys."""
        if data is None:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def width(self) -> int:
        return int(self.resolution.split("x")[0])

    @property
    def height(self) -> int:
        return int(self.resolution.split("x")[1])


DEFAULT_CAMERA_TRACKING_CONFIG = CameraTrackingConfig()
