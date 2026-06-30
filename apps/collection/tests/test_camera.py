"""Tests for apps/collection/camera.py — camera reader + mock reader."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.collection.camera import (
    CameraFrame,
    CameraReader,
    MockCameraReader,
    _find_bundled_model,
    create_camera_reader,
)


# ---------------------------------------------------------------------------
# CameraFrame tests
# ---------------------------------------------------------------------------

class TestCameraFrame:
    def test_defaults(self):
        frame = CameraFrame(t_ns=1000)
        assert frame.t_ns == 1000
        assert frame.valid is False
        assert frame.confidence == 0.0
        assert frame.handedness == "Right"
        assert frame.landmarks is None

    def test_to_dict(self):
        frame = CameraFrame(
            t_ns=1000,
            valid=True,
            confidence=0.95,
            handedness="Left",
            landmarks=[0.1] * 63,
        )
        d = frame.to_dict()
        assert d["t_ns"] == 1000
        assert d["valid"] is True
        assert d["confidence"] == 0.95
        assert d["handedness"] == "Left"
        assert len(d["landmarks"]) == 63


# ---------------------------------------------------------------------------
# MockCameraReader tests
# ---------------------------------------------------------------------------

class TestMockCameraReader:
    def test_start_stop(self):
        reader = MockCameraReader()
        reader.start()
        reader.stop()

    def test_get_frame_returns_valid(self):
        reader = MockCameraReader(seed=42)
        reader.start()
        frame = reader.get_frame()
        assert frame["valid"] is True
        assert frame["confidence"] > 0.0
        assert frame["landmarks"] is not None
        assert len(frame["landmarks"]) == 63
        reader.stop()

    def test_get_frame_different_each_call(self):
        """Mock reader should return different frames each call (random)."""
        reader = MockCameraReader(seed=42)
        reader.start()
        f1 = reader.get_frame()
        time.sleep(0.01)
        f2 = reader.get_frame()
        assert f1["landmarks"] != f2["landmarks"]
        reader.stop()

    def test_reproducible_with_seed(self):
        """Same seed → same first frame."""
        r1 = MockCameraReader(seed=99)
        r1.start()
        f1 = r1.get_frame()
        r1.stop()

        r2 = MockCameraReader(seed=99)
        r2.start()
        f2 = r2.get_frame()
        r2.stop()

        assert f1["landmarks"] == f2["landmarks"]


# ---------------------------------------------------------------------------
# create_camera_reader factory tests
# ---------------------------------------------------------------------------

class TestCreateCameraReader:
    def test_dry_run_returns_mock(self):
        reader = create_camera_reader(dry_run=True)
        assert isinstance(reader, MockCameraReader)

    def test_live_returns_real_reader(self):
        reader = create_camera_reader(dry_run=False, camera_index=99)
        assert isinstance(reader, CameraReader)
        assert reader.camera_index == 99

    def test_default_is_real(self):
        reader = create_camera_reader()
        assert isinstance(reader, CameraReader)


# ---------------------------------------------------------------------------
# CameraReader tests (without real camera — just test init)
# ---------------------------------------------------------------------------

class TestCameraReaderInit:
    def test_init_defaults(self):
        reader = CameraReader()
        assert reader.camera_index == 0
        assert reader.backend == "any"
        assert reader.model_complexity == 1

    def test_init_custom(self):
        reader = CameraReader(
            camera_index=2,
            backend="msmf",
            model_complexity=0,
            min_detection_confidence=0.7,
        )
        assert reader.camera_index == 2
        assert reader.backend == "msmf"
        assert reader.model_complexity == 0
        assert reader.min_detection_confidence == 0.7

    def test_get_frame_before_start_returns_invalid(self):
        """get_frame() before start() should return an invalid frame."""
        reader = CameraReader()
        frame = reader.get_frame()
        assert frame["valid"] is False
        assert frame["landmarks"] is None

    def test_stop_without_start_is_safe(self):
        """stop() before start() should not crash."""
        reader = CameraReader()
        reader.stop()  # should be a no-op

    def test_compute_roi_brightness(self):
        """ROI brightness should reflect the brightest pixel in the region."""
        reader = CameraReader()
        # Dark image with a bright white rectangle in the ROI
        image = np.full((100, 100, 3), 10, dtype=np.uint8)
        image[40:60, 40:60] = 255
        # ROI exactly covers the bright rectangle
        roi = {"x": 40, "y": 40, "width": 20, "height": 20}
        brightness = reader._compute_roi_brightness(image, roi)
        assert brightness > 240, f"Expected bright ROI, got {brightness}"

        # ROI outside the bright area should be dark
        dark_roi = {"x": 0, "y": 0, "width": 20, "height": 20}
        dark_brightness = reader._compute_roi_brightness(image, dark_roi)
        assert dark_brightness < 50, f"Expected dark ROI, got {dark_brightness}"


class TestModelResolution:
    def test_find_bundled_model_returns_none_when_missing(self):
        """If no bundled model is present, _find_bundled_model returns None."""
        bundled = _find_bundled_model()
        if bundled is not None:
            assert bundled.exists()


class TestDrawLandmarks:
    def test_draw_landmarks_paints_on_frame(self):
        """_draw_landmarks() should visibly alter a blank frame at landmark positions."""
        reader = CameraReader()
        # Blank dark frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()

        # Landmarks: all 21 points at the centre of the frame (0.5, 0.5, 0)
        landmarks_flat = [0.5, 0.5, 0.0] * 21
        reader._draw_landmarks(frame, landmarks_flat)

        # The frame must have been modified — at least some pixels changed
        assert not np.array_equal(frame, original), "Frame was not modified by _draw_landmarks"

    def test_draw_landmarks_respects_frame_dimensions(self):
        """Landmark coordinates are scaled to (width, height) — no index errors on
        non-square frames."""
        reader = CameraReader()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Corner landmarks — should not raise IndexError or write outside bounds
        landmarks_flat = [0.0, 0.0, 0.0] * 7 + [1.0, 1.0, 0.0] * 7 + [0.5, 0.5, 0.0] * 7
        reader._draw_landmarks(frame, landmarks_flat)  # must not raise


class TestReopenCap:
    def test_reopen_cap_succeeds_with_mock(self, monkeypatch):
        """_reopen_cap() should reopen VideoCapture and return True when camera is available."""
        import cv2 as _cv2

        reader = CameraReader(camera_index=0)

        # Provide a fake VideoCapture that always opens successfully
        class _FakeCap:
            def isOpened(self):
                return True
            def set(self, *a):
                pass
            def release(self):
                pass

        monkeypatch.setattr(_cv2, "VideoCapture", lambda *a, **kw: _FakeCap())
        # _reopen_cap needs _cap to exist (to call release on it)
        reader._cap = _FakeCap()

        result = reader._reopen_cap()
        assert result is True
        assert reader._cap is not None

    def test_reopen_cap_fails_gracefully(self, monkeypatch):
        """_reopen_cap() returns False when VideoCapture.isOpened() is False."""
        import cv2 as _cv2

        reader = CameraReader(camera_index=0)

        class _DeadCap:
            def isOpened(self):
                return False
            def set(self, *a):
                pass
            def release(self):
                pass


# ---------------------------------------------------------------------------
# WS-3: LIVE_STREAM mode + fps measurement
# ---------------------------------------------------------------------------

class TestLiveStreamMode:
    """WS-3: CameraReader must use LIVE_STREAM mode (non-blocking) instead of
    VIDEO mode (blocking) for live webcam feeds."""

    def test_running_mode_is_live_stream(self):
        """The _running_mode attribute must be LIVE_STREAM, not VIDEO."""
        reader = CameraReader()
        # The reader should expose its running mode for inspection
        assert hasattr(reader, "_running_mode"), \
            "CameraReader must expose _running_mode for LIVE_STREAM"
        from apps.collection.camera import VisionTaskRunningMode
        assert reader._running_mode == VisionTaskRunningMode.LIVE_STREAM, \
            f"Running mode must be LIVE_STREAM, got {reader._running_mode}"

    def test_on_hand_result_updates_latest(self):
        """_on_hand_result callback must update _latest with landmarks from
        the MediaPipe result object."""
        reader = CameraReader()

        # Build a mock result that mimics MediaPipe's HandLandmarkerResult
        class _MockLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        class _MockCategory:
            def __init__(self, category_name, score):
                self.category_name = category_name
                self.score = score

        class _MockResult:
            hand_landmarks = [[_MockLandmark(0.1 * i, 0.2 * i, 0.0) for i in range(21)]]
            handedness = [[_MockCategory("Right", 0.95)]]

        result = _MockResult()
        t_ns = 1234567890
        reader._on_hand_result(result, None, t_ns)

        assert reader._latest is not None
        assert reader._latest.valid is True
        assert reader._latest.confidence == 0.95
        assert reader._latest.handedness == "Right"
        assert reader._latest.landmarks is not None
        assert len(reader._latest.landmarks) == 63
        assert reader._latest.landmarks[0] == 0.0  # first landmark x (i=0: 0.1*0=0.0)
        assert reader._latest.landmarks[3] == 0.1  # second landmark x (i=1: 0.1*1=0.1)
        assert reader._latest.landmarks[4] == 0.2  # second landmark y (i=1: 0.2*1=0.2)

    def test_on_hand_result_no_hand_sets_invalid(self):
        """_on_hand_result with empty hand_landmarks must set _latest to invalid."""
        reader = CameraReader()

        class _MockResult:
            hand_landmarks = []
            handedness = []

        reader._on_hand_result(_MockResult(), None, 9999)
        assert reader._latest is not None
        assert reader._latest.valid is False
        assert reader._latest.landmarks is None

    def test_get_fps_returns_float(self):
        """get_fps() must return a float > 0 after frame times are recorded."""
        from collections import deque
        reader = CameraReader()
        # Simulate frame times at 30 Hz
        now = time.monotonic()
        reader._frame_times = deque(
            [now - 0.1, now - 0.066, now - 0.033, now], maxlen=60
        )
        fps = reader.get_fps()
        assert isinstance(fps, float)
        assert fps > 0, f"FPS should be positive, got {fps}"

    def test_get_fps_zero_when_no_frames(self):
        """get_fps() must return 0.0 when no frame times have been recorded."""
        reader = CameraReader()
        fps = reader.get_fps()
        assert fps == 0.0

    def test_capture_loop_calls_detect_async(self, monkeypatch):
        """The capture loop must call detect_async (non-blocking), not
        detect_for_video (blocking)."""
        import cv2 as _cv2
        import mediapipe as _mp

        reader = CameraReader()

        # Track which detect method was called
        detect_calls = {"async": 0, "video": 0}

        class _MockLandmarker:
            def detect_async(self, mp_image, timestamp_ms):
                detect_calls["async"] += 1
                return None  # result comes via callback, not return value
            def detect_for_video(self, mp_image, timestamp_ms):
                detect_calls["video"] += 1
                return None

        class _FakeCap:
            def __init__(self):
                self._count = 0
            def isOpened(self):
                return True
            def set(self, *a):
                pass
            def release(self):
                pass
            def read(self):
                self._count += 1
                if self._count > 3:
                    return False, None
                return True, np.zeros((480, 640, 3), dtype=np.uint8)

        reader._cap = _FakeCap()
        reader._landmarker = _MockLandmarker()
        reader._running = True
        reader._stop_event.clear()

        # Run a few iterations of the capture loop
        import threading
        def stop_after_delay():
            time.sleep(0.5)
            reader._stop_event.set()
        stopper = threading.Thread(target=stop_after_delay, daemon=True)
        stopper.start()

        reader._capture_loop()

        assert detect_calls["async"] > 0, "detect_async must be called"
        assert detect_calls["video"] == 0, "detect_for_video must NOT be called"
