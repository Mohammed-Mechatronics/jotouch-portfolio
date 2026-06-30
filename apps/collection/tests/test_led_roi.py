"""Tests for apps/collection/led_roi.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.collection.led_roi import LedRoi, _order_points, _RoiSelector, calibrate
from apps.collection.camera import MockCameraReader


class TestOrderPoints:
    def test_top_left_to_bottom_right(self):
        assert _order_points(10, 20, 50, 80) == (10, 20, 40, 60)

    def test_bottom_right_to_top_left(self):
        assert _order_points(50, 80, 10, 20) == (10, 20, 40, 60)


class TestLedRoi:
    def test_round_trip(self, tmp_path: Path):
        roi = LedRoi(x=10, y=20, width=40, height=60, camera_index=1, image_width=640, image_height=480)
        path = tmp_path / "roi.json"
        roi.save(path)
        loaded = LedRoi.load(path)
        assert loaded == roi
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["normalized"]["x"] == 10 / 640

    def test_normalized(self):
        roi = LedRoi(x=100, y=50, width=50, height=50, image_width=640, image_height=480)
        norm = roi.to_normalized()
        assert norm["x"] == 100 / 640
        assert norm["y"] == 50 / 480
        assert norm["width"] == 50 / 640
        assert norm["height"] == 50 / 480


class TestRoiSelector:
    def test_roi_from_drag(self):
        selector = _RoiSelector()
        selector.x1, selector.y1 = 10, 20
        selector.x2, selector.y2 = 50, 80
        roi = selector.roi(0, 640, 480)
        assert roi is not None
        assert roi.x == 10
        assert roi.y == 20
        assert roi.width == 40
        assert roi.height == 60

    def test_roi_zero_size_is_none(self):
        selector = _RoiSelector()
        selector.x1 = selector.x2 = 100
        selector.y1 = selector.y2 = 100
        assert selector.roi(0, 640, 480) is None


class TestCalibrateWithExternalReader:
    def test_calibrate_uses_shared_reader_without_stopping_it(self, monkeypatch, tmp_path):
        """calibrate(camera_reader=reader) must NOT call start() or stop() on the
        shared reader — it must leave it running so the WS stream keeps working."""
        import cv2

        reader = MockCameraReader(seed=42)
        reader.start()

        start_calls = [0]
        stop_calls = [0]
        orig_start = reader.start
        orig_stop = reader.stop

        def spy_start():
            start_calls[0] += 1
            return orig_start()

        def spy_stop():
            stop_calls[0] += 1
            orig_stop()

        monkeypatch.setattr(reader, "start", spy_start)
        monkeypatch.setattr(reader, "stop", spy_stop)

        # Stub out cv2 GUI calls so the calibration loop runs non-interactively
        key_sequence = iter([ord("s"), ord("q")])

        class _FakeWindow:
            pass

        monkeypatch.setattr(cv2, "namedWindow", lambda *a, **kw: None)
        monkeypatch.setattr(cv2, "setWindowProperty", lambda *a, **kw: None)
        monkeypatch.setattr(cv2, "setMouseCallback", lambda *a, **kw: None)
        monkeypatch.setattr(cv2, "imshow", lambda *a, **kw: None)
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
        # Simulate: first key press 's' selects ROI (but selector has zero-size →
        # prints message), second 'q' quits.
        monkeypatch.setattr(cv2, "waitKey", lambda ms=1: next(key_sequence, ord("q")) & 0xFF)
        monkeypatch.setattr(cv2, "imdecode", lambda *a, **kw: __import__("numpy").zeros((480, 640, 3), dtype=__import__("numpy").uint8))

        roi_path = tmp_path / "roi.json"
        calibrate(output=roi_path, camera_reader=reader)

        # The shared reader must never have been started or stopped
        assert start_calls[0] == 0, "calibrate() must not call start() on a shared reader"
        assert stop_calls[0] == 0, "calibrate() must not call stop() on a shared reader"
        assert reader._running, "Shared reader must still be running after calibrate()"

        reader.stop()

    def test_calibrate_owns_reader_when_none_provided(self, monkeypatch, tmp_path):
        """When no camera_reader is given, calibrate() creates and stops its own reader."""
        import cv2
        import apps.collection.led_roi as led_roi_mod

        created = []
        stopped = []

        class _TrackingMock(MockCameraReader):
            def start(self):
                created.append(self)
                return super().start()
            def stop(self):
                stopped.append(self)
                super().stop()

        monkeypatch.setattr(led_roi_mod, "create_camera_reader",
                            lambda **kw: _TrackingMock(seed=1))
        monkeypatch.setattr(cv2, "namedWindow", lambda *a, **kw: None)
        monkeypatch.setattr(cv2, "setWindowProperty", lambda *a, **kw: None)
        monkeypatch.setattr(cv2, "setMouseCallback", lambda *a, **kw: None)
        monkeypatch.setattr(cv2, "imshow", lambda *a, **kw: None)
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
        monkeypatch.setattr(cv2, "waitKey", lambda ms=1: ord("q") & 0xFF)
        monkeypatch.setattr(cv2, "imdecode", lambda *a, **kw: __import__("numpy").zeros((480, 640, 3), dtype=__import__("numpy").uint8))

        calibrate(dry_run=True)

        assert len(created) == 1, "calibrate() must start its own reader when none provided"
        assert len(stopped) == 1, "calibrate() must stop its own reader when it owns it"
