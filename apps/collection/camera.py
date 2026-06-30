"""Camera capture + MediaPipe Hand landmark extraction for the BIDS collection app.

This module provides a ``CameraReader`` class that opens a webcam, runs
MediaPipe Hands, and returns full 21-landmark arrays (63 floats) suitable
for writing to ``_camera.csv`` and converting to joint angles.

The implementation is adapted from ``sync/sync_check.py`` (which only
extracted a scalar grip-closure metric) and extended to return the full
landmark array.

Usage (real camera)::

    reader = CameraReader(camera_index=0)
    reader.start()
    frame = reader.get_frame()  # dict with landmarks, valid, confidence, handedness
    reader.stop()

Usage (dry-run / mock)::

    reader = MockCameraReader()
    reader.start()
    frame = reader.get_frame()  # random landmarks for testing
"""

from __future__ import annotations

import importlib
import random
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apps.collection.timer import now_ns

# MediaPipe 0.10+ Tasks API is laid out under mediapipe/tasks/python, but
# the package name is registered as mediapipe.tasks. Use importlib to avoid
# the fragile attribute-import path on the aliased module.
_mp_tasks_core = importlib.import_module("mediapipe.tasks.python.core")
_mp_base_options = importlib.import_module("mediapipe.tasks.python.core.base_options")
_mp_hand_landmarker = importlib.import_module("mediapipe.tasks.python.vision.hand_landmarker")
_mp_vision_running_mode = importlib.import_module(
    "mediapipe.tasks.python.vision.core.vision_task_running_mode"
)
BaseOptions = _mp_base_options.BaseOptions
HandLandmarker = _mp_hand_landmarker.HandLandmarker
HandLandmarkerOptions = _mp_hand_landmarker.HandLandmarkerOptions
VisionTaskRunningMode = _mp_vision_running_mode.VisionTaskRunningMode

# Pinned MediaPipe hand landmarker model URL and expected SHA256.
# This ensures the model version is stable and verifiable.
_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_HAND_LANDMARKER_SHA256 = None  # Set when the model is bundled or verified


def _find_bundled_model() -> Path | None:
    """Return the path to a bundled model if it exists in the repo."""
    bundled = Path(__file__).resolve().parent.parent.parent / "models" / "hand_landmarker.task"
    if bundled.exists() and bundled.stat().st_size > 0:
        return bundled
    return None


def _ensure_hand_landmarker_model() -> Path | None:
    """Return the local path to the MediaPipe hand landmarker model.

    Resolution order:
      1. Bundled model in ``repo_root/models/hand_landmarker.task``.
      2. Cached model in ``~/.cache/jotouch/hand_landmarker.task``.
      3. Download from the pinned URL (with optional SHA256 verification).

    Returns None if the model cannot be obtained.
    """
    bundled = _find_bundled_model()
    if bundled is not None:
        return bundled

    cache_dir = Path.home() / ".cache" / "jotouch"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "hand_landmarker.task"
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    try:
        print(f"[CAM ] Downloading hand landmarker model to {model_path} ...")
        urllib.request.urlretrieve(_HAND_LANDMARKER_URL, str(model_path))
        print(f"[CAM ] Model downloaded ({model_path.stat().st_size} bytes)")
        if _HAND_LANDMARKER_SHA256 is not None:
            import hashlib

            digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if digest != _HAND_LANDMARKER_SHA256:
                print(f"[CAM ] ERROR: Model checksum mismatch ({digest}). Deleting.")
                model_path.unlink()
                return None
        return model_path
    except Exception as exc:
        print(f"[CAM ] ERROR: Could not download hand landmarker model: {exc}")
        return None


@dataclass
class CameraFrame:
    """One camera frame result.

    Attributes:
        t_ns: monotonic timestamp in nanoseconds.
        valid: whether a hand was detected.
        confidence: MediaPipe detection confidence (0-1).
        handedness: "Left" or "Right".
        landmarks: flat list of 63 floats (21 landmarks × x,y,z),
                   or None if no hand detected.
    """

    t_ns: int
    valid: bool = False
    confidence: float = 0.0
    handedness: str = "Right"
    landmarks: list[float] | None = None
    led_brightness: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_ns": self.t_ns,
            "valid": self.valid,
            "confidence": self.confidence,
            "handedness": self.handedness,
            "landmarks": self.landmarks,
            "led_brightness": self.led_brightness,
        }


class MockCameraReader:
    """Mock camera reader for dry-run mode.

    Returns random landmark values so the BIDS writer and joint-angle
    derivation can be tested without a real camera.  Simulates LED
    brightness matching the PRBS preamble / periodic blink pattern so
    the sync check can run in dry-run mode.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._running = False
        self._frame: CameraFrame | None = None
        self._lock = threading.Lock()
        self._session_start_ns: int | None = None

    def start(self) -> bool:
        self._running = True
        self._session_start_ns = now_ns()
        return True

    def stop(self) -> None:
        self._running = False

    def set_led_roi(self, roi: dict[str, int] | None) -> None:
        """No-op for mock (LED brightness is simulated regardless of ROI)."""
        pass

    def get_frame(self) -> dict[str, Any]:
        """Return a mock frame with random landmarks and simulated LED brightness."""
        t_ns = now_ns()
        # Generate plausible-looking normalized landmark coordinates
        landmarks = [self._rng.uniform(-0.5, 0.5) for _ in range(63)]
        # Simulate LED brightness from the PRBS/periodic pattern
        led_brightness: float | None = None
        if self._session_start_ns is not None:
            from apps.collection.prbs import led_state_ns
            led = led_state_ns(t_ns, self._session_start_ns)
            led_brightness = 255.0 if led else 0.0
        frame = CameraFrame(
            t_ns=t_ns,
            valid=True,
            confidence=self._rng.uniform(0.80, 0.98),
            handedness="Right",
            landmarks=landmarks,
            led_brightness=led_brightness,
        )
        return frame.to_dict()

    def get_jpeg(self) -> bytes | None:
        """Return a synthetic JPEG frame for dry-run preview."""
        import cv2
        import numpy as np

        width, height = 640, 480
        # Generate a plausible blank image with a timestamp overlay
        img = np.full((height, width, 3), (200, 200, 200), dtype=np.uint8)
        # Draw a moving rectangle so the preview looks alive
        x = int((time.monotonic() * 60) % (width - 100))
        cv2.rectangle(img, (x, 200), (x + 80, 280), (0, 120, 255), -1)
        ok, buf = cv2.imencode(".jpg", img)
        return bytes(buf) if ok else None


class CameraReader:
    """Real camera reader using OpenCV + MediaPipe 0.10+ Tasks HandLandmarker.

    Opens a webcam, runs hand detection on each frame, and stores the latest
    result. ``get_frame()`` returns the most recent frame without blocking.

    Args:
        camera_index: OpenCV camera index (0 = default camera).
        backend: OpenCV backend ("any", "msmf", "dshow").
        min_detection_confidence: Minimum confidence for hand detection.
        min_presence_confidence: Minimum confidence for hand presence.
        min_tracking_confidence: Minimum confidence for tracking.
    """

    def __init__(
        self,
        camera_index: int = 0,
        backend: str = "any",
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        resolution: str = "640x480",
        auto_exposure: bool = True,
    ) -> None:
        self.camera_index = camera_index
        self.backend = backend
        # NOTE: model_complexity is NOT used by MediaPipe Tasks API.
        # The legacy mp.solutions.hands API had 3 variants:
        #   0 = Lite  (1.0M params, MSE=11.83, ~6.6ms)
        #   1 = Full  (1.98M params, MSE=10.05, ~16.1ms)
        #   2 = Heavy (4.02M params, MSE=9.82, ~36.9ms)
        # The current Tasks API (HandLandmarker) only ships one model
        # variant ("full", float16, 192x192/224x224 input).  There is no
        # model_complexity parameter in HandLandmarkerOptions.
        # To use a different model, you would need to download a different
        # .task file (Google does not currently provide lite/heavy .task
        # files) or switch to the deprecated legacy API.
        # This field is kept for API compatibility but has no effect.
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.resolution = resolution
        self.auto_exposure = auto_exposure

        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: CameraFrame | None = None
        self._latest_image: Any = None
        self._latest_jpeg: bytes | None = None  # pre-encoded in capture thread
        self._cap = None
        self._landmarker: Any = None
        self._led_roi: dict[str, int] | None = None
        # WS-3: LIVE_STREAM mode for non-blocking hand detection
        self._running_mode = VisionTaskRunningMode.LIVE_STREAM
        # WS-3: Frame timing for fps measurement (rolling 60-frame window)
        self._frame_times: deque = deque(maxlen=60)

    def start(self) -> bool:
        """Open camera, load MediaPipe model, and start the capture thread.

        Returns:
            True if camera and model are ready, False otherwise.
        """
        try:
            import cv2
            import mediapipe as mp
        except ImportError as e:
            print(f"[CAM ] ERROR: {e}. Install opencv-python and mediapipe.")
            return False

        model_path = _ensure_hand_landmarker_model()
        if model_path is None:
            print("[CAM ] ERROR: Hand landmarker model unavailable")
            return False

        backend_map = {
            "any": cv2.CAP_ANY,
            "msmf": cv2.CAP_MSMF,
            "dshow": cv2.CAP_DSHOW,
        }
        cv_backend = backend_map.get(self.backend.lower(), cv2.CAP_ANY)
        self._cap = cv2.VideoCapture(self.camera_index, cv_backend)
        if not self._cap.isOpened():
            print(f"[CAM ] ERROR: Cannot open camera index {self.camera_index}")
            return False

        # Force MJPG — works around OpenCV 4.13 crashes with virtual cameras
        try:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        # Set resolution from config.  Higher resolutions give more detail
        # for small hand features but increase processing latency.
        try:
            res_w, res_h = int(self.resolution.split("x")[0]), int(self.resolution.split("x")[1])
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)
        except Exception:
            # Fall back to 640×480 if the resolution string is malformed
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Target 30 fps.  MUST be set BEFORE the first read()/grab() — on
        # Windows MSMF, setting CAP_PROP_FPS after a grab caches a wrong
        # value (OpenCV issue #26250).  This is only a hint; the driver may
        # pick a different rate.  The actual rate is measured by get_fps().
        try:
            self._cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass

        # Auto-exposure: 0.25 = manual, 0.75 = auto (OpenCV convention)
        try:
            if self.auto_exposure:
                self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            else:
                self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        except Exception:
            pass

        # Warm-up: skip placeholder/malformed frames from virtual cameras
        for _ in range(20):
            try:
                ok, warm = self._cap.read()
            except cv2.error:
                ok, warm = False, None
            if ok and warm is not None and float(warm.mean()) > 5.0:
                break
            time.sleep(0.05)

        try:
            base_options = BaseOptions(model_asset_path=str(model_path))
            options = HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                running_mode=VisionTaskRunningMode.LIVE_STREAM,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                result_callback=self._on_hand_result,
            )
            self._landmarker = HandLandmarker.create_from_options(options)
        except Exception as exc:
            print(f"[CAM ] ERROR: Could not create MediaPipe hand landmarker: {exc}")
            self._cap.release()
            self._cap = None
            return False

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[CAM ] Camera open (index {self.camera_index}, backend={self.backend})")
        return True

    # ── MediaPipe hand-connection pairs (21 landmarks, standard topology) ──
    _HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),         # thumb
        (0,5),(5,6),(6,7),(7,8),          # index
        (0,9),(9,10),(10,11),(11,12),     # middle
        (0,13),(13,14),(14,15),(15,16),   # ring
        (0,17),(17,18),(18,19),(19,20),   # pinky
        (5,9),(9,13),(13,17),             # palm
    ]

    def _draw_landmarks(self, frame, landmarks_flat: list[float]) -> None:
        """Draw 21 hand landmarks + connections on a BGR frame in-place."""
        import cv2
        h, w = frame.shape[:2]
        pts = []
        for i in range(21):
            x = int(landmarks_flat[i * 3] * w)
            y = int(landmarks_flat[i * 3 + 1] * h)
            pts.append((x, y))
        # Connections
        for a, b in self._HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 220, 100), 2, cv2.LINE_AA)
        # Landmark dots
        for i, (x, y) in enumerate(pts):
            r = 5 if i in (0, 4, 8, 12, 16, 20) else 3
            cv2.circle(frame, (x, y), r, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), r, (0, 160, 80), 1, cv2.LINE_AA)

    def _on_hand_result(self, result: Any, image: Any, timestamp_ms: int) -> None:
        """LIVE_STREAM result callback — called asynchronously by MediaPipe
        when hand detection completes for a frame.

        Updates ``self._latest`` with the detected landmarks. This is
        non-blocking: the capture loop calls ``detect_async()`` which returns
        immediately, and this callback fires when the model is done.

        The landmarks may lag the displayed JPEG by 1 frame (~33ms at 30fps),
        which is visually imperceptible and within the 100ms PRBS chip duration.
        """
        t_ns = int(timestamp_ms) * 1_000_000
        flat: list[float] = []
        if result.hand_landmarks and result.handedness:
            lms = result.hand_landmarks[0]
            handedness = result.handedness[0][0].category_name
            flat = [coord for lm in lms for coord in (lm.x, lm.y, lm.z)]
            confidence = result.handedness[0][0].score
            cam_frame = CameraFrame(
                t_ns=t_ns,
                valid=True,
                confidence=float(confidence),
                handedness=handedness,
                landmarks=flat,
            )
        else:
            cam_frame = CameraFrame(t_ns=t_ns, valid=False)

        with self._lock:
            self._latest = cam_frame

    def get_fps(self) -> float:
        """Return the measured capture FPS based on recent frame intervals.

        Uses a rolling 60-frame window. Returns 0.0 if fewer than 2 frames
        have been recorded.
        """
        with self._lock:
            times = list(self._frame_times)
        if len(times) < 2:
            return 0.0
        elapsed = times[-1] - times[0]
        if elapsed <= 0:
            return 0.0
        return len(times) / elapsed

    def _reopen_cap(self) -> bool:
        """Try to reopen the VideoCapture after it was lost (e.g. another process
        briefly stole the device).  Called from the capture loop on the reader
        thread — never from the event loop."""
        import cv2
        backend_map = {"any": cv2.CAP_ANY, "msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}
        cv_backend = backend_map.get(self.backend.lower(), cv2.CAP_ANY)
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        try:
            cap = cv2.VideoCapture(self.camera_index, cv_backend)
            if not cap.isOpened():
                return False
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap = cap
            print(f"[CAM ] Reconnected VideoCapture (index {self.camera_index})")
            return True
        except Exception as exc:
            print(f"[CAM ] Reconnect failed: {exc}")
            return False

    def _capture_loop(self) -> None:
        """Background thread: read frames, run MediaPipe, encode JPEG with overlay.

        WS-3: Uses LIVE_STREAM mode with detect_async() (non-blocking). The
        hand detection result arrives via _on_hand_result callback. The JPEG
        is encoded here using the latest available landmarks (may lag by 1
        frame, which is visually imperceptible).
        """
        import cv2
        import mediapipe as mp

        _consecutive_fails = 0
        _REOPEN_THRESHOLD = 30  # ~150 ms of failures at 5ms/sleep before trying reopen
        _last_timestamp_ms = 0  # WS-3: ensure strictly increasing timestamps for LIVE_STREAM

        while not self._stop_event.is_set():
            try:
                ok, frame = self._cap.read()
            except cv2.error:
                ok, frame = False, None
            t_ns = now_ns()
            if not ok or frame is None:
                _consecutive_fails += 1
                if _consecutive_fails >= _REOPEN_THRESHOLD:
                    print(f"[CAM ] {_consecutive_fails} consecutive read failures — attempting reopen")
                    if self._reopen_cap():
                        _consecutive_fails = 0
                    else:
                        # Back off before next attempt
                        time.sleep(1.0)
                        _consecutive_fails = 0
                else:
                    time.sleep(0.005)
                continue
            _consecutive_fails = 0

            # WS-3: Track frame time for fps measurement
            now_mono = time.monotonic()
            with self._lock:
                self._frame_times.append(now_mono)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = t_ns // 1_000_000
            # WS-3: LIVE_STREAM requires strictly monotonically increasing
            # timestamps. If two frames arrive within the same millisecond,
            # increment by 1 to avoid ValueError.
            if timestamp_ms <= _last_timestamp_ms:
                timestamp_ms = _last_timestamp_ms + 1
            _last_timestamp_ms = timestamp_ms
            # WS-3: detect_async is non-blocking — returns immediately.
            # Results arrive via _on_hand_result callback.
            self._landmarker.detect_async(mp_image, timestamp_ms)

            # Get the latest landmarks (may be from the previous frame —
            # 1 frame lag is acceptable for visualization)
            with self._lock:
                latest_frame = self._latest
            flat = latest_frame.landmarks if (latest_frame and latest_frame.landmarks) else []

            # Draw landmarks on a copy of the frame and encode to JPEG here,
            # in the background thread.  This prevents the asyncio event loop
            # from blocking on JPEG encoding, which causes the UI to feel slow.
            vis = frame.copy()
            if flat:
                self._draw_landmarks(vis, flat)

            # Encode JPEG at quality 80 for web UI — lighter encoding keeps
            # the capture thread responsive so it can sustain ~30 fps.
            # Landmarks are now sent separately as JSON for client-side rendering.
            ok_enc, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg = bytes(buf) if ok_enc else None

            with self._lock:
                self._latest_image = frame
                self._latest_jpeg = jpeg

    def set_led_roi(self, roi: dict[str, int] | None) -> None:
        """Set a region of interest for LED brightness extraction.

        roi should be a dict with keys: x, y, width, height (pixel coords).
        """
        with self._lock:
            self._led_roi = roi

    def get_jpeg(self) -> bytes | None:
        """Return the latest camera frame as a JPEG byte string (with landmarks).

        The JPEG is pre-encoded by the capture thread so this call never blocks
        on encoding — it just returns the cached bytes under a short lock.
        """
        with self._lock:
            return self._latest_jpeg

    def get_landmarks(self) -> list[float] | None:
        """Return the latest hand landmarks (21 × 3 = 63 floats) if valid.

        Returns None if no hand is detected.
        """
        with self._lock:
            if self._latest and self._latest.valid and self._latest.landmarks:
                return self._latest.landmarks
            return None

    def get_frame(self) -> dict[str, Any]:
        """Return the most recent camera frame (non-blocking).

        Returns:
            Dict with keys: t_ns, valid, confidence, handedness, landmarks,
            led_brightness. If no frame has been captured yet, returns an
            invalid frame.
        """
        with self._lock:
            frame = self._latest
            image = self._latest_image
            roi = self._led_roi
        if frame is None:
            return CameraFrame(t_ns=now_ns()).to_dict()
        d = frame.to_dict()
        if roi is not None and image is not None:
            d["led_brightness"] = self._compute_roi_brightness(image, roi)
        # WS-4: Include ROI in the frame dict so the WS metadata channel
        # can send it to the frontend for overlay rendering.
        d["roi"] = roi
        return d

    @staticmethod
    def _compute_roi_brightness(image, roi: dict[str, int]) -> float:
        """Compute mean brightness in the ROI of a BGR image."""
        import numpy as np

        h, w = image.shape[:2]
        x = max(0, min(roi["x"], w - 1))
        y = max(0, min(roi["y"], h - 1))
        width = max(1, min(roi["width"], w - x))
        height = max(1, min(roi["height"], h - y))
        roi_img = image[y:y + height, x:x + width]
        if roi_img.size == 0:
            return 0.0
        # Use value channel (max across BGR) as a simple brightness proxy
        return float(np.max(roi_img, axis=2).mean())

    def stop(self) -> None:
        """Stop the capture thread and release camera resources."""
        self._stop_event.set()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._landmarker = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        print("[CAM ] Camera released.")


def create_camera_reader(
    dry_run: bool = False,
    camera_index: int = 0,
    backend: str = "any",
    *,
    resolution: str = "640x480",
    min_detection_confidence: float = 0.5,
    min_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    auto_exposure: bool = True,
) -> CameraReader | MockCameraReader:
    """Factory: create a real or mock camera reader.

    Args:
        dry_run: If True, return a MockCameraReader.
        camera_index: Camera index for real reader.
        backend: OpenCV backend for real reader.
        resolution: Capture resolution (e.g. "640x480", "1280x720").
        min_detection_confidence: Palm detection confidence threshold (0–1).
        min_presence_confidence: Hand presence confidence threshold (0–1).
        min_tracking_confidence: Frame-to-frame tracking IoU threshold (0–1).
        auto_exposure: If True, let the camera auto-adjust exposure.

    Returns:
        CameraReader (real) or MockCameraReader (dry-run).
    """
    if dry_run:
        return MockCameraReader()
    return CameraReader(
        camera_index=camera_index,
        backend=backend,
        resolution=resolution,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        auto_exposure=auto_exposure,
    )
