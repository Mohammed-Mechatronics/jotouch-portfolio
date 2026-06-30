"""FastAPI backend for the JoTouch collection web UI.

Endpoints:
  GET /            - Serve the static HTML UI.
  WS  /ws/session - Full-duplex JSON for session state, FSR plots, and control.
  WS  /ws/camera  - Binary JPEG stream from the camera reader.

Usage:
    uvicorn apps.collection.api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Ensure the app's INFO breadcrumbs (test progression, flush_buffer, etc.)
# appear in the console when launched via `uvicorn apps.collection.api:app`.
# Uvicorn only configures its own loggers, leaving the `apps.collection`
# package at the root WARNING level — which hid the per-test INFO logs we
# need to diagnose where pre-collection stalls.  We attach a stream handler
# to the package logger if one isn't already present.
_pkg_logger = logging.getLogger("apps.collection")
if not _pkg_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    _pkg_logger.addHandler(_h)
    _pkg_logger.setLevel(logging.INFO)
    _pkg_logger.propagate = False

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from apps.collection.camera import create_camera_reader
from apps.collection.protocol import build_protocol
from apps.collection.sensor_reader import create_sensor_reader
from apps.collection.config import QualityConfig, TestDurationConfig
from apps.collection.session import SessionEvent, run_session

app = FastAPI(title="JoTouch Collection UI")

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    """Serve the main UI page."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    """Return the current session status and event-loop health metrics."""
    return {
        "running": _manager.is_running(),
        "has_summary": _manager._last_summary is not None,
        "dropped_events": _manager._total_dropped_events,
        "clients": len(_manager._clients),
    }


@app.get("/api/ports")
async def api_ports() -> dict[str, Any]:
    """List available serial ports on the system.

    Uses pyserial's list_ports to find all COM ports (Windows) or /dev/tty* (Linux).
    """
    try:
        from serial.tools.list_ports import comports
    except ImportError:
        return {"ports": [], "error": "pyserial not installed"}
    ports = []
    for p in comports():
        ports.append({
            "device": p.device,
            "description": p.description,
            "manufacturer": p.manufacturer or "",
        })
    return {"ports": ports}


@app.get("/api/cameras")
async def api_cameras() -> dict[str, Any]:
    """Scan for available cameras by probing indexes 0-4.

    Returns a list of {"index": N, "available": true/false} entries.
    Uses cv2.VideoCapture to check each index.
    """
    try:
        import cv2
    except ImportError:
        return {"cameras": [], "error": "OpenCV not installed"}

    cameras = []
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                cameras.append({"index": i, "available": True})
                cap.release()
            else:
                cameras.append({"index": i, "available": False})
        except Exception:
            cameras.append({"index": i, "available": False})

    return {"cameras": cameras}


@app.get("/api/subjects")
async def api_subjects() -> dict[str, Any]:
    """List existing subjects in the raw data directory.

    Scans data/raw/ for sub-{label} directories and returns the labels sorted
    alphabetically. Used by the Setup screen "Existing Subject" dropdown so
    operators can pick a returning subject without typing.
    """
    from core import naming
    subjects = naming.list_subjects()
    return {"subjects": subjects}


@app.get("/api/next_subject")
async def api_next_subject() -> dict[str, Any]:
    """Return the next free subject label (e.g. "P02") following the P{NN} convention.

    Scans data/raw/ for sub-P{NN} directories and returns P{max+1:02d}.
    Used by the Setup screen "New Subject" mode to auto-suggest a subject ID.

    Automatically quarantines incomplete sessions (0 completed runs) before
    computing the next label, so abandoned subject directories don't block
    label reuse.
    """
    from core import naming
    from apps.collection.bids_writer import sweep_incomplete_sessions
    try:
        sweep_incomplete_sessions()
    except Exception as exc:
        logger.warning("sweep_incomplete_sessions failed: %s", exc)
    label = naming.next_subject_label()
    return {"subject": label}


@app.get("/api/next_session")
async def api_next_session(sub: str) -> dict[str, Any]:
    """Return the next free session label for a subject (e.g. "S02").

    Scans data/raw/sub-{sub}/ for ses-S{NN} directories and returns S{max+1:02d}.
    Used by the Setup screen to auto-suggest the session ID when the operator
    selects or enters a subject, preventing session collisions and run-number
    drift on re-runs.

    Automatically quarantines incomplete sessions (0 completed runs) before
    computing the next label, so abandoned session directories don't block
    label reuse.
    """
    from core import naming
    from apps.collection.bids_writer import sweep_incomplete_sessions
    try:
        sweep_incomplete_sessions()
    except Exception as exc:
        logger.warning("sweep_incomplete_sessions failed: %s", exc)
    label = naming.next_session_label(sub)
    return {"session": label, "sub": sub}


@app.post("/api/led_roi/calibrate")
async def api_calibrate_led_roi(camera_index: int = 0) -> dict[str, Any]:
    """Run interactive LED ROI calibration.

    Opens an OpenCV window for the user to select the LED region.
    Saves the ROI to led_roi.json in the project root.

    The calibration is run in a thread-pool thread so the FastAPI event loop
    stays responsive to other requests (e.g., FSR preview, camera stream).

    Critically: we acquire the SHARED camera reader and pass it into
    calibrate() rather than letting calibrate() create its own.  On Windows,
    cameras are exclusive-access — a second VideoCapture on the same index
    steals the device from the existing reader and breaks the live stream.
    By sharing the already-running CameraReader, calibration simply reads
    frames from it without touching the VideoCapture at all.
    """
    from pathlib import Path
    from apps.collection.led_roi import calibrate

    output_path = Path("led_roi.json")
    # Acquire the shared camera so calibration can read from it.
    # We release it when done — the WS stream keeps its own reference,
    # so release_camera() won't stop the camera while the WS is open.
    try:
        camera_reader = _manager.acquire_camera(dry_run=False, camera_index=camera_index)
    except RuntimeError as exc:
        return {"ok": False, "error": f"Cannot open camera: {exc}"}

    # Start the LED blinking in blink-only mode (no PRBS) so the operator
    # can see it in the camera and draw a rectangle around it.  We try the
    # session's sensor reader first, then the preview reader.  If neither
    # is available, calibration proceeds without the LED (the user can
    # still draw a rectangle around the LED location).
    _led_started = False
    for _reader in (_manager._sensor_reader, getattr(_manager, '_preview_reader', None)):
        if _reader is not None and hasattr(_reader, 'start_led_preview'):
            try:
                _reader.start_led_preview()
                _led_started = True
                break
            except Exception:
                pass

    try:
        roi = await asyncio.to_thread(
            calibrate,
            camera_index=camera_index,
            output=output_path,
            camera_reader=camera_reader,  # share — do NOT let calibrate() open its own
        )
        if roi is None:
            return {"ok": False, "error": "No ROI selected"}
        # Apply the newly saved ROI to the shared camera reader immediately
        roi_dict = {"x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height}
        if hasattr(camera_reader, "set_led_roi"):
            camera_reader.set_led_roi(roi_dict)
            logger.info("Applied new LED ROI to shared camera reader: %s", roi_dict)
        return {
            "ok": True,
            "roi": roi_dict,
            "path": str(output_path),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        _manager.release_camera()


@app.get("/api/led_roi")
async def api_get_led_roi() -> dict[str, Any]:
    """Load the saved LED ROI configuration."""
    from pathlib import Path
    import json
    roi_path = Path("led_roi.json")
    if roi_path.exists():
        try:
            with open(roi_path, encoding="utf-8") as f:
                return {"ok": True, "roi": json.load(f)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
    return {"ok": False, "error": "No ROI saved. Click 'Calibrate LED ROI' first."}


@app.get("/api/camera_settings")
async def api_get_camera_settings() -> dict[str, Any]:
    """Return the current camera tracking settings."""
    cam = _manager._camera_reader
    if cam is None or not hasattr(cam, "min_detection_confidence"):
        from apps.collection.config import DEFAULT_CAMERA_TRACKING_CONFIG as _cfg
        return {"ok": True, "settings": {
            "resolution": _cfg.resolution,
            "min_detection_confidence": _cfg.min_detection_confidence,
            "min_presence_confidence": _cfg.min_presence_confidence,
            "min_tracking_confidence": _cfg.min_tracking_confidence,
            "auto_exposure": _cfg.auto_exposure,
        }}
    return {"ok": True, "settings": {
        "resolution": getattr(cam, "resolution", "640x480"),
        "min_detection_confidence": cam.min_detection_confidence,
        "min_presence_confidence": cam.min_presence_confidence,
        "min_tracking_confidence": cam.min_tracking_confidence,
        "auto_exposure": getattr(cam, "auto_exposure", True),
    }}


@app.post("/api/camera_settings")
async def api_set_camera_settings(body: dict[str, Any] = None) -> dict[str, Any]:
    """Update camera tracking settings.

    Accepts a JSON body with any of:
      resolution, min_detection_confidence, min_presence_confidence,
      min_tracking_confidence, auto_exposure

    Changes to confidence thresholds take effect immediately on the running
    landmarker (the next frame uses the new values).  Resolution and
    auto_exposure require a camera restart to take effect.
    """
    if body is None:
        body = {}
    cam = _manager._camera_reader
    if cam is None or not hasattr(cam, "min_detection_confidence"):
        return {"ok": False, "error": "Camera not started. Start a session first."}

    changed = []
    needs_restart = False

    if "resolution" in body:
        new_res = str(body["resolution"])
        if new_res != getattr(cam, "resolution", ""):
            cam.resolution = new_res
            needs_restart = True
            changed.append("resolution")

    if "min_detection_confidence" in body:
        val = float(body["min_detection_confidence"])
        val = max(0.0, min(1.0, val))
        cam.min_detection_confidence = val
        changed.append("min_detection_confidence")

    if "min_presence_confidence" in body:
        val = float(body["min_presence_confidence"])
        val = max(0.0, min(1.0, val))
        cam.min_presence_confidence = val
        changed.append("min_presence_confidence")

    if "min_tracking_confidence" in body:
        val = float(body["min_tracking_confidence"])
        val = max(0.0, min(1.0, val))
        cam.min_tracking_confidence = val
        changed.append("min_tracking_confidence")

    if "auto_exposure" in body:
        cam.auto_exposure = bool(body["auto_exposure"])
        needs_restart = True
        changed.append("auto_exposure")

    # Confidence thresholds can't be changed on a running landmarker —
    # they're baked in at creation time.  We need to recreate the landmarker.
    # The simplest reliable way is to restart the camera reader.
    if any(c in changed for c in ("min_detection_confidence", "min_presence_confidence", "min_tracking_confidence")):
        needs_restart = True

    if needs_restart:
        try:
            cam.stop()
            if not cam.start():
                return {"ok": False, "error": "Camera restart failed after settings change."}
        except Exception as exc:
            return {"ok": False, "error": f"Camera restart failed: {exc}"}

    logger.info("Camera settings updated: %s (restart=%s)", changed, needs_restart)
    return {"ok": True, "changed": changed, "needs_restart": needs_restart}


@app.get("/api/fsr")
async def api_fsr(port: str | None = None, n_sensors: int = 4, dry_run: bool = False) -> dict[str, Any]:
    """Return live FSR values for UI bars (Setup + Tests screens).

    Query params:
      port: Serial port (e.g. COM4). If provided and dry_run is false,
            opens a real SerialSensorReader for live preview.
      n_sensors: Number of FSR channels.
      dry_run: If true, use mock reader.

    If a session is running, reads from the active session reader (ignores params).
    Uses read_sensors_preview() which is non-blocking and returns the latest
    sample from the reader thread without touching the serial port directly.
    """
    # If a session is active, starting, or has a sensor reader, do NOT
    # touch the serial port. The session thread is reading from it via
    # read_sensors(), and calling read_sensors_preview() concurrently on
    # the same serial.Serial object crashes the USB driver
    # ("ReadFile failed, device does not recognize the command").
    # FSR data during a session comes through the WebSocket /ws/session
    # events, not through this HTTP endpoint.
    if _manager.is_running() or getattr(_manager, '_session_starting', False) or _manager._sensor_reader is not None:
        return {"values": [], "available": False, "source": "session"}

    # No session active — create a preview reader based on user's port/dry_run choice
    # Treat empty string as None
    if port is not None and port.strip() == "":
        port = None

    reader = _manager.get_or_create_preview_reader(port=port, n_sensors=n_sensors, dry_run=dry_run)
    if reader is None:
        return {"values": [], "available": False, "source": "none"}

    # Determine source label
    is_mock = dry_run or port is None
    source = "mock" if is_mock else "serial"

    try:
        # read_sensors_preview is non-blocking (returns the latest sample
        # from the reader thread). Call it directly to avoid thread-pool
        # overhead that caused the UI preview to fall behind at 20 Hz.
        values = reader.read_sensors_preview()
        if values is not None:
            return {"values": values, "available": True, "source": source}
        # No sample yet (reader not started or no data)
        return {"values": [], "available": False, "source": source, "error": "No data received yet"}
    except Exception as exc:
        return {"values": [], "available": False, "source": "error", "error": str(exc)}


@app.get("/api/session_state")
async def api_session_state() -> dict[str, Any]:
    """Return the current authoritative session state for UI reconciliation.

    The UI can call this on page load or after a WebSocket reconnect to
    discover whether a session is already running and what phase/run it is in.
    """
    return {
        "running": _manager.is_running(),
        "has_summary": _manager._last_summary is not None,
        "total_runs": _manager._total_runs,
        "completed_runs": len(_manager._run_history),
        "tests": {
            "total": len(_manager._test_results),
            "passed": sum(1 for t in _manager._test_results if t.get("passed")),
        },
        "current_state": _manager._current_state,
        "last_summary": _manager._last_summary,
        "dropped_events": _manager._total_dropped_events,
    }


@app.get("/api/report")
async def api_report() -> dict[str, Any]:
    """Return a post-session first-pass report.

    Includes run counts, test results, quality flags, and warnings.
    """
    summary = _manager._last_summary or {}
    tests = _manager._test_results
    quality = _manager._quality_history
    runs = _manager._run_history

    reds = [q for q in quality if q.get("level") == "red"]
    yellows = [q for q in quality if q.get("level") == "yellow"]
    all_tests_passed = all(t.get("passed", False) for t in tests) if tests else True

    return {
        "summary": summary,
        "tests": {
            "total": len(tests),
            "passed": sum(1 for t in tests if t.get("passed")),
            "all_passed": all_tests_passed,
            "details": tests,
        },
        "quality": {
            "red_count": len(reds),
            "yellow_count": len(yellows),
            "red_reasons": [q["reason"] for q in reds],
            "yellow_reasons": [q["reason"] for q in yellows],
        },
        "runs": runs,
    }


# ── Event serialization ────────────────────────────────────────────────────


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy types and other non-JSON-serializable
    objects to native Python types.

    numpy.bool_ → bool, numpy.integer → int, numpy.floating → float,
    numpy.ndarray → list.  This is needed because test functions store
    numpy scalars in TestResult.details (e.g. ``corr > 0`` produces
    ``numpy.bool_``, not ``bool``), and ``json.dumps`` cannot serialize
    numpy types by default.
    """
    # Check the module name to detect numpy types without a hard import.
    # numpy.bool_ has type name "bool" on some platforms, so a class-name
    # check alone is unreliable.
    mod = type(obj).__module__
    if mod == "numpy":
        # numpy scalar or ndarray
        if hasattr(obj, "tolist"):
            # ndarray or scalar with .tolist()
            return _json_safe(obj.tolist())
        if hasattr(obj, "__bool__"):
            return bool(obj)
        if hasattr(obj, "__int__"):
            return int(obj)
        if hasattr(obj, "__float__"):
            return float(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _event_to_dict(event: SessionEvent | dict) -> dict[str, Any]:
    """Convert a session event dataclass (or plain dict) to a JSON-serializable dict."""
    if isinstance(event, dict):
        # Plain dicts from run_all_tests_interactive — still sanitize
        # because some may contain numpy types from test functions.
        return _json_safe(event)
    d = dataclasses.asdict(event)
    # Deep-convert numpy types, Path objects, etc.
    return _json_safe(d)


# Event priorities for broadcast backpressure. Critical events must survive;
# low-priority events (e.g., FSR) are dropped first when a client is slow.
_EVENT_PRIORITY: dict[str, int] = {
    "error": 0,
    "warning": 0,
    "session_ended": 0,
    "setup": 0,
    "test": 0,
    "quality": 0,
    "state": 0,
    "collection_ready": 0,
    "summary": 0,
    "mock_mode": 0,
    "test_instruction": 1,
    "test_ready": 1,
    "test_countdown": 1,
    "test_running": 1,
    "fsr": 2,
    "pong": 2,
}


def _event_priority(event: dict[str, Any]) -> int:
    """Return priority for an event (lower = more important)."""
    return _EVENT_PRIORITY.get(event.get("type", ""), 1)


_CLIENT_QUEUE_MAXLEN = 500
"""Maximum *control* events buffered per connected browser tab.

Control events (test lifecycle, quality, errors) are low-rate — a whole
session emits well under 500 of them — so this cap is effectively never hit.

Ephemeral events (FSR bars, pong) are NOT subject to this cap: they are
*coalesced* into a single latest-value slot.  Only the newest FSR reading
matters for a live bar display, so there is no reason to buffer a backlog of
stale frames.  This is the key fix for the "UI freezes after a test" bug:
previously a backlog of thousands of FSR frames sat ahead of the next
``test_instruction`` event in a FIFO queue, so the browser couldn't advance
to the next test until it had drained ~minutes of stale FSR data.
"""


class _ClientQueue:
    """Broadcast queue for one WebSocket client, with FSR coalescing.

    Two-tier design:
    - ``_dq``: FIFO deque of *control* events (priority 0 and 1). These are
      rare and must never be dropped or delayed.
    - ``_latest_ephemeral``: a single slot holding the most recent ephemeral
      event (priority ≥ 2, e.g. ``fsr``). New ephemeral events overwrite it.

    ``get()`` always returns control events first, so a pending
    ``test_instruction`` is delivered immediately even while FSR is streaming.

    Must only be accessed from the asyncio event-loop thread (via
    ``call_soon_threadsafe``).  ``get()`` awaits an ``asyncio.Event`` so the
    ``_send_events`` task does not busy-poll.
    """

    def __init__(self, maxlen: int = _CLIENT_QUEUE_MAXLEN) -> None:
        self._maxlen = maxlen
        self._dq: collections.deque[dict[str, Any]] = collections.deque()
        self._latest_ephemeral: dict[str, Any] | None = None
        self._ready = asyncio.Event()

    def put(self, event: dict[str, Any]) -> int:
        """Enqueue *event*; return the number of events dropped (0 or 1).

        Priority policy (lower number = more important):
          0 — critical (session_ended, error, quality, test result, …)
          1 — important (test_instruction, test_ready, test_countdown, …)
          2 — ephemeral (fsr, pong) — coalesced to the latest value only.

        Ephemeral events overwrite the single latest-value slot (the
        superseded frame is counted as dropped, since the browser never sees
        it). Control events append to the FIFO deque; in the practically
        impossible case the deque is full, the oldest control event is dropped.
        """
        priority = _event_priority(event)
        if priority >= 2:
            # Coalesce: keep only the most recent ephemeral event.
            dropped = 1 if self._latest_ephemeral is not None else 0
            self._latest_ephemeral = event
            self._ready.set()
            return dropped

        # Control event — FIFO, never coalesced.
        dropped = 0
        if len(self._dq) >= self._maxlen:
            # Should never happen in practice; drop oldest to stay bounded.
            self._dq.popleft()
            dropped = 1
            logger.error(
                "Control queue full (%d) — dropped oldest to enqueue %s",
                self._maxlen, event.get("type"),
            )
        self._dq.append(event)
        self._ready.set()
        return dropped

    async def get(self) -> dict[str, Any]:
        """Block until an event is available; control events take priority."""
        while not self._dq and self._latest_ephemeral is None:
            self._ready.clear()
            await self._ready.wait()
        if self._dq:
            return self._dq.popleft()
        event = self._latest_ephemeral
        self._latest_ephemeral = None
        return event  # type: ignore[return-value]


def _safe_put(q: _ClientQueue, event: dict[str, Any], overflow_info: dict) -> None:
    """Schedule *event* into *q*; called via ``call_soon_threadsafe``."""
    dropped = q.put(event)
    overflow_info["dropped"] += dropped


# ── Session manager ─────────────────────────────────────────────────────────


class SessionManager:
    """Manages a single collection session and broadcasts events to clients.

    Each connected session WebSocket gets its own ``asyncio.Queue`` so slow
    clients do not block the broadcast. Events are produced by running
    ``run_session`` in a dedicated thread and scheduled into the event loop with
    ``call_soon_threadsafe``.
    """

    def __init__(self) -> None:
        self._clients: list[_ClientQueue] = []
        self._thread: threading.Thread | None = None
        self._sensor_reader = None
        self._camera_reader = None
        self._camera_refcount = 0
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._begin_event = threading.Event()
        self._retry_event = threading.Event()
        self._retry_test_name: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.RLock()
        self._session_starting = False
        self._last_summary: dict[str, Any] | None = None
        self._quality_history: list[dict[str, str]] = []
        self._test_results: list[dict[str, Any]] = []
        self._run_history: list[dict[str, Any]] = []
        self._total_dropped_events: int = 0
        # Authoritative session state for UI reconciliation on reconnect
        self._current_state: dict[str, Any] = {}
        self._total_runs: int = 0

    async def register(self) -> _ClientQueue:
        """Register a new session client and return its event queue."""
        # Capture the running event loop lazily so it matches the server loop.
        self._loop = asyncio.get_running_loop()
        q = _ClientQueue()
        self._clients.append(q)
        return q

    def unregister(self, q: _ClientQueue) -> None:
        """Remove a client queue."""
        if q in self._clients:
            self._clients.remove(q)

    def _broadcast(self, event: dict[str, Any]) -> None:
        """Send an event to every connected session client."""
        # Track events for the post-session report
        etype = event.get("type")
        if etype == "quality":
            self._quality_history.append({
                "level": event.get("level", ""),
                "reason": event.get("reason", ""),
            })
        elif etype == "test":
            self._test_results.append(event)
        elif etype == "run_complete":
            self._run_history.append(event)
        elif etype == "summary":
            self._last_summary = event
            self._current_state = {}
        elif etype == "state":
            self._current_state = {
                "phase": event.get("phase"),
                "run": event.get("run"),
                "task": event.get("task"),
                "rep": event.get("rep"),
                "remaining_s": event.get("remaining_s"),
                "elapsed_session_s": event.get("elapsed_session_s"),
            }
        elif etype == "setup":
            self._total_runs = event.get("total_runs", 0)
            self._current_state = {}
            self._quality_history = []
            self._test_results = []
            self._run_history = []

        loop = self._loop
        if loop is None:
            return
        # Broadcast to all clients. Ephemeral (FSR) events are silently dropped
        # when the queue is full; critical/important events evict ephemerals.
        # Never broadcast a warning from inside _broadcast — that causes a
        # call_soon_threadsafe → _safe_put → eviction → cascade death spiral.
        overflow_info = {"dropped": 0}
        for q in list(self._clients):
            loop.call_soon_threadsafe(_safe_put, q, event, overflow_info)
        if overflow_info["dropped"] > 0:
            self._total_dropped_events += overflow_info["dropped"]
            # Log at DEBUG to avoid spamming stderr; the operator doesn't need
            # a per-drop warning — a periodic summary is sufficient.
            logger.debug(
                "Broadcast drop: %d events (running total %d)",
                overflow_info["dropped"], self._total_dropped_events,
            )

    def is_running(self) -> bool:
        """Return True if a session is currently running."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def get_or_create_preview_reader(self, port: str | None = None, n_sensors: int = 4, dry_run: bool = False):
        """Return a sensor reader for live FSR preview (Setup/Tests screens).

        If the user provided a port and dry_run is false, creates a real
        SerialSensorReader so they see live hardware data.
        If dry_run is true or no port, creates a MockSensorReader.
        Reuses an existing preview reader if the params match.

        Returns None if a session is running or starting (to avoid port conflict).
        """
        with self._lock:
            # Don't create a preview reader if a session is active or starting
            if self._thread is not None and self._thread.is_alive():
                return None
            if getattr(self, '_session_starting', False):
                return None
            if self._sensor_reader is not None:
                # Session is starting — sensor reader exists but thread not yet running
                return None
            # Check if we need to (re)create the preview reader
            need_new = False
            if not hasattr(self, '_preview_reader') or self._preview_reader is None:
                need_new = True
            elif hasattr(self, '_preview_params'):
                if self._preview_params != (port, n_sensors, dry_run):
                    # Params changed — stop old reader, create new one
                    try:
                        self._preview_reader.stop()
                    except Exception:
                        pass
                    need_new = True
            else:
                need_new = True

            if need_new:
                try:
                    self._preview_reader = create_sensor_reader(
                        port=port, n_sensors=n_sensors, dry_run=dry_run,
                    )
                    self._preview_reader.start()
                    self._preview_params = (port, n_sensors, dry_run)
                    # WS-1: Start the LED blinking in blink-only mode ('B')
                    # so the operator can see the LED in the cam view on the
                    # Setup screen. The 'S' command at RECORD overrides this.
                    # Safe for mock readers (no-op serial write) and real
                    # readers (sends 'B' to Arduino, see ADR 003).
                    if hasattr(self._preview_reader, 'start_led_preview'):
                        try:
                            self._preview_reader.start_led_preview()
                            logger.info("Started LED preview (blink-only) for preview reader")
                        except Exception as exc:
                            logger.warning("Could not start LED preview: %s", exc)
                except Exception:
                    self._preview_reader = None
                    return None
            return self._preview_reader

    def stop_preview_reader(self) -> None:
        """Stop and clear the preview reader (called when session starts)."""
        with self._lock:
            if hasattr(self, '_preview_reader') and self._preview_reader is not None:
                try:
                    self._preview_reader.stop()
                except Exception:
                    pass
                self._preview_reader = None

    def _load_led_roi(self) -> dict[str, int] | None:
        """Load the saved LED ROI from led_roi.json if it exists."""
        from pathlib import Path
        import json
        roi_path = Path("led_roi.json")
        if not roi_path.exists():
            return None
        try:
            with open(roi_path, encoding="utf-8") as f:
                data = json.load(f)
            return {
                "x": int(data["x"]),
                "y": int(data["y"]),
                "width": int(data["width"]),
                "height": int(data["height"]),
            }
        except Exception:
            return None

    def acquire_camera(
        self,
        dry_run: bool = False,
        camera_index: int = 0,
        *,
        resolution: str = "640x480",
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        auto_exposure: bool = True,
    ):
        """Acquire a shared camera reader, creating one if necessary.

        The first caller determines whether a real or mock camera is used.
        Subsequent callers share the existing reader. Must be paired with a
        call to ``release_camera``.
        """
        with self._lock:
            if self._camera_reader is None:
                self._camera_reader = create_camera_reader(
                    dry_run=dry_run,
                    camera_index=camera_index,
                    resolution=resolution,
                    min_detection_confidence=min_detection_confidence,
                    min_presence_confidence=min_presence_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                    auto_exposure=auto_exposure,
                )
                if not self._camera_reader.start():
                    if not dry_run:
                        # B2: Block start in live mode if no camera detected
                        self._camera_reader = None
                        raise RuntimeError(
                            "No camera detected in live mode. Connect a webcam or enable Dry run."
                        )
                    # In dry-run mode, fall back to mock camera
                    self._camera_reader = create_camera_reader(dry_run=True)
                    self._camera_reader.start()
                # Load and apply LED ROI if available
                roi = self._load_led_roi()
                if roi is not None and hasattr(self._camera_reader, "set_led_roi"):
                    self._camera_reader.set_led_roi(roi)
                    logger.info("Applied LED ROI to camera reader: %s", roi)
            self._camera_refcount += 1
            return self._camera_reader

    def release_camera(self) -> None:
        """Release the shared camera reader. Stops it when no users remain."""
        with self._lock:
            self._camera_refcount = max(0, self._camera_refcount - 1)
            if self._camera_refcount == 0 and self._camera_reader is not None:
                try:
                    self._camera_reader.stop()
                except Exception:
                    pass
                self._camera_reader = None

    async def start(
        self,
        *,
        sub: str = "P01",
        ses: str = "S01",
        port: str | None = None,
        n_sensors: int = 4,
        dry_run: bool = True,
        skip_precollect: bool = False,
        record_duration: float | None = None,
        prep_duration: float | None = None,
        rest_duration: float | None = None,
        n_reps: int = 3,
        include_freeform: bool = True,
        camera_index: int = 0,
        quality_config: Any | None = None,
        duration_overrides: dict[str, float] | None = None,
    ) -> None:
        """Start a new collection session in a background thread."""
        if self.is_running():
            self._broadcast({"type": "error", "message": "Session already running"})
            return

        # Validate BIDS labels before starting — prevents invalid filenames
        # that would break parse_filename round-trip, next_run_number, and the
        # loader. This is the AGENTS.md contract: "BIDS labels are alphanumeric
        # only — validate via naming.validate_label()".
        from core import naming
        try:
            naming.validate_label(sub, "sub")
            naming.validate_label(ses, "ses")
        except ValueError as exc:
            self._broadcast({"type": "error", "message": f"Invalid label: {exc}"})
            return

        protocol_kwargs: dict[str, Any] = {"n_reps": n_reps, "include_freeform": include_freeform}
        if record_duration is not None:
            protocol_kwargs["record_duration"] = record_duration
        if prep_duration is not None:
            protocol_kwargs["prep_duration"] = prep_duration
        if rest_duration is not None:
            protocol_kwargs["rest_duration"] = rest_duration
        runs = build_protocol(**protocol_kwargs)

        with self._lock:
            self._stop_event.clear()
            self._ready_event.clear()
            self._begin_event.clear()
            self._retry_event.clear()
            self._retry_test_name = None
            self._last_summary = None
            self._quality_history = []
            self._test_results = []
            self._run_history = []
            # Set flag to prevent preview reader recreation during startup
            self._session_starting = True

            # ── Transfer preview reader to session (avoid close/open cycle) ──
            # The preview reader already has the serial port open. Instead of
            # closing it and reopening (which causes USB crashes on Windows),
            # transfer it directly to the session if the port matches.
            preview = getattr(self, '_preview_reader', None)
            from apps.collection.sensor_reader import SerialSensorReader, MockSensorReader

            if not dry_run and port and isinstance(preview, SerialSensorReader) and preview.port == port:
                # Reuse the existing preview reader — no close/open, no USB glitch
                logger.info("Transferring preview reader (port=%s) to session", port)
                self._sensor_reader = preview
                self._preview_reader = None
                started = True
            elif dry_run and isinstance(preview, MockSensorReader):
                # Reuse existing mock preview reader
                logger.info("Transferring mock preview reader to session")
                self._sensor_reader = preview
                self._preview_reader = None
                started = True
            else:
                # Need to close preview and create new reader
                self.stop_preview_reader()
                # Release lock for the sleep, then re-acquire
                pass

        if not dry_run and port and self._sensor_reader is None:
            # Wait outside the lock for Windows to release the port
            import time as _time
            _time.sleep(1.0)

        with self._lock:
            if self._sensor_reader is not None:
                # Already transferred from preview — skip creation
                pass
            elif not dry_run and not port:
                self._session_starting = False
                self._broadcast({
                    "type": "error",
                    "message": "Serial port is required for live mode. Enter a port (e.g. COM4) or enable Dry run.",
                })
                return
            else:
                self._sensor_reader = create_sensor_reader(
                    port=port,
                    n_sensors=n_sensors,
                    dry_run=dry_run,
                )
                try:
                    logger.info("Starting sensor reader: port=%s dry_run=%s", port, dry_run)
                    started = self._sensor_reader.start()
                    # WS-1: Start LED blink-only mode for new session readers
                    # (not transferred from preview). The 'S' command at RECORD
                    # overrides this. Transfer case already has LED blinking.
                    if started and hasattr(self._sensor_reader, 'start_led_preview'):
                        try:
                            self._sensor_reader.start_led_preview()
                            logger.info("Started LED preview (blink-only) for session reader")
                        except Exception as exc:
                            logger.warning("Could not start LED preview for session: %s", exc)
                except Exception as exc:
                    # Serial port not available (permission denied, port in use, etc.)
                    logger.error("Sensor reader start failed: %s", exc, exc_info=True)
                    msg = str(exc)
                    if "Access is denied" in msg or "PermissionError" in msg:
                        msg = (
                            f"Cannot open {port}: port is in use by another application "
                            f"(e.g. Arduino IDE, CoolTerm). Close it and try again."
                        )
                    elif "could not open port" in msg:
                        msg = f"Cannot open {port}: {msg}"
                    self._sensor_reader = None
                    self._session_starting = False
                    self._broadcast({"type": "error", "message": msg})
                    return
            if not started:
                self._broadcast({"type": "error", "message": "Failed to start sensor reader"})
                self._sensor_reader = None
                self._session_starting = False
                return
            try:
                self.acquire_camera(dry_run=dry_run, camera_index=camera_index)
            except RuntimeError as exc:
                if not dry_run:
                    # Live mode requires a real camera or explicit Dry run (simulated camera).
                    # Do not silently continue without camera.
                    self._sensor_reader.stop()
                    self._sensor_reader = None
                    self._session_starting = False
                    self._broadcast({
                        "type": "error",
                        "message": f"No camera detected in live mode. {exc} Enable Dry run to use simulated camera, or connect a webcam.",
                    })
                    return
                # In dry-run mode, the camera fallback is already handled by acquire_camera
                logger.warning("No camera available in dry-run mode: %s", exc)
                self._broadcast({"type": "warning", "message": f"Camera not available: {exc}. Using simulated camera."})
                self._broadcast({"type": "mock_mode", "sensor": dry_run, "camera": True})

            # Session is starting — clear the flag once thread is about to run
            self._session_starting = False

            self._thread = threading.Thread(
                target=self._run,
                args=(sub, ses, n_sensors, dry_run, skip_precollect, runs, quality_config, duration_overrides),
                daemon=True,
            )
            self._thread.start()

    def _run(
        self,
        sub: str,
        ses: str,
        n_sensors: int,
        dry_run: bool,
        skip_precollect: bool,
        runs: list,
        quality_config: Any | None,
        duration_overrides: dict[str, float] | None,
    ) -> None:
        """Thread target: run the session generator and broadcast events."""
        try:
            logger.info("Session thread started: sub=%s ses=%s dry_run=%s", sub, ses, dry_run)
            generator = run_session(
                sub=sub,
                ses=ses,
                sensor_reader=self._sensor_reader,
                camera_reader=self._camera_reader,
                runs=runs,
                n_sensors=n_sensors,
                dry_run=dry_run,
                skip_precollect=skip_precollect,
                stop_event=self._stop_event,
                ready_event=self._ready_event,
                begin_event=self._begin_event,
                retry_event=self._retry_event,
                retry_test_name=self._retry_test_name,
                get_retry_test_name=lambda: self._retry_test_name,
                broadcast_fn=self._broadcast,
                quality_config=quality_config,
                duration_overrides=duration_overrides,
            )
            for event in generator:
                self._broadcast(_event_to_dict(event))
        except Exception as exc:
            logger.error("Session thread crashed: %s", exc, exc_info=True)
            self._broadcast({"type": "error", "message": f"Session crashed: {exc}"})
        finally:
            self._cleanup()
            logger.info("Session thread ended")

    def stop(self) -> None:
        """Request the running session to stop."""
        self._stop_event.set()

    def _cleanup(self) -> None:
        """Stop readers, clear the thread reference, and notify the UI.

        Also sweeps incomplete sessions to _incomplete/ so they don't sit
        alongside completed sessions in data/raw/. This runs after every
        session ends (whether completed or aborted), ensuring the data
        directory only contains sessions that completed their chosen
        protocol.
        """
        with self._lock:
            if self._sensor_reader is not None:
                try:
                    self._sensor_reader.stop()
                except Exception:
                    pass
                self._sensor_reader = None
            self.release_camera()
            self._thread = None
            self._session_starting = False
        # Sweep incomplete sessions to _incomplete/ so completed sessions
        # are never mixed with incomplete ones in data/raw/.
        try:
            from apps.collection.bids_writer import sweep_incomplete_sessions
            swept = sweep_incomplete_sessions()
            if swept > 0:
                logger.info("Post-session sweep moved %d incomplete session(s) to _incomplete/", swept)
        except Exception as exc:
            logger.warning("Post-session sweep_incomplete_sessions failed: %s", exc)
        # Emit a handshake event so the UI knows the session has fully ended.
        # This prevents race conditions where a stray 'setup' or 'state' event
        # received after a reconnect auto-advances the screen.
        self._broadcast({"type": "session_ended"})

    def reset(self) -> None:
        """Reset transient session state.

        Intended for test teardown only.  Clears history buffers and stops any
        residual readers without touching the client queue list or loop
        reference.  Does *not* wait for a running thread to finish — call
        ``stop()`` first if a session may be active.
        """
        with self._lock:
            self._last_summary = None
            self._quality_history = []
            self._test_results = []
            self._run_history = []
            if self._camera_reader is not None:
                try:
                    self._camera_reader.stop()
                except Exception:
                    pass
                self._camera_reader = None
                self._camera_refcount = 0


# Global manager for the API (one active session at a time)
_manager = SessionManager()


# ── WebSocket endpoints ────────────────────────────────────────────────────


@app.websocket("/ws/session")
async def ws_session(websocket: WebSocket) -> None:
    """JSON WebSocket for session events and control."""
    await websocket.accept()
    q = await _manager.register()

    async def _send_events() -> None:
        while True:
            event = await q.get()
            try:
                # Timeout: if the browser isn't reading (tab in background,
                # JS event loop blocked, network stall), drop this event and
                # continue rather than blocking the entire event pipeline.
                # Critical events (test results, errors) are in the same queue
                # and would be stuck behind a blocked send — the timeout
                # ensures they get a chance to be delivered on the next loop.
                await asyncio.wait_for(websocket.send_json(event), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "WebSocket send timed out (5s) — dropping event type=%s. "
                    "Browser may be in background or JS event loop blocked.",
                    event.get("type"),
                )
            except TypeError as exc:
                # JSON serialization error (e.g. numpy types that slipped
                # through _json_safe).  Log and continue — killing the
                # _send_events task here would permanently silence the
                # WebSocket and freeze the UI.
                logger.error(
                    "WebSocket send serialization error for event type=%s: %s. "
                    "Event dropped. This is a bug — _json_safe should have "
                    "converted this type.",
                    event.get("type"), exc,
                )
            except WebSocketDisconnect:
                return
            except Exception as exc:
                logger.warning("Session WS send error: %s", exc)
                return

    send_task = asyncio.create_task(_send_events())
    try:
        while True:
            message = await websocket.receive_json()
            await _handle_command(message)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("Session WS error: %s", exc)
    finally:
        send_task.cancel()
        _manager.unregister(q)
        try:
            await websocket.close()
        except Exception:
            pass  # Already closed


async def _handle_command(message: dict[str, Any]) -> None:
    """Process a control command from the UI."""
    cmd = message.get("command")
    if cmd == "start":
        quality_config = QualityConfig.from_dict(message.get("quality_config"))
        duration_cfg = TestDurationConfig.from_dict(message.get("duration_overrides"))
        await _manager.start(
            sub=message.get("sub", "P01"),
            ses=message.get("ses", "S01"),
            port=message.get("port"),
            n_sensors=int(message.get("n_sensors", 4)),
            dry_run=bool(message.get("dry_run", True)),
            skip_precollect=bool(message.get("skip_precollect", False)),
            record_duration=message.get("record_duration"),
            prep_duration=message.get("prep_duration"),
            rest_duration=message.get("rest_duration"),
            n_reps=int(message.get("n_reps", 3)),
            include_freeform=bool(message.get("include_freeform", True)),
            camera_index=int(message.get("camera_index", 0)),
            quality_config=quality_config,
            duration_overrides=duration_cfg.__dict__,
        )
    elif cmd == "stop":
        _manager.stop()
    elif cmd == "test_ready":
        _manager._ready_event.set()
    elif cmd == "begin_collection":
        _manager._begin_event.set()
    elif cmd == "retry_tests":
        # Retry all failed tests — clear any stale single-test name first
        _manager._retry_test_name = None
        _manager._retry_event.set()
    elif cmd == "retry_test":
        # Retry a specific test by name (must be a failed test)
        _manager._retry_test_name = message.get("name")
        _manager._retry_event.set()
    elif cmd == "ping":
        _manager._broadcast({"type": "pong"})
    else:
        _manager._broadcast({"type": "error", "message": f"Unknown command: {cmd}"})


@app.websocket("/ws/camera")
async def ws_camera(websocket: WebSocket) -> None:
    """Binary WebSocket that streams JPEG camera frames.

    Shares the camera reader with the active session (if any) so only one
    camera is opened at a time in live mode. Falls back to mock camera
    if no real camera is available.
    """
    await websocket.accept()

    # Try real camera first; if no camera available, show "no camera" message
    # but do NOT silently fall back to mock — the user needs to know.
    camera_reader = None
    try:
        camera_reader = _manager.acquire_camera(dry_run=False)
    except RuntimeError:
        # No real camera — send a "no camera" placeholder frame
        pass

    if camera_reader is None:
        # Send a text message indicating no camera, then keep connection open
        try:
            import cv2
            import numpy as np
            # Generate a "NO CAMERA" placeholder image
            img = np.full((480, 640, 3), (40, 40, 40), dtype=np.uint8)
            cv2.putText(img, "NO CAMERA", (160, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        2.0, (80, 80, 80), 3, cv2.LINE_AA)
            cv2.putText(img, "Connect a webcam or enable Dry run", (120, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", img)
            if ok:
                await websocket.send_bytes(bytes(buf))
        except Exception:
            pass
        # Keep connection open but don't stream frames
        try:
            while True:
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        return

    try:
        while True:
            # get_jpeg() returns pre-encoded bytes from the capture thread —
            # no blocking work here, safe to call directly on the event loop.
            jpeg = camera_reader.get_jpeg()
            if jpeg:
                # WS-4: Multiplex JPEG + JSON metadata with 1-byte prefix.
                # 0x00 = JPEG frame, 0x01 = JSON metadata.
                await websocket.send_bytes(bytes([0x00]) + jpeg)
                # Send JSON metadata after each JPEG frame
                try:
                    frame_data = camera_reader.get_frame()
                    fps = camera_reader.get_fps() if hasattr(camera_reader, 'get_fps') else 0.0
                    metadata = {
                        "fps": round(fps, 1),
                        "led_brightness": frame_data.get("led_brightness"),
                        "roi": frame_data.get("roi"),
                        "valid": frame_data.get("valid", False),
                        "confidence": frame_data.get("confidence", 0.0),
                        "handedness": frame_data.get("handedness", "Right"),
                        "t_ns": frame_data.get("t_ns", 0),
                    }
                    import json as _json
                    await websocket.send_bytes(
                        bytes([0x01]) + _json.dumps(metadata).encode("utf-8")
                    )
                except Exception as exc:
                    logger.warning("Camera metadata send failed: %s", exc)
            await asyncio.sleep(1.0 / 30.0)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("Camera WS error: %s", exc)
    finally:
        _manager.release_camera()
        try:
            await websocket.close()
        except Exception:
            pass  # Already closed
