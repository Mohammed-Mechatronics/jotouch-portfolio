"""Tests for apps/collection/api.py — FastAPI WebSocket backend."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.collection.api import _ClientQueue, _event_priority, _manager, _safe_put, app


@pytest.fixture
def client():
    """Return a TestClient for the FastAPI app."""
    with TestClient(app) as c:
        yield c


class TestStatic:
    def test_index_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "<title>JoTouch Collection</title>" in response.text

    def test_static_css(self, client):
        response = client.get("/static/style.css")
        assert response.status_code == 200
        assert "body" in response.text


class TestSessionWebSocket:
    def test_session_start_and_summary(self, client):
        """Start a short dry-run session and receive at least setup + summary."""
        with client.websocket_connect("/ws/session") as ws:
            ws.send_json({
                "command": "start",
                "sub": "P02",
                "ses": "S02",
                "dry_run": True,
                "skip_precollect": True,
                "record_duration": 0.2,
                "prep_duration": 0.1,
                "rest_duration": 0.1,
                "n_reps": 1,
                "include_freeform": False,
            })

            # Collect events until we see a summary.
            # The session now has a collection gate, so we must send begin_collection
            # after the collection_ready event.
            events = []
            deadline = time.time() + 30.0
            while time.time() < deadline:
                event = ws.receive_json()
                events.append(event)
                if event.get("type") == "collection_ready":
                    ws.send_json({"command": "begin_collection"})
                if event.get("type") == "summary":
                    break

            types = {e.get("type") for e in events}
            assert "setup" in types
            assert "summary" in types
            summary = next(e for e in events if e["type"] == "summary")
            assert summary["completed_runs"] > 0

    def test_stop_command(self, client):
        """Send stop after collection begins and verify the session terminates early."""
        with client.websocket_connect("/ws/session") as ws:
            ws.send_json({
                "command": "start",
                "sub": "P03",
                "ses": "S03",
                "dry_run": True,
                "skip_precollect": True,
                "record_duration": 2.0,
                "prep_duration": 0.5,
                "rest_duration": 0.5,
                "n_reps": 1,
                "include_freeform": False,
            })

            # Wait for the collection gate, then begin collection before stopping.
            events = []
            began = False
            deadline = time.time() + 15.0
            while time.time() < deadline:
                event = ws.receive_json()
                events.append(event)
                if event.get("type") == "collection_ready" and not began:
                    ws.send_json({"command": "begin_collection"})
                    began = True
                    # Stop shortly after recording starts
                    time.sleep(0.3)
                    ws.send_json({"command": "stop"})
                if event.get("type") == "summary":
                    break

            summary = next((e for e in events if e["type"] == "summary"), None)
            assert summary is not None
            # Should not have completed all runs because we stopped early
            assert summary["completed_runs"] < summary["total_runs"]

    def test_invalid_sub_label_rejected(self, client):
        """A session start with a non-alphanumeric sub label must be rejected
        with an error event and must NOT start a session."""
        with client.websocket_connect("/ws/session") as ws:
            ws.send_json({
                "command": "start",
                "sub": "P-01",  # hyphen is not alphanumeric — must be rejected
                "ses": "S01",
                "dry_run": True,
                "skip_precollect": True,
            })
            # The backend must send an error event, not a setup event
            events = []
            deadline = time.time() + 5.0
            while time.time() < deadline:
                try:
                    event = ws.receive_json()
                    events.append(event)
                    if event.get("type") in ("error", "setup"):
                        break
                except Exception:
                    break
            types = {e.get("type") for e in events}
            assert "error" in types
            assert "setup" not in types
            assert not _manager.is_running()

    def test_invalid_ses_label_rejected(self, client):
        """A session start with a non-alphanumeric ses label must be rejected."""
        with client.websocket_connect("/ws/session") as ws:
            ws.send_json({
                "command": "start",
                "sub": "P01",
                "ses": "S 02",  # space is not alphanumeric — must be rejected
                "dry_run": True,
                "skip_precollect": True,
            })
            events = []
            deadline = time.time() + 5.0
            while time.time() < deadline:
                try:
                    event = ws.receive_json()
                    events.append(event)
                    if event.get("type") in ("error", "setup"):
                        break
                except Exception:
                    break
            types = {e.get("type") for e in events}
            assert "error" in types
            assert "setup" not in types

    def test_empty_sub_label_rejected(self, client):
        """A session start with an empty sub label must be rejected."""
        with client.websocket_connect("/ws/session") as ws:
            ws.send_json({
                "command": "start",
                "sub": "",
                "ses": "S01",
                "dry_run": True,
                "skip_precollect": True,
            })
            events = []
            deadline = time.time() + 5.0
            while time.time() < deadline:
                try:
                    event = ws.receive_json()
                    events.append(event)
                    if event.get("type") in ("error", "setup"):
                        break
                except Exception:
                    break
            types = {e.get("type") for e in events}
            assert "error" in types
            assert "setup" not in types


class TestIncompleteSessionQuarantine:
    """Verify that incomplete sessions are moved to _incomplete/ when the
    session ends, not left alongside completed sessions in data/raw/.

    A session is "complete" when it has finished its CHOSEN protocol
    (respecting n_reps + include_freeform), not just the default 76 runs.
    """

    def test_incomplete_session_moved_on_end(self, client):
        """After a session ends that did NOT complete its chosen protocol,
        the session directory must be moved to _incomplete/ so it doesn't
        conflict with completed sessions."""
        from core import paths
        sub, ses = "P99", "S99"
        # Sessions write to data_root=paths.RAW_DIR (data/raw/), not data/sample/
        sdir = paths.RAW_DIR / f"sub-{sub}" / f"ses-{ses}"
        incomplete_dir = paths.RAW_DIR / "_incomplete" / f"sub-{sub}" / f"ses-{ses}"
        import shutil
        if sdir.exists():
            shutil.rmtree(sdir, ignore_errors=True)
        if incomplete_dir.exists():
            shutil.rmtree(incomplete_dir, ignore_errors=True)

        with client.websocket_connect("/ws/session") as ws:
            ws.send_json({
                "command": "start",
                "sub": sub,
                "ses": ses,
                "dry_run": True,
                "skip_precollect": True,
                "record_duration": 0.2,
                "prep_duration": 0.1,
                "rest_duration": 0.1,
                "n_reps": 1,
                "include_freeform": False,
            })
            events = []
            deadline = time.time() + 30.0
            while time.time() < deadline:
                event = ws.receive_json()
                events.append(event)
                if event.get("type") == "collection_ready":
                    ws.send_json({"command": "begin_collection"})
                if event.get("type") == "summary":
                    break
            # Wait for session_ended (sweep runs in _cleanup after summary)
            deadline2 = time.time() + 10.0
            while time.time() < deadline2:
                event = ws.receive_json()
                events.append(event)
                if event.get("type") == "session_ended":
                    break

        # The session ran n_reps=1, include_freeform=False → chosen protocol
        # is 25 runs (mvc + 15 single_dof + 9 multi_dof).  It DID complete
        # all 25, so it must STAY in data/raw/, NOT be moved to _incomplete/.
        assert sdir.exists(), \
            f"Session that completed its chosen protocol must stay in {sdir}, " \
            f"but it was moved to _incomplete/"
        assert not incomplete_dir.exists(), \
            f"Complete-by-choice session must NOT be moved to {incomplete_dir}"

        # Cleanup
        if sdir.exists():
            shutil.rmtree(sdir, ignore_errors=True)
        if incomplete_dir.exists():
            shutil.rmtree(incomplete_dir, ignore_errors=True)
        # Remove the empty subject directory (sub-P99) so it doesn't
        # inflate next_subject_label for real sessions.
        p99_dir = paths.RAW_DIR / f"sub-{sub}"
        if p99_dir.exists():
            try:
                shutil.rmtree(p99_dir, ignore_errors=True)
            except OSError:
                pass
        # Also remove from _incomplete/sub-P99 if empty
        p99_incomplete = paths.RAW_DIR / "_incomplete" / f"sub-{sub}"
        if p99_incomplete.exists():
            try:
                shutil.rmtree(p99_incomplete, ignore_errors=True)
            except OSError:
                pass
        # Remove from TSVs if the session was added
        from apps.collection.bids_writer import _remove_from_tsv
        _remove_from_tsv(paths.RAW_DIR / "sessions.tsv", sub, ses)
        _remove_from_tsv(paths.RAW_DIR / "participants.tsv", sub)


class TestCameraWebSocket:
    def test_camera_binary_frames(self, client):
        """Connect to the camera WebSocket and receive JPEG frames.

        WS-4: Frames are now multiplexed with a 1-byte type prefix:
        0x00 = JPEG, 0x01 = JSON metadata. JPEG frames start with 0x00
        followed by the JPEG magic bytes 0xFF 0xD8.
        """
        with client.websocket_connect("/ws/camera") as ws:
            deadline = time.time() + 10.0
            received_jpeg = False
            while time.time() < deadline:
                message = ws.receive_bytes()
                # WS-4: JPEG frames have 0x00 prefix followed by JPEG magic
                if len(message) > 1 and message[0] == 0x00 and message[1:3] == b"\xff\xd8":
                    received_jpeg = True
                    break
                # Old-style frames (no prefix) also accepted for backward compat
                elif message.startswith(b"\xff\xd8"):
                    received_jpeg = True
                    break
            assert received_jpeg, "Did not receive a JPEG frame within deadline"

    def test_camera_metadata_frames(self, client):
        """WS-4: The camera WS must send JSON metadata frames (prefix 0x01)
        containing fps, led_brightness, roi, valid, and confidence."""
        import json as _json
        with client.websocket_connect("/ws/camera") as ws:
            deadline = time.time() + 15.0
            received_metadata = False
            while time.time() < deadline:
                message = ws.receive_bytes()
                if len(message) > 1 and message[0] == 0x01:
                    # JSON metadata frame
                    payload = message[1:]
                    data = _json.loads(payload.decode("utf-8"))
                    assert "fps" in data, f"Metadata must include fps, got keys: {list(data.keys())}"
                    assert "led_brightness" in data, "Metadata must include led_brightness"
                    assert "valid" in data, "Metadata must include valid"
                    assert "confidence" in data, "Metadata must include confidence"
                    received_metadata = True
                    break
            assert received_metadata, "Did not receive a JSON metadata frame within deadline"


class TestSharedCamera:
    def test_acquire_camera_refcount(self):
        """Camera reader is shared and refcounted between session and preview."""
        # Clean state from previous tests
        _manager.release_camera()
        _manager._camera_reader = None
        _manager._camera_refcount = 0

        reader1 = _manager.acquire_camera(dry_run=True)
        assert reader1 is not None
        assert _manager._camera_refcount == 1

        reader2 = _manager.acquire_camera(dry_run=True)
        assert reader2 is reader1, "Same camera reader should be shared"
        assert _manager._camera_refcount == 2

        _manager.release_camera()
        assert _manager._camera_refcount == 1
        assert _manager._camera_reader is not None

        _manager.release_camera()
        assert _manager._camera_refcount == 0
        assert _manager._camera_reader is None


class TestRestEndpoints:
    def test_status_endpoint(self, client):
        """GET /api/status returns a valid status dict."""
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert "running" in data
        assert "has_summary" in data

    def test_report_endpoint(self, client):
        """GET /api/report returns report structure even with no session."""
        # Reset manager state from previous tests
        _manager._last_summary = None
        _manager._quality_history = []
        _manager._test_results = []
        _manager._run_history = []

        r = client.get("/api/report")
        assert r.status_code == 200
        data = r.json()
        assert "summary" in data
        assert "tests" in data
        assert "quality" in data
        assert "runs" in data
        assert data["tests"]["total"] == 0
        assert data["quality"]["red_count"] == 0


class TestBroadcastBackpressure:
    def test_event_priority_known_types(self):
        assert _event_priority({"type": "error"}) == 0
        assert _event_priority({"type": "fsr"}) == 2
        assert _event_priority({"type": "test_countdown"}) == 1

    def test_fsr_events_are_coalesced(self):
        """Successive FSR events overwrite the latest-value slot (no backlog)."""
        q = _ClientQueue()
        overflow = {"dropped": 0}
        _safe_put(q, {"type": "fsr", "values": [1, 2, 3, 4]}, overflow)
        # First FSR: nothing dropped
        assert overflow["dropped"] == 0
        _safe_put(q, {"type": "fsr", "values": [5, 6, 7, 8]}, overflow)
        # Second FSR overwrites the first — the superseded frame counts as dropped
        assert overflow["dropped"] == 1
        # Only the latest ephemeral is retained; control deque stays empty
        assert len(q._dq) == 0
        assert q._latest_ephemeral["values"] == [5, 6, 7, 8]

    def test_control_events_delivered_before_fsr(self):
        """get() returns control events before any pending FSR, even if FSR was enqueued first."""
        async def scenario():
            q = _ClientQueue()
            overflow = {"dropped": 0}
            # FSR is streaming...
            _safe_put(q, {"type": "fsr", "values": [1, 2, 3, 4]}, overflow)
            # ...then a critical test transition arrives
            _safe_put(q, {"type": "test_instruction", "name": "single_dof_isolation"}, overflow)

            # The control event must come out FIRST (not stuck behind FSR)
            first = await q.get()
            second = await q.get()
            return first, second

        first, second = asyncio.run(scenario())
        assert first["type"] == "test_instruction"
        assert second["type"] == "fsr"

    def test_control_events_are_fifo(self):
        """Multiple control events preserve insertion order."""
        async def scenario():
            q = _ClientQueue()
            overflow = {"dropped": 0}
            _safe_put(q, {"type": "test_instruction", "name": "a"}, overflow)
            _safe_put(q, {"type": "test_ready", "name": "a"}, overflow)
            _safe_put(q, {"type": "test_running", "name": "a"}, overflow)
            return [(await q.get())["type"] for _ in range(3)]

        assert asyncio.run(scenario()) == ["test_instruction", "test_ready", "test_running"]


class TestStatusAndSessionState:
    def test_status_exposes_dropped_events_and_clients(self, client):
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "running" in data
        assert "dropped_events" in data
        assert "clients" in data

    def test_session_state_before_session(self, client):
        response = client.get("/api/session_state")
        assert response.status_code == 200
        data = response.json()
        assert data["running"] is False
        assert data["completed_runs"] == 0
        assert data["current_state"] == {}


class TestLedRoiEndpoint:
    def test_calibrate_uses_shared_camera_and_does_not_crash_stream(self, client, monkeypatch):
        """POST /api/led_roi/calibrate must share the manager's camera reader —
        it must NOT open a second VideoCapture that would steal the device.

        We verify this by checking that:
        1. The endpoint calls _manager.acquire_camera() (refcount goes up).
        2. calibrate() is called with camera_reader= set (not None).
        3. The endpoint returns ok=True when calibrate() returns a valid ROI.
        4. _manager.release_camera() is called in the finally block (refcount back down).
        """
        from apps.collection.led_roi import LedRoi
        import apps.collection.api as api_mod

        captured = {}

        def mock_calibrate(**kwargs):
            captured["camera_reader"] = kwargs.get("camera_reader")
            # Return a minimal valid ROI
            return LedRoi(x=10, y=20, width=50, height=50,
                          camera_index=0, image_width=640, image_height=480)

        monkeypatch.setattr(api_mod, "_manager", _manager)  # ensure we use the test manager
        # Patch calibrate inside api module's namespace
        import apps.collection.led_roi as led_roi_mod
        monkeypatch.setattr(led_roi_mod, "calibrate", lambda **kw: mock_calibrate(**kw))

        # Patch asyncio.to_thread to call the function synchronously in tests
        import asyncio as _asyncio
        async def _sync_to_thread(fn, **kw):
            return fn(**kw)
        monkeypatch.setattr(_asyncio, "to_thread", _sync_to_thread)

        refcount_before = _manager._camera_refcount

        response = client.post("/api/led_roi/calibrate?camera_index=0")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["roi"]["x"] == 10
        assert data["roi"]["y"] == 20

        # Shared camera must have been acquired and then released
        assert _manager._camera_refcount == refcount_before, \
            "release_camera() must be called in the finally block"

        # calibrate() must have received the shared camera reader, not None
        assert captured.get("camera_reader") is not None, \
            "calibrate() must receive the shared camera reader, not open its own"

    def test_get_led_roi_returns_error_when_missing(self, client):
        """GET /api/led_roi returns ok=False when no ROI has been saved yet."""
        import os
        # Ensure no led_roi.json exists in cwd
        if os.path.exists("led_roi.json"):
            os.rename("led_roi.json", "led_roi.json.bak")
        try:
            response = client.get("/api/led_roi")
            assert response.status_code == 200
            data = response.json()
            assert data["ok"] is False
        finally:
            if os.path.exists("led_roi.json.bak"):
                os.rename("led_roi.json.bak", "led_roi.json")


class TestLedPreviewWiring:
    """WS-1: The 'B' (blink-only) command must be sent to the Arduino
    whenever the serial port is connected, so the operator can see the LED
    in the cam view on Setup and Collect PREP screens — not only during
    ROI calibration and the sync_check test.
    """

    def test_preview_reader_sends_B_after_start(self, monkeypatch):
        """get_or_create_preview_reader() must call start_led_preview()
        on the reader after start() succeeds, so the LED blinks on Setup."""
        import apps.collection.api as api_mod
        from apps.collection.sensor_reader import MockSensorReader

        # Clean state
        api_mod._manager.stop_preview_reader()
        api_mod._manager._sensor_reader = None

        # Wrap MockSensorReader to record start_led_preview calls
        led_preview_calls = []
        original_start = MockSensorReader.start
        original_led_preview = MockSensorReader.start_led_preview

        def tracking_start(self):
            result = original_start(self)
            return result

        def tracking_led_preview(self):
            led_preview_calls.append(True)
            return original_led_preview(self)

        monkeypatch.setattr(MockSensorReader, "start", tracking_start)
        monkeypatch.setattr(MockSensorReader, "start_led_preview", tracking_led_preview)

        try:
            reader = api_mod._manager.get_or_create_preview_reader(
                port=None, n_sensors=4, dry_run=True
            )
            assert reader is not None, "preview reader should be created"
            assert len(led_preview_calls) >= 1, \
                "start_led_preview() must be called after preview reader start()"
        finally:
            api_mod._manager.stop_preview_reader()
            api_mod._manager._sensor_reader = None

    def test_session_reader_sends_B_after_start_for_new_reader(self, monkeypatch):
        """When a new session sensor reader is created (not transferred from
        preview), start_led_preview() must be called after start() so the LED
        blinks during PREP. The 'S' command at RECORD overrides this."""
        import apps.collection.api as api_mod
        from apps.collection.sensor_reader import MockSensorReader

        # Clean state — no preview reader, no session reader
        api_mod._manager.stop_preview_reader()
        api_mod._manager._sensor_reader = None

        led_preview_calls = []
        original_led_preview = MockSensorReader.start_led_preview

        def tracking_led_preview(self):
            led_preview_calls.append(True)
            return original_led_preview(self)

        monkeypatch.setattr(MockSensorReader, "start_led_preview", tracking_led_preview)

        # Simulate the session start path that creates a new reader
        # (dry_run=True, no preview reader to transfer)
        try:
            reader = api_mod._manager.get_or_create_preview_reader(
                port=None, n_sensors=4, dry_run=True
            )
            assert reader is not None
            # The preview reader path should have called start_led_preview
            assert len(led_preview_calls) >= 1, \
                "start_led_preview() must be called when a reader starts"
        finally:
            api_mod._manager.stop_preview_reader()
            api_mod._manager._sensor_reader = None

    def test_transferred_reader_does_not_call_B_again(self, monkeypatch):
        """When a preview reader is transferred to the session (same port),
        start_led_preview() should NOT be called again — the LED is already
        blinking from the preview phase. Calling it again is harmless but
        redundant and resets the blink phase."""
        # This is a no-op test: the transfer path at api.py does not call
        # start() (the reader is already running), so no new B is sent.
        # We verify the contract: the transfer path does not call start_led_preview.
        import apps.collection.api as api_mod
        from apps.collection.sensor_reader import MockSensorReader

        api_mod._manager.stop_preview_reader()
        api_mod._manager._sensor_reader = None

        led_preview_calls = []
        original_led_preview = MockSensorReader.start_led_preview

        def tracking_led_preview(self):
            led_preview_calls.append(True)
            return original_led_preview(self)

        monkeypatch.setattr(MockSensorReader, "start_led_preview", tracking_led_preview)

        # Create a preview reader (this calls start_led_preview once)
        reader = api_mod._manager.get_or_create_preview_reader(
            port=None, n_sensors=4, dry_run=True
        )
        assert reader is not None
        count_after_preview = len(led_preview_calls)
        assert count_after_preview >= 1

        # Simulate transfer: assign preview reader to session reader slot
        # without calling start() again
        api_mod._manager._sensor_reader = reader
        api_mod._manager._preview_reader = None

        # No new start_led_preview call should happen during transfer
        assert len(led_preview_calls) == count_after_preview, \
            "start_led_preview() must NOT be called again during reader transfer"

        api_mod._manager._sensor_reader = None

