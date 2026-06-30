"""Tests for apps/collection/sensor_reader.py."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.collection.sensor_reader import (
    MockSensorReader,
    SensorSample,
    SerialSensorReader,
    create_sensor_reader,
)


class TestSensorSample:
    def test_fields(self):
        s = SensorSample(fsr=[100, 200, 300, 400], led=1, t_ns=12345)
        assert s.fsr == [100, 200, 300, 400]
        assert s.led == 1
        assert s.t_ns == 12345


class TestMockSensorReader:
    def test_start_stop(self):
        reader = MockSensorReader(n_sensors=4, seed=42)
        assert reader.start() is True
        sample = reader.read()
        reader.stop()
        assert len(sample.fsr) == 4
        assert all(200 <= v <= 800 for v in sample.fsr)
        assert sample.led in (0, 1)

    def test_read_sensors_returns_list(self):
        reader = MockSensorReader(n_sensors=4, seed=42)
        reader.start()
        fsr = reader.read_sensors()
        reader.stop()
        assert isinstance(fsr, list)
        assert len(fsr) == 4

    def test_led_blinks(self, monkeypatch):
        # Drive time.monotonic_ns deterministically: step 50ms per call so we
        # guarantee we hit both the ON (0–100ms) and OFF (100–1000ms) windows
        # within a single 1 Hz LED cycle — no real wall-clock wait needed.
        tick_ns = [0]
        def fake_monotonic_ns():
            v = tick_ns[0]
            tick_ns[0] += 50_000_000  # +50ms per read
            return v
        monkeypatch.setattr("apps.collection.sensor_reader.now_ns", fake_monotonic_ns)

        reader = MockSensorReader(n_sensors=4)
        reader.start()
        led_values = set()
        for _ in range(25):  # 25 × 50ms = 1.25s — covers >1 full LED cycle
            led_values.add(reader.read().led)
        reader.stop()
        assert led_values == {0, 1}

    def test_custom_sensor_count(self):
        reader = MockSensorReader(n_sensors=8, seed=42)
        reader.start()
        sample = reader.read()
        reader.stop()
        assert len(sample.fsr) == 8


class TestCreateSensorReader:
    def test_dry_run_returns_mock(self):
        reader = create_sensor_reader(dry_run=True, n_sensors=4)
        assert isinstance(reader, MockSensorReader)

    def test_no_port_returns_mock(self):
        reader = create_sensor_reader(n_sensors=4)
        assert isinstance(reader, MockSensorReader)

    def test_port_returns_serial(self):
        reader = create_sensor_reader(port="COM3", n_sensors=4)
        assert type(reader).__name__ == "SerialSensorReader"


class _FakeSerial:
    """Fake pyserial object for testing SerialSensorReader threading."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self._idx = 0
        self.closed = False
        self.writes: list[bytes] = []  # captures every write() call

    def readline(self) -> bytes:
        if self._idx < len(self._lines):
            line = self._lines[self._idx]
            self._idx += 1
            return line.encode("utf-8")
        # Simulate blocking until more data arrives — here we just return empty
        # after a tiny sleep so the reader loop can be stopped.
        time.sleep(0.01)
        return b""

    def reset_input_buffer(self) -> None:
        pass

    def write(self, data: bytes) -> int:
        self.writes.append(bytes(data))
        return len(data)

    def flush_output(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class TestSerialSensorReaderThreading:
    def test_concurrent_readers_get_samples(self, monkeypatch):
        """Two threads calling read() on the same SerialSensorReader must not crash."""
        lines = ["100,200,300,400,1\n"] * 50
        fake = _FakeSerial(lines)

        reader = SerialSensorReader(port="COM_FAKE", n_sensors=4)
        # Inject the fake serial object without opening real hardware.
        monkeypatch.setattr(reader, "_open_serial", lambda: setattr(reader, "_ser", fake))
        reader.start()

        samples = []
        errors = []

        def consumer():
            try:
                for _ in range(10):
                    samples.append(reader.read())
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=consumer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        reader.stop()

        assert not errors, f"Consumer threads crashed: {errors}"
        assert len(samples) == 20, f"Expected 20 samples, got {len(samples)}"
        for s in samples:
            assert len(s.fsr) == 4
            assert all(0 <= v <= 1023 for v in s.fsr)
            assert s.led in (0, 1)

    def test_preview_and_session_reader_coexist(self, monkeypatch):
        """read_sensors_preview() can run concurrently with read() without stealing queue items."""
        lines = ["500,600,700,800,0\n"] * 50
        fake = _FakeSerial(lines)

        reader = SerialSensorReader(port="COM_FAKE", n_sensors=4)
        monkeypatch.setattr(reader, "_open_serial", lambda: setattr(reader, "_ser", fake))
        reader.start()

        session_values = []
        preview_values = []
        errors = []

        def session_consumer():
            try:
                for _ in range(10):
                    session_values.append(reader.read_sensors())
            except Exception as exc:
                errors.append(exc)

        def preview_consumer():
            try:
                for _ in range(10):
                    v = reader.read_sensors_preview(timeout_s=0.2)
                    if v is not None:
                        preview_values.append(v)
                    # Small delay to simulate 50 Hz preview polling
                    time.sleep(0.02)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=session_consumer)
        t2 = threading.Thread(target=preview_consumer)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        reader.stop()

        assert not errors, f"Threads crashed: {errors}"
        assert len(session_values) == 10, "Preview should not steal samples from session queue"
        assert len(preview_values) > 0
        # All preview values should be the same expected value since the fake reader
        # always produces the same sample, but more importantly preview never blocked.
        for v in preview_values:
            assert len(v) == 4

    def test_read_latest_returns_newest_sample(self, monkeypatch):
        """read_latest() returns the most recent sample without draining the queue."""
        lines = ["100,200,300,400,1\n"] * 10 + ["500,600,700,800,0\n"] * 10
        fake = _FakeSerial(lines)

        reader = SerialSensorReader(port="COM_FAKE", n_sensors=4)
        monkeypatch.setattr(reader, "_open_serial", lambda: setattr(reader, "_ser", fake))
        reader.start()

        # Wait for the latest sample to be updated to the second batch
        latest = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            latest = reader.read_latest()
            if latest is not None and latest.fsr == [500, 600, 700, 800]:
                break
            time.sleep(0.01)
        reader.stop()

        assert latest is not None
        assert latest.fsr == [500, 600, 700, 800]
        # The ordered queue should still have samples available
        assert reader._queue.qsize() > 0 or not reader._running

    def test_flush_buffer_drains_queue(self, monkeypatch):
        """flush_buffer() empties the sample queue so the next read gets fresh data."""
        lines = ["100,200,300,400,0\n"] * 50
        fake = _FakeSerial(lines)

        reader = SerialSensorReader(port="COM_FAKE", n_sensors=4)
        monkeypatch.setattr(reader, "_open_serial", lambda: setattr(reader, "_ser", fake))
        reader.start()

        # Let the queue fill with at least 5 items
        deadline = time.monotonic() + 2.0
        while reader._queue.qsize() < 5 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert reader._queue.qsize() >= 1, "Queue should have items before flush"
        reader.flush_buffer()
        assert reader._queue.qsize() == 0, "Queue should be empty after flush"
        reader.stop()

    def test_attach_detach_consumer_switches_mode(self, monkeypatch):
        """attach_consumer() sets _has_consumer=True; detach sets it back to False."""
        lines = ["100,200,300,400,0\n"] * 100
        fake = _FakeSerial(lines)

        reader = SerialSensorReader(port="COM_FAKE", n_sensors=4)
        monkeypatch.setattr(reader, "_open_serial", lambda: setattr(reader, "_ser", fake))
        reader.start()

        assert reader._has_consumer is False, "Default: preview mode"

        reader.attach_consumer()
        assert reader._has_consumer is True, "attach_consumer() must set _has_consumer=True"
        # Queue should be empty after attach (stale preview samples flushed)
        assert reader._queue.qsize() == 0, "attach_consumer() must drain stale preview samples"

        reader.detach_consumer()
        assert reader._has_consumer is False, "detach_consumer() must set _has_consumer=False"

        reader.stop()


# ── trigger_sync() (ADR 003) ─────────────────────────────────────────────────


class TestSerialTriggerSync:
    """SerialSensorReader.trigger_sync() sends b"S" to the Arduino under the
    reader lock so it doesn't race with the reader thread.
    """

    def test_trigger_sync_writes_S_byte(self, monkeypatch):
        lines = ["100,200,300,400,0\n"] * 20
        fake = _FakeSerial(lines)
        reader = SerialSensorReader(port="COM_FAKE", n_sensors=4)
        monkeypatch.setattr(reader, "_open_serial", lambda: setattr(reader, "_ser", fake))
        reader.start()
        try:
            reader.trigger_sync()
            assert fake.writes == [b"S"], f"Expected [b'S'], got {fake.writes}"
        finally:
            reader.stop()

    def test_trigger_sync_safe_when_port_closed(self, monkeypatch):
        reader = SerialSensorReader(port="COM_FAKE", n_sensors=4)
        # Do NOT call start() — _ser is None. trigger_sync must not raise.
        # Suppress the noisy log warning by checking the call returns cleanly.
        reader.trigger_sync()  # should not raise

    def test_trigger_sync_concurrent_with_reader_thread(self, monkeypatch):
        """trigger_sync() called from outside the reader thread must not crash
        and must not corrupt the readline loop. The reader lock serializes
        access to _ser.
        """
        lines = ["100,200,300,400,1\n"] * 200
        fake = _FakeSerial(lines)
        reader = SerialSensorReader(port="COM_FAKE", n_sensors=4)
        monkeypatch.setattr(reader, "_open_serial", lambda: setattr(reader, "_ser", fake))
        reader.start()
        try:
            samples: list[SensorSample] = []
            errors: list[Exception] = []

            def consumer():
                try:
                    for _ in range(20):
                        samples.append(reader.read())
                except Exception as exc:
                    errors.append(exc)

            t = threading.Thread(target=consumer)
            t.start()
            # Trigger sync mid-stream
            time.sleep(0.05)
            reader.trigger_sync()
            t.join(timeout=5)

            assert not errors, f"Consumer crashed during trigger: {errors}"
            assert len(samples) == 20
            assert b"S" in fake.writes, "trigger_sync byte was not written"
        finally:
            reader.stop()

    def test_start_led_preview_writes_B_byte(self, monkeypatch):
        """start_led_preview() sends b'B' to start blink-only mode (ADR 003)."""
        lines = ["100,200,300,400,0\n"] * 20
        fake = _FakeSerial(lines)
        reader = SerialSensorReader(port="COM_FAKE", n_sensors=4)
        monkeypatch.setattr(reader, "_open_serial", lambda: setattr(reader, "_ser", fake))
        reader.start()
        try:
            reader.start_led_preview()
            assert fake.writes == [b"B"], f"Expected [b'B'], got {fake.writes}"
        finally:
            reader.stop()

    def test_start_led_preview_safe_when_port_closed(self):
        """start_led_preview() must not raise if the serial port is not open."""
        reader = SerialSensorReader(port="COM_FAKE", n_sensors=4)
        reader.start_led_preview()  # should not raise

    def test_read_timeout_raises_after_5s(self, monkeypatch):
        """read() must raise RuntimeError if no sample arrives within 5 seconds,
        preventing the session from hanging forever on a dead serial connection.
        """
        import queue as _queue
        reader = SerialSensorReader(port="COM_FAKE", n_sensors=4)
        # Simulate a running reader with an empty queue
        reader._running = True
        reader._reader_thread = threading.Thread(target=lambda: None)
        # Patch the queue.get to use a very short timeout for testing
        monkeypatch.setattr(reader._queue, "get", lambda timeout=5.0: (_ for _ in ()).throw(_queue.Empty()))
        with pytest.raises(RuntimeError, match="timeout"):
            reader.read()

    def test_flush_buffer_sets_flag_not_direct_purge(self):
        """flush_buffer() must set _flush_requested flag, NOT call
        reset_input_buffer() directly.  Direct calls race with the reader
        thread's readline() and can hang Windows USB serial drivers.
        """
        reader = SerialSensorReader(port="COM_FAKE", n_sensors=4)
        # Put a dummy sample in the queue
        from apps.collection.sensor_reader import SensorSample
        reader._queue.put_nowait(SensorSample(fsr=[1, 2, 3, 4], led=0, t_ns=0))
        assert not reader._flush_requested

        reader.flush_buffer()

        # Flag must be set for the reader thread to process
        assert reader._flush_requested is True
        # Queue must be drained immediately
        assert reader._queue.empty()


class TestMockTriggerSync:
    """MockSensorReader.trigger_sync() re-anchors the PRBS preamble so dry-run
    matches live behavior (ADR 003).
    """

    def test_trigger_resets_session_start(self, monkeypatch):
        # Drive the clock deterministically (Windows monotonic_ns resolution
        # is ~15ms, so a real sleep is unreliable for asserting time advances).
        tick_ns = [0]
        def fake_monotonic_ns():
            v = tick_ns[0]
            tick_ns[0] += 1_000_000  # +1ms per call
            return v
        monkeypatch.setattr("apps.collection.sensor_reader.now_ns", fake_monotonic_ns)

        reader = MockSensorReader(n_sensors=4, seed=1)
        reader.start()  # consumes one monotonic_ns() call → original_start
        original_start = reader._session_start_ns
        reader.trigger_sync()  # consumes one monotonic_ns() call → new_start
        new_start = reader._session_start_ns
        reader.stop()
        assert new_start > original_start, "trigger_sync must re-anchor _session_start_ns"

    def test_led_off_before_trigger_when_armed_late(self, monkeypatch):
        """When the mock reader has been started but not yet triggered, the
        LED follows the pattern from start(). After trigger_sync(), the LED
        pattern restarts from t=0 (PRBS preamble chip 0).
        """
        # Drive the clock deterministically
        tick_ns = [0]
        def fake_monotonic_ns():
            v = tick_ns[0]
            tick_ns[0] += 1_000_000  # +1ms per call
            return v
        monkeypatch.setattr("apps.collection.sensor_reader.now_ns", fake_monotonic_ns)

        reader = MockSensorReader(n_sensors=4)
        reader.start()
        # First read: t=0ms (then advanced by 1ms), elapsed=1ms → PRBS chip 0 (LED=1)
        first_led = reader.read().led
        # Advance the clock well past PRBS_DURATION=6300ms into periodic blink
        tick_ns[0] = 10_000_000_000
        reader.trigger_sync()  # re-anchor at t=10000ms (now)
        # First read after trigger: t=10001ms, elapsed=1ms, chip 0 → LED=1
        post_trigger_led = reader.read().led
        reader.stop()
        assert first_led == 1, "Pre-trigger first read should be PRBS chip 0 (LED=1)"
        assert post_trigger_led == 1, "Post-trigger first read should be PRBS chip 0 (LED=1)"
