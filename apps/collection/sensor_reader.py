"""Sensor readers for FSR + LED data.

Provides a uniform interface for mock (dry-run) and real serial hardware.
Each reader returns a ``SensorSample`` with FSR values and the LED state.

The real reader parses the Arduino CSV line:

    fsr0,fsr1,fsr2,fsr3,led_state

Usage:
    reader = create_sensor_reader(port="COM3", baud=115200, n_sensors=4)
    reader.start()
    sample = reader.read()
    print(sample.fsr, sample.led)
    reader.stop()
"""

from __future__ import annotations

import logging
import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from apps.collection.timer import now_ns

logger = logging.getLogger(__name__)


@dataclass
class SensorSample:
    """One FSR sample with LED state."""

    fsr: list[int]
    led: int
    t_ns: int


class SensorReader(Protocol):
    """Protocol for sensor readers."""

    def start(self) -> bool:
        ...

    def stop(self) -> None:
        ...

    def read(self) -> SensorSample:
        ...

    def read_sensors(self) -> list[int]:
        """Return only FSR values (compatibility with precollect tests)."""
        return self.read().fsr


def _mock_led_state(t_ns: int) -> int:
    """Return the LED state for a 1 Hz blink with 100 ms ON.

    .. deprecated:: Use :func:`prbs.led_state_ns` for PRBS-aware LED simulation.
    """
    ms = (t_ns // 1_000_000) % 1000
    return 1 if ms < 100 else 0


class MockSensorReader:
    """Mock sensor reader for dry-run mode.

    Returns plausible random FSR values and a simulated LED blink.
    The LED follows the PRBS preamble (first 6.3 s) then switches to
    periodic 1 Hz blink, matching the Arduino firmware behavior.
    """

    def __init__(self, n_sensors: int = 4, seed: int | None = None) -> None:
        self.n_sensors = n_sensors
        self._rng = random.Random(seed)
        self._running = False
        self._session_start_ns: int | None = None

    def start(self) -> bool:
        self._running = True
        self._session_start_ns = now_ns()
        return True

    def stop(self) -> None:
        self._running = False

    def flush_buffer(self) -> None:
        """No-op for mock reader (interface parity with SerialSensorReader)."""
        pass

    def trigger_sync(self) -> None:
        """Re-anchor the PRBS preamble to NOW.

        Mirrors the ``SerialSensorReader.trigger_sync()`` contract: the LED
        pattern restarts its PRBS preamble from this moment.  Used by the
        session loop right before the first RECORD phase so the recorded
        CSV begins with the PRBS preamble (see ADR 003).
        """
        self._session_start_ns = now_ns()
        logger.info("MockSensorReader: PRBS preamble re-anchored to now")

    def start_led_preview(self) -> None:
        """Start blink-only mode (periodic 1 Hz, no PRBS preamble).

        Mirrors the ``SerialSensorReader.start_led_preview()`` contract.
        Used before ROI calibration and the sync check test so the operator
        can see the LED and the camera can detect brightness changes
        without wasting the PRBS preamble (see ADR 003).
        """
        # Re-anchor to now but mark skip_prbs so the mock LED does periodic
        # blink only.  The mock's led_state_ns doesn't have a skip_prbs
        # parameter, so we just re-anchor — the PRBS preamble will play, but
        # for the mock reader this is harmless (dry-run only).
        self._session_start_ns = now_ns()
        logger.info("MockSensorReader: LED preview started (blink-only)")

    def read(self) -> SensorSample:
        t_ns = now_ns()
        fsr = [self._rng.randint(200, 800) for _ in range(self.n_sensors)]
        if self._session_start_ns is not None:
            from apps.collection.prbs import led_state_ns
            led = led_state_ns(t_ns, self._session_start_ns)
        else:
            led = _mock_led_state(t_ns)
        return SensorSample(fsr=fsr, led=led, t_ns=t_ns)

    def read_sensors(self) -> list[int]:
        return self.read().fsr

    def read_latest(self) -> SensorSample | None:
        """Return the latest sample for UI preview (mock reader is always fresh)."""
        if not self._running:
            return None
        return self.read()

    def read_sensors_preview(self, timeout_s: float = 0.5) -> list[int] | None:
        """Non-blocking read for mock — always returns immediately."""
        sample = self.read_latest()
        return sample.fsr if sample is not None else None


class SerialSensorReader:
    """Real serial sensor reader.

    Reads CSV lines from the Arduino at 115200 baud and parses:
        fsr0,fsr1,fsr2,fsr3,led_state

    Uses ``pyserial.Serial.readline()`` which blocks until a line is
    available, so the loop naturally paces at the Arduino's output rate
    (100 Hz for the JoTouch band firmware).
    """

    def __init__(self, port: str, baud: int = 115200, n_sensors: int = 4) -> None:
        self.port = port
        self.baud = baud
        self.n_sensors = n_sensors
        self._ser = None
        self._running = False
        # One dedicated reader thread owns the serial port. Samples are pushed
        # to this queue; consumers (session) pull from it in order for BIDS
        # recording.  This prevents concurrent readline() calls on the same
        # pyserial object, which crashes the Windows USB driver.
        # Large enough to absorb burst at 100 Hz for ~30 s without blocking.
        # The test loops drain at ~100 Hz so steady-state depth stays near 0.
        self._queue: queue.Queue[SensorSample] = queue.Queue(maxsize=3000)
        self._reader_thread: threading.Thread | None = None
        self._reader_lock = threading.Lock()
        # Atomic reference to the most recent sample. Used by the live preview
        # so it can show the newest value without draining the FIFO queue.
        self._latest_sample: SensorSample | None = None
        self._latest_lock = threading.Lock()
        # Set to True by the session thread when it starts consuming the queue.
        # While False (Setup preview only) the reader thread never blocks on
        # queue.put — it discards the oldest queued sample instead.  This
        # prevents the 1-second stall caused by queue.put(timeout=1.0) when
        # nobody is draining the queue.
        self._has_consumer: bool = False
        # Set by flush_buffer() to request that the reader thread purge the
        # serial input buffer at a safe point (between readline() calls).
        # The reader thread checks this flag at the top of each loop iteration.
        self._flush_requested: bool = False

    def _open_serial(self) -> None:
        """Open the serial port and wait for Arduino reset."""
        try:
            import serial as pyserial
        except ImportError as e:
            raise RuntimeError("pyserial not installed. Run: pip install pyserial") from e

        # timeout=0.05 s is enough for a full line at 115200 baud.
        # One line is at most ~25 bytes; at 115200 baud that is < 2 ms.
        # A 50 ms timeout is therefore >25× the line duration and will never
        # return a partial line under normal conditions.  The old 0.1 s value
        # was too long and created an unnecessary 100 ms stall each time the
        # Arduino happened to be between lines.
        self._ser = pyserial.Serial(self.port, self.baud, timeout=0.05)
        # Arduino resets when the serial port is opened. Wait 2 seconds for
        # the bootloader to finish and the sketch to start sending data.
        time.sleep(2.0)
        self._ser.reset_input_buffer()

    def _close_serial(self) -> None:
        """Close the serial port safely."""
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def _reader_loop(self) -> None:
        """Background thread: own the serial port, parse lines, push to queue."""
        expected_parts = self.n_sensors + 1
        while self._running:
            # Check if a flush was requested by flush_buffer().  Drain the
            # in-process queue here (between readline() calls) so the consumer
            # sees only fresh samples.  We never call reset_input_buffer() —
            # see flush_buffer() docstring for why PurgeComm hangs on Windows.
            if self._flush_requested:
                self._flush_requested = False
                # Drain the queue so the consumer gets fresh samples.
                # We intentionally do NOT call ser.reset_input_buffer() here.
                # On Windows USB-CDC drivers, PurgeComm() can hang even between
                # readline() calls if the driver still has internal I/O state.
                # Draining the queue is sufficient: the Arduino continues
                # streaming and any stale bytes in the hardware buffer will
                # be read and parsed in ≤10 ms (one 100 Hz sample interval).
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break
                logger.debug("Reader thread drained queue on flush request")

            with self._reader_lock:
                ser = self._ser
            if ser is None:
                time.sleep(0.01)
                continue

            try:
                raw = ser.readline().decode("utf-8", errors="replace").strip()
            except Exception as exc:
                logger.warning("Serial read failed: %s — attempting reconnect", exc)
                with self._reader_lock:
                    self._close_serial()
                time.sleep(1.0)
                try:
                    self._open_serial()
                    logger.info("Serial reconnect succeeded")
                except Exception as exc2:
                    logger.error("Serial reconnect failed: %s", exc2)
                    # Stop the reader thread; the consumer will see queue empty and error out
                    self._running = False
                continue

            if not raw:
                continue
            parts = raw.split(",")
            if len(parts) != expected_parts:
                continue
            try:
                vals = [int(p.strip()) for p in parts]
            except ValueError:
                continue

            # Capture timestamp AFTER readline returns so it reflects when the
            # data actually arrived, not when the read started.
            t_ns = now_ns()
            fsr = vals[: self.n_sensors]
            led = int(vals[self.n_sensors]) if len(vals) > self.n_sensors else 0
            # Clamp to valid ADC range
            fsr = [max(0, min(1023, v)) for v in fsr]
            led = 1 if led else 0
            sample = SensorSample(fsr=fsr, led=led, t_ns=t_ns)

            # Always keep the latest sample accessible for preview without
            # draining the ordered queue used by the session thread.
            with self._latest_lock:
                self._latest_sample = sample

            # Push to ordered queue only when a session consumer is active.
            # During Setup preview _has_consumer is False and no one drains
            # the queue, so we must never block here.
            #
            # Even when _has_consumer is True we never block on put().
            # Blocking would deadlock: the reader blocks trying to push while
            # the consumer blocks waiting to pop because the reader never gets
            # to produce the next item.  Instead, drop the OLDEST sample and
            # push the new one so the consumer always gets fresh data.
            try:
                self._queue.put_nowait(sample)
            except queue.Full:
                try:
                    self._queue.get_nowait()   # drop oldest
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(sample)
                except queue.Full:
                    pass
                if self._has_consumer:
                    logger.warning("Serial sample queue full — dropping oldest sample")

    def start(self) -> bool:
        """Open the serial port and start the dedicated reader thread."""
        self._open_serial()
        self._running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        return True

    def stop(self) -> None:
        """Stop the reader thread and close the serial port."""
        self._running = False
        if self._reader_thread is not None and self._reader_thread.is_alive():
            try:
                self._reader_thread.join(timeout=0.5)
            except Exception:
                pass
        with self._reader_lock:
            self._close_serial()
        # Drain the queue so a future reader starts clean
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def flush_buffer(self) -> None:
        """Discard queued samples so the next read() gets fresh data.

        Sets a ``_flush_requested`` flag.  The reader thread drains the queue
        at the top of its next loop iteration (between ``readline()`` calls),
        which gives the consumer a clean slate.

        We never call ``ser.reset_input_buffer()`` (Windows ``PurgeComm``).
        On Windows USB-CDC drivers ``PurgeComm`` can hang both while
        ``ReadFile`` is active *and* between reads if the driver has internal
        I/O state.  Draining the in-process queue is sufficient: any stale
        bytes in the hardware FIFO are read and parsed in ≤10 ms (one 100 Hz
        sample interval) and are simply discarded by the consumer.
        """
        self._flush_requested = True
        # Drain the sample queue immediately so the consumer doesn't see
        # stale data while waiting for the reader thread to process the flag.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def trigger_sync(self) -> None:
        """Send the ``S`` command to the Arduino to arm the PRBS preamble.

        The firmware sits idle (LED OFF, FSR streaming) until it receives
        ``b"S"``.  On receipt, it sets its internal ``s_start_ms = millis()``
        and the LED begins the 6.3 s PRBS preamble followed by the periodic
        1 Hz blink.  This must be called once, immediately before the first
        RECORD phase, so the recorded CSV starts with PRBS chips and
        ``led_sync.py`` Stage 1 can disambiguate direction (see ADR 003).

        Safe to call from the session thread while the reader thread is
        running: the write is guarded by ``self._reader_lock``, the same
        lock the reader thread holds while accessing ``self._ser``.
        """
        with self._reader_lock:
            if self._ser is not None:
                try:
                    # write() hands the byte to the OS driver for transmission.
                    # We deliberately do NOT call flush()/flush_output(): the
                    # former can block on a stalled Windows COM port, the latter
                    # does not exist on pyserial.Serial.
                    self._ser.write(b"S")
                    logger.info("Sent 'S' trigger to Arduino — PRBS preamble armed")
                except Exception as exc:
                    logger.warning("Failed to send 'S' trigger to Arduino: %s", exc)
            else:
                logger.warning("trigger_sync() called but serial port is not open")

    def start_led_preview(self) -> None:
        """Send the ``B`` command to start blink-only mode (no PRBS preamble).

        The firmware starts a periodic 1 Hz blink (100 ms ON) immediately.
        Used before ROI calibration and the sync check test so the operator
        can see the LED and the camera can detect brightness changes
        without wasting the PRBS preamble (see ADR 003).

        Safe to call from any thread — uses the same ``self._reader_lock``
        as ``trigger_sync()``.
        """
        with self._reader_lock:
            if self._ser is not None:
                try:
                    # write() hands the byte to the OS driver for transmission.
                    # No flush() — it can block on a stalled Windows COM port.
                    self._ser.write(b"B")
                    logger.info("Sent 'B' to Arduino — LED blink-only mode started")
                except Exception as exc:
                    logger.warning("Failed to send 'B' to Arduino: %s", exc)
            else:
                logger.warning("start_led_preview() called but serial port is not open")

    def attach_consumer(self) -> None:
        """Signal that a session thread is about to start reading the queue.

        Must be called before the first ``read()`` so the reader thread knows
        it must block rather than discard samples.  Call ``detach_consumer()``
        when the session ends.
        """
        self._has_consumer = True
        # Drain any stale preview samples so the session sees fresh data.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def detach_consumer(self) -> None:
        """Signal that the session thread has stopped reading the queue."""
        self._has_consumer = False

    def read(self) -> SensorSample:
        """Block until a sample is available from the reader thread queue.

        Raises ``RuntimeError`` if no sample arrives within 5 seconds, which
        typically means the reader thread has crashed or the serial port
        disconnected.  This prevents the session from hanging forever on a
        dead queue.
        """
        if not self._running and self._reader_thread is None:
            raise RuntimeError("Serial sensor reader not started")
        try:
            sample = self._queue.get(timeout=5.0)
        except queue.Empty:
            raise RuntimeError(
                "Serial sensor reader timeout — no sample received in 5s. "
                "The Arduino may have disconnected or the reader thread crashed."
            )
        if sample is None:
            raise RuntimeError("Serial reader stopped")
        return sample

    def read_sensors(self) -> list[int]:
        return self.read().fsr

    def read_latest(self) -> SensorSample | None:
        """Return the most recent sample without blocking or draining the queue.

        This is the right method for UI preview because it never removes the
        ordered queue samples that the session thread needs for recording.
        """
        if not self._running:
            return None
        with self._latest_lock:
            return self._latest_sample

    def read_sensors_preview(self, timeout_s: float = 0.5) -> list[int] | None:
        """Non-blocking read for UI preview. Returns the latest sample values.

        Uses the atomic latest-sample reference so the preview always shows
        the newest value without draining the FIFO queue. Does NOT touch the
        serial port directly, so it is safe to call while the session thread
        is also reading.
        """
        sample = self.read_latest()
        if sample is not None:
            return sample.fsr
        return None


def create_sensor_reader(
    *,
    port: str | None = None,
    baud: int = 115200,
    n_sensors: int = 4,
    dry_run: bool = False,
) -> SensorReader:
    """Factory for sensor readers."""
    if dry_run or port is None:
        return MockSensorReader(n_sensors=n_sensors)
    return SerialSensorReader(port=port, baud=baud, n_sensors=n_sensors)
