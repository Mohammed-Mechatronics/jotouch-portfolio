"""PRBS generation and FFT cross-correlation for LED synchronization.

This module provides the coarse acquisition stage of the hybrid LED sync:
  - A 63-bit maximal-length sequence (m-sequence) from a 6-stage LFSR
    with primitive polynomial x^6 + x + 1 (taps at positions 1 and 6).
  - FFT cross-correlation with zero-padding and parabolic interpolation
    for unambiguous offset estimation.

The m-sequence has ideal circular autocorrelation: peak = 63 at lag 0,
exactly -1 at all other 62 lags. This eliminates the direction ambiguity
that plagues periodic-signal synchronization.

Used by:
  - apps/collection/sensor_reader.py  (mock LED simulation)
  - apps/collection/camera.py         (mock camera LED simulation)
  - apps/collection/precollect.py     (real-time sync check)
  - apps/collection/led_sync.py       (offline hybrid sync)
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft, ifft

# ── Constants ─────────────────────────────────────────────────────────────────

PRBS_LENGTH = 63          # 2^6 - 1 (6-stage LFSR)
PRBS_CHIP_S = 0.100       # 100 ms per chip (3 camera frames at 30 Hz)
PRBS_DURATION_S = PRBS_LENGTH * PRBS_CHIP_S  # 6.3 seconds

# Periodic blink parameters (after PRBS preamble)
PERIODIC_PERIOD_S = 1.0   # 1 Hz blink
PERIODIC_ON_S = 0.100     # 100 ms ON time

# NAd search parameters
NAD_HALF_WINDOW_S = 0.200  # ±200 ms around PRBS estimate (fixes 500ms edge case)
NAD_WINDOW_S = 60.0        # windowed NAd window size
NAD_STEP_S = 30.0          # windowed NAd step size
CROSS_VAL_GATE_MS = 2.0    # cross-validation: PRBS vs NAd must agree within 2 ms


# ── m-sequence generation ────────────────────────────────────────────────────

def _generate_mls() -> np.ndarray:
    """Generate the 63-bit m-sequence from x^6 + x + 1 (taps [1, 6]).

    Verified: period = 63, circular autocorrelation peak = 63 at lag 0,
    -1 at all other lags (ideal m-sequence property).
    """
    taps = [1, 6]
    n_stages = 6
    length = 63
    register = np.ones(n_stages, dtype=np.int8)
    seq = np.zeros(length, dtype=np.int8)
    for i in range(length):
        seq[i] = register[0]
        fb = 0
        for t in taps:
            fb ^= register[t - 1]
        register[1:] = register[:-1]
        register[0] = fb
    return seq


# Hardcoded verified sequence (generated once, verified)
PRBS_SEQUENCE: list[int] = _generate_mls().tolist()

# Verify at import time
assert len(PRBS_SEQUENCE) == 63, f"PRBS sequence length {len(PRBS_SEQUENCE)}, expected 63"


# ── LED state functions ──────────────────────────────────────────────────────

def led_state(t_s: float, session_start_s: float = 0.0) -> int:
    """Get LED state at time ``t_s`` (seconds since session start).

    Phase 1 (0 to PRBS_DURATION_S): PRBS preamble.
    Phase 2 (after): 1 Hz periodic blink with 100 ms ON.

    Parameters
    ----------
    t_s : float
        Time in seconds (from session start or monotonic clock).
    session_start_s : float
        Session start time in seconds. If 0, t_s is already relative to start.

    Returns
    -------
    int
        0 or 1 (LED OFF or ON).
    """
    elapsed = t_s - session_start_s
    if elapsed < PRBS_DURATION_S:
        # PRBS preamble
        chip_idx = int(elapsed / PRBS_CHIP_S) % PRBS_LENGTH
        return int(PRBS_SEQUENCE[chip_idx])
    else:
        # Periodic 1 Hz blink
        phase = (elapsed % PERIODIC_PERIOD_S)
        return 1 if phase < PERIODIC_ON_S else 0


def led_state_ns(t_ns: int, session_start_ns: int = 0) -> int:
    """Get LED state at time ``t_ns`` (nanoseconds since session start).

    Convenience wrapper around :func:`led_state` for nanosecond timestamps.
    """
    return led_state(t_ns / 1e9, session_start_ns / 1e9)


def is_prbs_phase(t_s: float, session_start_s: float = 0.0) -> bool:
    """Check if we're in the PRBS preamble phase."""
    return (t_s - session_start_s) < PRBS_DURATION_S


# ── FFT cross-correlation (coarse acquisition) ───────────────────────────────

def prbs_xcorr_offset(
    fsr_t: np.ndarray,
    fsr_s: np.ndarray,
    cam_t: np.ndarray,
    cam_s: np.ndarray,
    duration_s: float = PRBS_DURATION_S,
    search_range_s: float | None = None,
) -> tuple[float, float]:
    """FFT cross-correlation to find the coarse LED delay.

    Convention: returns the delay ``d`` such that ``cam(t) = led(t - d)``.
    Positive ``d`` = camera is delayed (sees old LED state).

    Uses zero-padded FFT for linear (non-circular) cross-correlation,
    plus parabolic interpolation for sub-grid precision.

    Parameters
    ----------
    fsr_t, fsr_s : np.ndarray
        FSR timestamps (seconds) and binary LED values.
    cam_t, cam_s : np.ndarray
        Camera timestamps (seconds) and LED brightness values (0-255 or binary).
    duration_s : float
        Duration of the PRBS signal in seconds.
    search_range_s : float | None
        Search range for the offset. Defaults to ``duration_s``.

    Returns
    -------
    (offset_s, score)
        Estimated delay in seconds and cross-correlation peak magnitude.
    """
    if search_range_s is None:
        search_range_s = duration_s

    dt = 0.001  # 1 ms grid
    t_grid = np.arange(0, duration_s, dt)

    # Interpolate both signals onto the grid (treat -1 as "missing" → 0.5)
    s_fsr = np.interp(t_grid, fsr_t, np.where(fsr_s < 0, 0.5, fsr_s))
    s_cam = np.interp(t_grid, cam_t, np.where(cam_s < 0, 0.5, cam_s))

    # Center (subtract mean)
    s_fsr_c = s_fsr - s_fsr.mean()
    s_cam_c = s_cam - s_cam.mean()

    # Zero-padded FFT cross-correlation (LINEAR, not circular)
    n = len(t_grid)
    n_fft = 2 * n
    ref_p = np.zeros(n_fft)
    sig_p = np.zeros(n_fft)
    ref_p[:n] = s_fsr_c
    sig_p[:n] = s_cam_c

    # xcorr[k] = sum_t ref[t] * sig[t+k]
    # If cam(t) = ref(t - d), peak at k = -d (shift sig left by d)
    xcorr = np.real(ifft(fft(ref_p) * np.conj(fft(sig_p))))

    lags = np.arange(n_fft) * dt
    lags[n_fft // 2:] -= n_fft * dt

    mask = np.abs(lags) <= search_range_s
    if not np.any(mask):
        return 0.0, 0.0

    peak_idx = np.argmax(np.abs(xcorr[mask]))
    # Negate: ifft(fft(ref)*conj(fft(sig))) peaks at k=-d when sig is delayed by d
    offset = -lags[mask][peak_idx]

    # Parabolic interpolation for sub-grid precision
    xc = xcorr[mask]
    if 0 < peak_idx < len(xc) - 1:
        y0, y1, y2 = xc[peak_idx - 1], xc[peak_idx], xc[peak_idx + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-10:
            delta = 0.5 * (y0 - y2) / denom
            offset -= delta * dt  # negate because offset = -lag

    return float(offset), float(abs(xc[peak_idx]))


# ── Directed NAd (fine tracking) ─────────────────────────────────────────────

def directed_nad_offset(
    fsr_edges: np.ndarray,
    cam_edges: np.ndarray,
    prbs_offset: float,
    half_window: float = NAD_HALF_WINDOW_S,
    sps: float = 2000.0,
) -> float:
    """Run NAd in a narrow window centered on the PRBS estimate.

    The PRBS tells us which period the offset falls in. NAd provides
    sub-millisecond precision within that period when ``sps`` is set high
    enough.  With ``sps=2000`` and a ±200ms window, the search grid is
    0.5ms — well within the 2ms target.

    No period clamping is applied — the ±200 ms window is narrower than
    the 500 ms half-period, so there is no ambiguity.

    Parameters
    ----------
    fsr_edges, cam_edges : np.ndarray
        Rising edge timestamps (seconds) for FSR and camera.
    prbs_offset : float
        Coarse offset estimate from PRBS cross-correlation (seconds).
    half_window : float
        Half-width of the search window (±200 ms default).
    sps : float
        Search resolution (time-shifts per second).  Higher = finer
        precision but slower.  Default 2000 → 0.5ms grid.

    Returns
    -------
    float
        Refined offset in seconds, or NaN if NAd fails.
    """
    from nearest_advocate import nearest_advocate as _na

    if len(fsr_edges) < 3 or len(cam_edges) < 3:
        return float('nan')

    td_min = float(prbs_offset - half_window)
    td_max = float(prbs_offset + half_window)

    try:
        res = _na(
            arr_ref=fsr_edges.astype(np.float32),
            arr_sig=cam_edges.astype(np.float32),
            td_min=td_min, td_max=td_max,
            sps=sps,
        )
        if len(res) > 0:
            bi = np.argmin(res[:, 1])
            return float(res[bi, 0])
    except (ValueError, AssertionError):
        pass

    return float('nan')


def windowed_nad_directed(
    fsr_edges: np.ndarray,
    cam_edges: np.ndarray,
    prbs_offset: float,
    win: float = NAD_WINDOW_S,
    step: float = NAD_STEP_S,
) -> tuple[float, float, int]:
    """Windowed NAd with PRBS-guided direction for drift tracking.

    Runs NAd on overlapping windows, using the PRBS offset as the initial
    search center. Each window's result updates the search center for the
    next window, allowing drift tracking.

    Parameters
    ----------
    fsr_edges, cam_edges : np.ndarray
        Rising edge timestamps (seconds) for FSR and camera.
    prbs_offset : float
        Coarse offset from PRBS cross-correlation (seconds).
    win, step : float
        Window size and step (seconds).

    Returns
    -------
    (b, a, n_windows)
        Linear correction parameters where ``t_fsr = a * t_cam + b``,
        and the number of valid windows used.
    """
    if len(fsr_edges) < 3 or len(cam_edges) < 3:
        return float('nan'), float('nan'), 0

    t_start = min(fsr_edges[0], cam_edges[0])
    t_end = max(fsr_edges[-1], cam_edges[-1])

    if t_end - t_start < win:
        # Too short for windowing — single directed NAd
        off = directed_nad_offset(fsr_edges, cam_edges, prbs_offset)
        return off, 1.0, 1

    search_center = prbs_offset
    w_off, w_ctr = [], []

    for s in np.arange(t_start, t_end - win, step):
        e = s + win
        fw = fsr_edges[(fsr_edges >= s) & (fsr_edges < e)]
        cw = cam_edges[(cam_edges >= s) & (cam_edges < e)]
        if len(fw) < 3 or len(cw) < 3:
            continue
        off = directed_nad_offset(fw, cw, search_center)
        if not np.isnan(off):
            w_off.append(off)
            w_ctr.append(s + win / 2)
            search_center = off  # track drift

    if len(w_off) < 3:
        return float('nan'), float('nan'), len(w_off)

    w_off = np.array(w_off)
    w_ctr = np.array(w_ctr)
    slope, intercept = np.polyfit(w_ctr, w_off, 1)

    # NAd convention: phi = cam - fsr (arr_sig shifted by phi to match arr_ref)
    # t_fsr = t_cam - phi(t) = t_cam - (slope*t + intercept)
    # a = 1 - slope, b = -intercept
    a = 1.0 - slope
    b = -intercept

    return float(b), float(a), len(w_off)


def find_rising_edges(timestamps: np.ndarray, values: np.ndarray,
                      threshold: float = 0.5) -> np.ndarray:
    """Return timestamps of rising edges where values cross threshold."""
    if len(values) < 2:
        return np.array([])
    above = values > threshold
    rising = np.where(above[1:] & ~above[:-1])[0] + 1
    return timestamps[rising]
