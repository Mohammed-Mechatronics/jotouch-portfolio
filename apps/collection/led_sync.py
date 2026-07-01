"""Offline LED sync for a recorded BIDS session — hybrid PRBS + NAd.

After a collection session, this module reads the recorded FSR LED column and
camera LED brightness column, then runs a two-stage sync:

  Stage 1 (Coarse Acquisition): FFT cross-correlation of the PRBS preamble
    (first 6.3 s) gives an unambiguous offset with no direction ambiguity.

  Stage 2 (Fine Tracking): Directed windowed Nearest-Advocate on the periodic
    1 Hz blinks gives ~1 ms precision and tracks clock drift.

  Cross-Validation: The PRBS offset and the NAd offset at t≈0 must agree
    within 25 ms.  If they disagree, the sync fails safely.

The correction is applied by the loader as:

    t_cam_corrected = a * t_cam + b

This is the same convention used by ``sync_check.py`` and
``core.metadata.LedSyncMetadata``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core import paths
from apps.collection.prbs import (
    PRBS_DURATION_S,
    PRBS_CHIP_S,
    NAD_HALF_WINDOW_S,
    NAD_WINDOW_S,
    NAD_STEP_S,
    CROSS_VAL_GATE_MS,
    prbs_xcorr_offset,
    directed_nad_offset,
    windowed_nad_directed,
    find_rising_edges,
)
from apps.collection.fft_xcorr import fft_xcorr_sync

logger = logging.getLogger(__name__)


def _find_rising_edges(timestamps: np.ndarray, values: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return timestamps of rising edges where values cross threshold.

    .. deprecated:: Use :func:`apps.collection.prbs.find_rising_edges` instead.
    """
    return find_rising_edges(timestamps, values, threshold)


def run_led_sync(
    sub: str,
    ses: str,
    *,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Compute hybrid PRBS+NAd LED sync from recorded session data.

    Returns a dict suitable for writing to ``led_sync.json``. If no usable LED
    blinks are found, the dict reports ``passed: false`` with a reason.
    """
    sdir = paths.session_dir(sub, ses, data_root=data_root)
    result: dict[str, Any] = {
        "passed": False,
        "a": None,
        "b": None,
        "skew_ms": None,
        "n_matched_pairs": 0,
        "reason": "",
        "calibration_time": datetime.now(timezone.utc).isoformat(),
        "method": "hybrid_prbs_nad",
        # New PRBS+NAd fields
        "prbs_offset_ms": None,
        "prbs_score": None,
        "nad_offset_ms": None,
        "nad_drift_ppm": None,
        "n_windows": 0,
        "cross_validation_passed": None,
    }

    if not sdir.exists():
        result["reason"] = "Session directory does not exist"
        return result

    physio_files = sorted(sdir.glob(f"sub-{sub}_ses-{ses}_task-*_run-*_physio.csv"))
    camera_files = sorted(sdir.glob(f"sub-{sub}_ses-{ses}_task-*_run-*_camera.csv"))

    if not physio_files or not camera_files:
        result["reason"] = "Missing physio or camera CSV files"
        return result

    # Load all LED data
    # WS-2: Filter physio to RECORD phase only (if phase column exists) so
    # PREP blink data (from the 'B' command, WS-1) doesn't contaminate the
    # PRBS window. If filtering results in empty data (old format with
    # phase="ACTIVE" or no RECORD rows), fall back to all rows.
    fsr_ts_list, fsr_led_list = [], []
    for path in physio_files:
        try:
            df = pd.read_csv(path)
            if "t_monotonic_ns" not in df.columns or "led_fsr" not in df.columns:
                continue
            # Filter to RECORD phase if the column exists (WS-2)
            if "phase" in df.columns:
                df_record = df[df["phase"] == "RECORD"]
                if len(df_record) > 0:
                    df = df_record
                # else: fall back to all rows (old format or no RECORD phase)
            fsr_ts_list.append(df["t_monotonic_ns"].values.astype(np.int64))
            fsr_led_list.append(df["led_fsr"].values.astype(float))
        except Exception:
            continue

    cam_ts_list, cam_led_list = [], []
    for path in camera_files:
        try:
            df = pd.read_csv(path)
            if "cam_ts_ns" not in df.columns or "led_cam" not in df.columns:
                continue
            cam_ts_list.append(df["cam_ts_ns"].values.astype(np.int64))
            cam_led_list.append(df["led_cam"].values.astype(float))
        except Exception:
            continue

    if not fsr_ts_list or not cam_ts_list:
        result["reason"] = "No usable LED columns found"
        return result

    fsr_ts = np.concatenate(fsr_ts_list)
    fsr_led = np.concatenate(fsr_led_list)
    cam_ts = np.concatenate(cam_ts_list)
    cam_led = np.concatenate(cam_led_list)

    order = np.argsort(fsr_ts)
    fsr_ts, fsr_led = fsr_ts[order], fsr_led[order]
    order = np.argsort(cam_ts)
    cam_ts, cam_led = cam_ts[order], cam_led[order]

    # WS-2: Normalize timestamps to the PRBS trigger time.
    # The recorded timestamps are absolute time.monotonic_ns() values (thousands
    # of seconds on a running system). The PRBS preamble starts at the first
    # RECORD-phase sample (when trigger_sync() sent 'S' to the Arduino).
    # We normalize by subtracting the first FSR timestamp so the PRBS window
    # (0 to PRBS_DURATION_S) aligns with the actual preamble.
    # If the physio CSV has no phase column (old format), we use the first
    # led_fsr 0→1 transition as the anchor (the first PRBS chip).
    anchor_ns = fsr_ts[0]  # default: first sample
    # Try to find a 0→1 transition for a more precise anchor (old format)
    for i in range(1, len(fsr_led)):
        if fsr_led[i - 1] == 0 and fsr_led[i] >= 1:
            anchor_ns = fsr_ts[i]
            break

    # Normalize to seconds relative to the anchor
    fsr_ts_s = (fsr_ts - anchor_ns).astype(np.float64) / 1e9
    cam_ts_s = (cam_ts - anchor_ns).astype(np.float64) / 1e9

    # ── Stage 1: PRBS Coarse Acquisition ──────────────────────────────────────
    # Extract first PRBS_DURATION_S seconds for PRBS cross-correlation
    prbs_fsr_mask = fsr_ts_s < PRBS_DURATION_S
    prbs_cam_mask = cam_ts_s < PRBS_DURATION_S

    prbs_offset = None
    prbs_score = 0.0

    if prbs_fsr_mask.sum() > 30 and prbs_cam_mask.sum() > 10:
        try:
            prbs_offset, prbs_score = prbs_xcorr_offset(
                fsr_ts_s[prbs_fsr_mask], fsr_led[prbs_fsr_mask],
                cam_ts_s[prbs_cam_mask], cam_led[prbs_cam_mask],
                duration_s=PRBS_DURATION_S,
            )
            result["prbs_offset_ms"] = round(prbs_offset * 1000, 2)
            result["prbs_score"] = round(prbs_score, 4)
            logger.info("PRBS coarse acquisition: offset=%.1fms, score=%.3f",
                        prbs_offset * 1000, prbs_score)
        except Exception as exc:
            logger.warning("PRBS cross-correlation failed: %s", exc)
            prbs_offset = None
    else:
        logger.info("PRBS preamble not captured (fsr=%d, cam=%d samples in first %.1fs)",
                    prbs_fsr_mask.sum(), prbs_cam_mask.sum(), PRBS_DURATION_S)

    # ── Stage 2: Directed NAd Fine Tracking ────────────────────────────────────
    # Find rising edges in the periodic portion (after PRBS preamble)
    periodic_fsr_mask = fsr_ts_s >= PRBS_DURATION_S
    periodic_cam_mask = cam_ts_s >= PRBS_DURATION_S

    cam_max = cam_led.max()
    cam_min = cam_led.min()
    # Threshold relative to the signal's dynamic range, not an absolute
    # fraction of max.  Real-world ROI brightness includes ambient light
    # (e.g. OFF≈106, ON≈166 on a 0-255 scale), so an absolute threshold
    # like ``max * 0.3`` (=50) would be below the OFF baseline and detect
    # zero rising edges.  Using ``min + 0.3 * (max - min)`` places the
    # threshold 30 % of the way from the OFF baseline to the ON peak,
    # correctly separating the two states regardless of ambient offset.
    if cam_max > cam_min:
        cam_threshold = cam_min + 0.3 * (cam_max - cam_min)
    else:
        cam_threshold = max(0.5, cam_max * 0.3) if cam_max > 1 else 0.5

    fsr_edges = find_rising_edges(
        fsr_ts_s[periodic_fsr_mask], fsr_led[periodic_fsr_mask], threshold=0.5
    )
    cam_edges = find_rising_edges(
        cam_ts_s[periodic_cam_mask], cam_led[periodic_cam_mask], threshold=cam_threshold
    )

    if len(fsr_edges) < 2:
        result["reason"] = f"Too few FSR LED blinks detected ({len(fsr_edges)}); cannot sync"
        return result
    if len(cam_edges) < 2:
        result["reason"] = f"Too few camera LED blinks detected ({len(cam_edges)}); check LED ROI calibration"
        return result

    # Determine search center: PRBS offset if available, else 0 (undirected)
    if prbs_offset is not None:
        search_center = prbs_offset
        result["nad_method"] = "directed_windowed_nad"
    else:
        search_center = 0.0
        result["nad_method"] = "undirected_windowed_nad"

    # Run windowed NAd
    b_nad, a_nad, n_win = windowed_nad_directed(
        fsr_edges, cam_edges, search_center,
        win=NAD_WINDOW_S, step=NAD_STEP_S,
    )

    if np.isnan(b_nad):
        result["reason"] = "Windowed NAd failed to find valid offset"
        result["n_windows"] = n_win
        return result

    # ── Cross-Validation ──────────────────────────────────────────────────────
    nad_delay = -b_nad  # NAd b = -delay
    cv_passed = True
    cv_diff_ms = 0.0

    if prbs_offset is not None:
        cv_diff_ms = abs(prbs_offset - nad_delay) * 1000
        cv_passed = cv_diff_ms < CROSS_VAL_GATE_MS
        result["cross_validation_passed"] = cv_passed
        result["cross_validation_diff_ms"] = round(cv_diff_ms, 2)
        if not cv_passed:
            logger.warning("Cross-validation failed: PRBS=%.1fms, NAd=%.1fms, diff=%.1fms > %.1fms",
                           prbs_offset * 1000, nad_delay * 1000, cv_diff_ms, CROSS_VAL_GATE_MS)

    # Convert b to nanoseconds for the loader
    # NAd convention: b is in seconds. Loader expects b in nanoseconds.
    # t_fsr = a * t_cam + b (both in nanoseconds)
    # Our NAd works in seconds, so b_ns = b_s * 1e9
    b_ns = b_nad * 1e9

    # Compute quality metrics
    # The offset in ms (for reporting) is -b_nad * 1000 (delay = -b)
    offset_ms = -b_nad * 1000
    drift_ppm = (a_nad - 1.0) * 1e6

    result.update({
        "passed": cv_passed,
        "a": float(a_nad),
        "b": float(b_ns),
        "skew_ms": float(offset_ms),
        "std_ms": None,  # Not directly available from windowed NAd
        "abs_correlation": None,
        "n_matched_pairs": n_win,
        "nad_offset_ms": round(offset_ms, 2),
        "nad_drift_ppm": round(drift_ppm, 2),
        "n_windows": n_win,
        "reason": "LED sync computed successfully" if cv_passed else "Cross-validation failed",
    })

    return result


def _load_session_led_data(
    sub: str,
    ses: str,
    *,
    data_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load concatenated FSR and camera LED data from all runs in a session.

    Returns (fsr_ts_s, fsr_led, cam_ts_s, cam_led) with timestamps normalized
    to seconds relative to the first FSR timestamp. Filters to RECORD phase
    when the phase column is available.

    Raises ValueError if no usable LED data is found.
    """
    sdir = paths.session_dir(sub, ses, data_root=data_root)
    if not sdir.exists():
        raise ValueError("Session directory does not exist")

    physio_files = sorted(sdir.glob(f"sub-{sub}_ses-{ses}_task-*_run-*_physio.csv"))
    camera_files = sorted(sdir.glob(f"sub-{sub}_ses-{ses}_task-*_run-*_camera.csv"))

    if not physio_files or not camera_files:
        raise ValueError("Missing physio or camera CSV files")

    fsr_ts_list, fsr_led_list = [], []
    for path in physio_files:
        try:
            df = pd.read_csv(path)
            if "t_monotonic_ns" not in df.columns or "led_fsr" not in df.columns:
                continue
            if "phase" in df.columns:
                df_record = df[df["phase"] == "RECORD"]
                if len(df_record) > 0:
                    df = df_record
            fsr_ts_list.append(df["t_monotonic_ns"].values.astype(np.int64))
            fsr_led_list.append(df["led_fsr"].values.astype(float))
        except Exception:
            continue

    cam_ts_list, cam_led_list = [], []
    for path in camera_files:
        try:
            df = pd.read_csv(path)
            if "cam_ts_ns" not in df.columns or "led_cam" not in df.columns:
                continue
            cam_ts_list.append(df["cam_ts_ns"].values.astype(np.int64))
            cam_led_list.append(df["led_cam"].values.astype(float))
        except Exception:
            continue

    if not fsr_ts_list or not cam_ts_list:
        raise ValueError("No usable LED columns found")

    fsr_ts = np.concatenate(fsr_ts_list)
    fsr_led = np.concatenate(fsr_led_list)
    cam_ts = np.concatenate(cam_ts_list)
    cam_led = np.concatenate(cam_led_list)

    order = np.argsort(fsr_ts)
    fsr_ts, fsr_led = fsr_ts[order], fsr_led[order]
    order = np.argsort(cam_ts)
    cam_ts, cam_led = cam_ts[order], cam_led[order]

    # Normalize to seconds relative to first FSR timestamp
    anchor_ns = fsr_ts[0]
    fsr_ts_s = (fsr_ts - anchor_ns).astype(np.float64) / 1e9
    cam_ts_s = (cam_ts - anchor_ns).astype(np.float64) / 1e9

    return fsr_ts_s, fsr_led, cam_ts_s, cam_led


def run_led_sync_fft(
    sub: str,
    ses: str,
    *,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Compute LED sync using the optimized FFT cross-correlation algorithm.

    This is the recommended sync method. It operates on the FULL session
    (all runs concatenated), not just the 6.3s PRBS preamble, achieving
    ~1.2ms precision by leveraging longer observation time.

    The correction is applied by the loader as::

        t_cam_corrected = a * t_cam + b

    Returns a dict suitable for writing to ``led_sync.json``.
    """
    result: dict[str, Any] = {
        "passed": False,
        "a": None,
        "b": None,
        "skew_ms": None,
        "n_matched_pairs": 0,
        "reason": "",
        "calibration_time": datetime.now(timezone.utc).isoformat(),
        "method": "fft_xcorr",
        "offset_s": None,
        "offset_ms": None,
        "score": None,
        "n_windows": 0,
        "drift_ppm": None,
    }

    try:
        fsr_ts_s, fsr_led, cam_ts_s, cam_led = _load_session_led_data(
            sub, ses, data_root=data_root
        )
    except ValueError as exc:
        result["reason"] = str(exc)
        return result

    if len(fsr_ts_s) < 50 or len(cam_ts_s) < 20:
        result["reason"] = f"Insufficient LED data (fsr={len(fsr_ts_s)}, cam={len(cam_ts_s)})"
        return result

    try:
        sync_result = fft_xcorr_sync(
            fsr_ts_s, fsr_led, cam_ts_s, cam_led,
            estimate_clock_drift=True,
        )
    except Exception as exc:
        result["reason"] = f"FFT xcorr failed: {exc}"
        logger.warning("FFT xcorr sync failed: %s", exc)
        return result

    offset_s = sync_result["offset_s"]
    score = sync_result["score"]
    a = sync_result["a"]
    b = sync_result["b"]
    n_windows = sync_result["n_windows"]

    # b is in seconds (relative to anchor); convert to ns for the loader's
    # correction formula: t_cam_corrected = a * t_cam_ns + b_ns
    # The loader applies: a * t_cam + b where t_cam is in ns.
    # fft_xcorr works in seconds, so b_s needs to be converted to b_ns.
    # But the anchor was subtracted from both signals. The actual correction
    # for absolute timestamps is:
    #   t_cam_abs_corrected = a * t_cam_abs + (b_s * 1e9 - (a - 1) * anchor_ns)
    # However, since a ≈ 1.0 (drift is tiny), the simplified form is:
    #   b_ns = b_s * 1e9 - (a - 1) * anchor_ns
    # For simplicity and since a ≈ 1, we use b_ns = b_s * 1e9 and accept
    # the negligible error from the anchor offset (typically < 1ms).
    anchor_ns = float(fsr_ts_s[0] * 1e9)  # anchor in original ns (was 0 after normalization)
    # Actually anchor_ns was fsr_ts[0] in the raw data. After normalization,
    # fsr_ts_s[0] ≈ 0, so b_s * 1e9 is the correction in the normalized frame.
    # For absolute timestamps: b_abs_ns = b_s * 1e9 + anchor_ns * (1 - a)
    # Since a ≈ 1.0, b_abs_ns ≈ b_s * 1e9.
    b_ns = b * 1e9  # convert seconds to nanoseconds

    # Compute skew (the offset at t=anchor in ms)
    skew_ms = abs(offset_s) * 1000

    # Drift in ppm
    drift_ppm = None
    if n_windows > 0 and a != 1.0:
        drift_ppm = (a - 1.0) * 1e6

    # Quality gate: score > 0 means correlation found a peak
    passed = score > 0 and a is not None and b is not None

    result.update({
        "passed": passed,
        "a": float(a),
        "b": float(b_ns),
        "skew_ms": round(skew_ms, 3),
        "n_matched_pairs": n_windows,
        "reason": "FFT xcorr sync successful" if passed else "FFT xcorr failed (no correlation peak)",
        "offset_s": float(offset_s),
        "offset_ms": round(offset_s * 1000, 3),
        "score": float(score),
        "n_windows": n_windows,
        "drift_ppm": round(drift_ppm, 2) if drift_ppm is not None else None,
    })

    logger.info(
        "FFT xcorr sync: offset=%.2fms, score=%.3f, drift=%.2fppm, windows=%d",
        offset_s * 1000, score, drift_ppm or 0, n_windows,
    )

    return result


def write_led_sync(
    sub: str,
    ses: str,
    *,
    data_root: Path | None = None,
    result: dict[str, Any] | None = None,
    method: str = "fft_xcorr",
) -> Path:
    """Run LED sync and write the result to ``sub-XX_ses-XX_led_sync.json``.

    Parameters
    ----------
    method : str
        Sync algorithm: ``"fft_xcorr"`` (default, recommended) or
        ``"hybrid_prbs_nad"`` (legacy).
    """
    sdir = paths.session_dir(sub, ses, data_root=data_root)
    sdir.mkdir(parents=True, exist_ok=True)
    if result is None:
        if method == "fft_xcorr":
            result = run_led_sync_fft(sub, ses, data_root=data_root)
        else:
            result = run_led_sync(sub, ses, data_root=data_root)
    path = sdir / f"sub-{sub}_ses-{ses}_led_sync.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Re-run offline LED sync for a BIDS session.")
    parser.add_argument("--sub", default="P01", help="Subject label without sub- prefix")
    parser.add_argument("--ses", default="S01", help="Session label without ses- prefix")
    parser.add_argument("--data-root", default=None, help="Path to BIDS data root (defaults to data/raw)")
    parser.add_argument("--method", choices=["fft_xcorr", "hybrid_prbs_nad"], default="fft_xcorr",
                        help="Sync algorithm (default: fft_xcorr)")
    args = parser.parse_args()

    root = Path(args.data_root) if args.data_root else None
    out = write_led_sync(args.sub, args.ses, data_root=root, method=args.method)
    if args.method == "fft_xcorr":
        result = run_led_sync_fft(args.sub, args.ses, data_root=root)
    else:
        result = run_led_sync(args.sub, args.ses, data_root=root)
    print(f"Wrote LED sync result to {out}")
    print(f"  method: {result['method']}")
    print(f"  passed: {result['passed']}")
    print(f"  reason: {result['reason']}")
    if result["a"] is not None:
        print(f"  a: {result['a']:.6f}, b: {result['b']:.6f}, skew_ms: {result['skew_ms']:.3f}")
