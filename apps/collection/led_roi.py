"""Interactive LED ROI calibration for camera-FSR sync.

Usage:
    python -m apps.collection.led_roi --output led_roi.json
    python -m apps.collection.led_roi --camera 0 --output led_roi.json

Controls:
    Click + drag   : draw the bounding box around the LED
    s              : save ROI to JSON
    r              : reset ROI
    q / ESC        : quit

The saved JSON has keys:
    {
      "x": 100,
      "y": 80,
      "width": 40,
      "height": 40,
      "camera_index": 0,
      "image_width": 640,
      "image_height": 480
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from apps.collection.camera import create_camera_reader


@dataclass
class LedRoi:
    """Normalized and pixel-space ROI around the LED.

    ``transition_threshold`` is the brightness delta (0-255) that counts as
    an LED ON/OFF transition, established during calibration via frame
    differencing.  It is persisted so the precollect sync check and the
    offline LED sync use the same threshold the operator validated.
    """

    x: int
    y: int
    width: int
    height: int
    camera_index: int = 0
    image_width: int = 640
    image_height: int = 480
    transition_threshold: float = 15.0

    def to_normalized(self) -> dict[str, float]:
        """Return ROI coordinates normalized to [0, 1]."""
        return {
            "x": self.x / self.image_width,
            "y": self.y / self.image_height,
            "width": self.width / self.image_width,
            "height": self.height / self.image_height,
        }

    def save(self, path: Path) -> None:
        """Save the ROI as JSON."""
        data = {**asdict(self), "normalized": self.to_normalized()}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LedRoi":
        """Load a previously saved ROI."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data["width"]),
            height=int(data["height"]),
            camera_index=int(data.get("camera_index", 0)),
            image_width=int(data.get("image_width", 640)),
            image_height=int(data.get("image_height", 480)),
            transition_threshold=float(data.get("transition_threshold", 15.0)),
        )


def _order_points(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    """Return (x, y, width, height) from two corner points."""
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return x, y, w, h


def _compute_roi_brightness(image, roi: dict[str, int]) -> float:
    """Compute mean brightness in the ROI of a BGR image.

    Uses the value channel (max across BGR) as a simple brightness proxy,
    matching CameraReader._compute_roi_brightness.
    """
    import numpy as np
    h, w = image.shape[:2]
    x = max(0, min(roi["x"], w - 1))
    y = max(0, min(roi["y"], h - 1))
    width = max(1, min(roi["width"], w - x))
    height = max(1, min(roi["height"], h - y))
    roi_img = image[y:y + height, x:x + width]
    if roi_img.size == 0:
        return 0.0
    return float(np.max(roi_img, axis=2).mean())


class _RoiSelector:
    """OpenCV mouse callback helper for drawing the ROI rectangle."""

    def __init__(self) -> None:
        self.drawing = False
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: None) -> None:
        import cv2
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x1 = x
            self.y1 = y
            self.x2 = x
            self.y2 = y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.x2 = x
            self.y2 = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.x2 = x
            self.y2 = y

    def roi(self, camera_index: int, image_width: int, image_height: int) -> LedRoi | None:
        x, y, w, h = _order_points(self.x1, self.y1, self.x2, self.y2)
        if w == 0 or h == 0:
            return None
        return LedRoi(
            x=x, y=y, width=w, height=h,
            camera_index=camera_index,
            image_width=image_width,
            image_height=image_height,
        )


def calibrate(
    camera_index: int = 0,
    output: Path | None = None,
    dry_run: bool = False,
    camera_reader=None,
) -> LedRoi | None:
    """Open a camera window and let the user select the LED ROI.

    Returns the selected ROI, or None if the user quit without saving.

    Parameters
    ----------
    camera_reader : CameraReader | None
        An already-running shared camera reader to use.  When provided,
        this function will NOT call ``start()`` or ``stop()`` on it —
        the caller owns the lifecycle.  This prevents the calibration
        window from stealing the camera device away from the live
        WebSocket stream.  If None, a private reader is created, started,
        and stopped automatically.
    """
    import cv2

    _owns_camera = camera_reader is None
    if _owns_camera:
        camera_reader = create_camera_reader(dry_run=dry_run, camera_index=camera_index)
        if not camera_reader.start():
            print("ERROR: Could not open camera.")
            return None

    # ── Lock camera exposure during calibration ──────────────────────────
    # Auto-exposure compensates for the LED brightness over several frames,
    # reducing the ON/OFF contrast to near zero.  By locking the exposure
    # to the current (ambient) value, the LED's brightness jumps are
    # preserved and the frame-differencing meter can detect them reliably.
    # Reference: OpenCV GitHub issue #9738 — the convention on Windows MSMF
    # is CAP_PROP_AUTO_EXPOSURE = 0.25 for manual, 0.75 for auto.
    _exposure_locked = False
    if hasattr(camera_reader, "_cap") and camera_reader._cap is not None:
        try:
            # Read current exposure as the baseline, then lock it
            camera_reader._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            _exposure_locked = True
        except Exception:
            pass  # Not all cameras support manual exposure

    selector = _RoiSelector()
    window_name = "LED ROI Calibration"
    # Create a sizeable window and keep it on top so it is visible above other apps
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(window_name, selector.mouse_callback)

    # Instructions shown as an overlay on the video frame
    _INSTRUCTIONS = [
        "Draw a rectangle around the blinking LED (click + drag).",
        "S = save   R = reset   Q / ESC = quit",
        "Keep the LED visible and avoid bright backgrounds.",
    ]

    # ── Live brightness meter (frame differencing) ──────────────────────
    # Instead of guessing the LED state from time (which is unsynchronized
    # with the firmware blink), we use frame differencing:
    #   - Compare ROI brightness of consecutive frames
    #   - Large positive jump = LED turned ON
    #   - Large negative jump = LED turned OFF
    #   - Small change = LED state unchanged (keep last known state)
    # This is immune to auto-exposure drift (which changes slowly over many
    # frames) because LED transitions are instant (1-frame difference).
    # Reference: Stack Overflow accepted answer on blinking light detection,
    # and the burgshrimps/led_detection GitHub project.
    _brightness_on: list[float] = []
    _brightness_off: list[float] = []
    _prev_brightness: float | None = None
    _cur_led_state: int = 0  # start assuming OFF
    _MAX_BRIGHTNESS_SAMPLES = 100  # rolling window
    # Threshold for detecting a transition (in brightness units 0-255).
    # The LED should produce a jump of > 20-30 units; ambient noise is < 5.
    _TRANSITION_THRESHOLD = 15.0

    roi: LedRoi | None = None
    try:
        while True:
            # get_jpeg() returns the pre-encoded frame (with landmark overlay).
            # Decode it here so we can draw the ROI selection rectangle on top
            # for the local calibration window only — the WS stream is unaffected.
            jpeg = camera_reader.get_jpeg()
            if jpeg is None:
                cv2.waitKey(10)
                continue
            frame = cv2.imdecode(
                np.frombuffer(jpeg, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if frame is None:
                cv2.waitKey(10)
                continue

            h, w = frame.shape[:2]
            candidate = selector.roi(camera_index, w, h)

            # ── Compute live brightness for the candidate/saved ROI ───────
            # Use whichever ROI is available (saved > candidate > none).
            _meter_roi = roi if roi is not None else candidate
            _meter_text = ""
            if _meter_roi is not None:
                try:
                    brightness = _compute_roi_brightness(
                        frame,
                        {"x": _meter_roi.x, "y": _meter_roi.y,
                         "width": _meter_roi.width, "height": _meter_roi.height},
                    )

                    # ── Frame differencing to detect LED transitions ──────
                    if _prev_brightness is not None:
                        delta = brightness - _prev_brightness
                        if delta > _TRANSITION_THRESHOLD:
                            # Brightness jumped up → LED turned ON
                            _cur_led_state = 1
                        elif delta < -_TRANSITION_THRESHOLD:
                            # Brightness dropped → LED turned OFF
                            _cur_led_state = 0
                        # else: small change, keep current state
                    _prev_brightness = brightness

                    # Accumulate brightness into ON/OFF buckets based on
                    # the detected LED state
                    if _cur_led_state:
                        _brightness_on.append(brightness)
                    else:
                        _brightness_off.append(brightness)
                    # Trim to rolling window
                    if len(_brightness_on) > _MAX_BRIGHTNESS_SAMPLES:
                        _brightness_on = _brightness_on[-_MAX_BRIGHTNESS_SAMPLES:]
                    if len(_brightness_off) > _MAX_BRIGHTNESS_SAMPLES:
                        _brightness_off = _brightness_off[-_MAX_BRIGHTNESS_SAMPLES:]

                    # Compute meter text
                    if len(_brightness_on) > 3 and len(_brightness_off) > 3:
                        on_mean = float(np.mean(_brightness_on))
                        off_mean = float(np.mean(_brightness_off))
                        diff = on_mean - off_mean
                        contrast = diff / max(off_mean, 1.0)
                        good = contrast > 0.15 and diff > 10
                        status = "OK" if good else "NO"
                        color = (0, 255, 0) if good else (0, 0, 255)
                        _meter_text = (
                            f"ON:{on_mean:.0f}  OFF:{off_mean:.0f}  "
                            f"D:{diff:.0f}  [{status}]"
                        )
                    elif len(_brightness_on) + len(_brightness_off) > 0:
                        _meter_text = (
                            f"Detecting... (ON={len(_brightness_on)}, "
                            f"OFF={len(_brightness_off)})"
                        )
                        color = (255, 255, 0)
                    else:
                        _meter_text = ""
                        color = (255, 255, 0)
                except Exception:
                    _meter_text = ""
                    color = (255, 255, 0)
            else:
                # Reset brightness tracking when no ROI is selected
                _brightness_on.clear()
                _brightness_off.clear()
                _prev_brightness = None
                _cur_led_state = 0

            if candidate is not None:
                x2 = candidate.x + candidate.width
                y2 = candidate.y + candidate.height
                cv2.rectangle(frame, (candidate.x, candidate.y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{candidate.width}x{candidate.height}",
                    (candidate.x, candidate.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            if roi is not None:
                x2 = roi.x + roi.width
                y2 = roi.y + roi.height
                cv2.rectangle(frame, (roi.x, roi.y), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "SAVED",
                    (roi.x, roi.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            # Draw instruction overlay at the top of the frame
            y_offset = 20
            for line in _INSTRUCTIONS:
                cv2.putText(
                    frame,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                y_offset += 22

            # Draw brightness meter at the bottom of the frame
            if _meter_text:
                meter_y = h - 30
                # Background bar for readability
                (tw, th), _ = cv2.getTextSize(
                    _meter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame,
                    (5, meter_y - th - 5),
                    (5 + tw + 10, meter_y + 5),
                    (0, 0, 0), -1,
                )
                cv2.putText(
                    frame,
                    _meter_text,
                    (10, meter_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("r"):
                selector.x1 = selector.y1 = selector.x2 = selector.y2 = 0
                roi = None
            elif key == ord("s"):
                candidate = selector.roi(camera_index, w, h)
                if candidate is not None:
                    # Stamp the transition threshold used during calibration
                    # onto the saved ROI so the precollect sync check and the
                    # offline LED sync reuse the operator-validated value.
                    candidate.transition_threshold = _TRANSITION_THRESHOLD
                    roi = candidate
                    if output:
                        roi.save(output)
                        print(f"Saved ROI to {output} (threshold={_TRANSITION_THRESHOLD})")
                else:
                    print("No ROI selected; click and drag to draw one.")
    finally:
        # Restore auto-exposure if we locked it
        if _exposure_locked and hasattr(camera_reader, "_cap") and camera_reader._cap is not None:
            try:
                import cv2 as _cv2
                camera_reader._cap.set(_cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            except Exception:
                pass
        # Only stop the camera if we created it — never stop a shared reader.
        if _owns_camera:
            camera_reader.stop()
        cv2.destroyAllWindows()

    return roi


def main() -> int:
    parser = argparse.ArgumentParser(description="LED ROI calibration")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--output", type=str, default="led_roi.json", help="Output JSON path")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic camera feed")
    args = parser.parse_args()

    output = Path(args.output)
    roi = calibrate(camera_index=args.camera, output=output, dry_run=args.dry_run)
    if roi is None:
        print("No ROI saved.")
        return 1
    print(f"ROI: {roi.x},{roi.y} {roi.width}x{roi.height}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
