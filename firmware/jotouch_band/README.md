# JoTouch Band Firmware

Arduino Nano firmware for the 4-channel FSR band used in the JoTouch portfolio data collection app.

## What it does

- Reads 4 FSR channels via 10-bit ADC (`A0`–`A3`) at **100 Hz**.
- Blinks an LED on pin `D8` at **1 Hz** (100 ms ON, 900 ms OFF) for temporal synchronization with the camera.
- Streams CSV lines over serial at `115200` baud:

```text
fsr0,fsr1,fsr2,fsr3,led_state
```

## Hardware requirements

- Arduino Nano (or compatible 5 V ATmega328P board)
- 4x Interlink FSR-400 sensors (or equivalent force-sensitive resistors)
- 1x LED + current-limiting resistor on pin D8
- External analog reference voltage connected to `AREF` (because `analogReference(EXTERNAL)` is used)

## Known gaps

- **Voltage divider resistor value is not documented in the sketch.** For reproducible force calibration, the pull-down / series resistor value and the `AREF` voltage should be recorded in the hardware notes (see `docs/ARCHITECTURE.md` or the session `physio.json`).
- **LED placement** is not documented. The LED must be visible to the camera inside the calibrated ROI (see `apps/collection/led_roi.py`).

## Flashing

```bash
arduino-cli compile --fqbn arduino:avr:nano firmware/jotouch_band
arduino-cli upload -p COM3 --fqbn arduino:avr:nano firmware/jotouch_band
```

## Origin

This firmware was originally developed in the main JoTouch repo at `JoTouch/firmware/jotouch_band/jotouch_band.ino`. It is copied here so the portfolio repo is self-contained for reproducibility and review.
