# JoTouch — FMG Data Collection Workbench

> Force-Myography sensor acquisition system with real-time sensor-camera synchronization.

This repository contains the **data collection workbench** for the JoTouch capstone
project: a low-cost forearm band of Force-Sensitive Resistors (FSRs) that measures
muscle bulging for prosthetic hand control. The workbench handles multi-channel
sensor acquisition, real-time sensor-camera sync via a hybrid PRBS + nearest-advocate
LED method, and writes data in a BIDS-inspired structure.

---

## Quick start

```bash
# 1. Create environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt

# 2. Run the collection app with mock hardware (no Arduino/camera needed)
python -m apps.collection --sub P01 --ses S01 --dry-run

# 3. Browser UI (recommended for operators)
uvicorn apps.collection.api:app --host 0.0.0.0 --port 8000
#   Open http://localhost:8000 in a browser.
#   Four-screen flow: Setup → Tests → Collect → Summary
#   Live FSR bars, camera preview, cue cards, quality alerts.

# 4. Real hardware
python -m apps.collection --sub P01 --ses S01 --port COM3 --camera 0
```

---

## What the collection app does

- **4-channel FSR acquisition** from an Arduino Uno over serial (100 Hz sample rate)
- **Camera capture** with MediaPipe hand-landmark tracking (30 Hz)
- **Hybrid PRBS + nearest-advocate LED synchronization** aligning sensor and camera
  streams to a shared monotonic clock (see below)
- **BIDS-inspired output**: every run produces `physio.csv`, `camera.csv`,
  `targets.csv` + a `manifest.json` with `complete=true`
- **Pre-collection tests** (8 mandatory checks): creep/drift, channel activation,
  dead/stuck channels, baseline stability, response linearity, camera tracking,
  sync check, single-DOF isolation
- **76-run protocol**: 1 MVC baseline + 15 single-DOF isolation tasks (3 reps) +
  9 multi-DOF grasps (3 reps) + 3 freeform runs (~25 min per session)

---

## Temporal synchronization (PRBS + NAd)

Sensor and camera run on independent clocks that drift apart. A hybrid two-stage
LED-based method keeps them aligned:

1. **PRBS coarse acquisition** — a 63-bit maximal-length m-sequence (LFSR with
   primitive polynomial x^6 + x + 1) blinks an LED during the first 6.3 s.
   FFT cross-correlation with parabolic interpolation recovers the coarse offset
   with unambiguous direction (no periodic-signal ambiguity).

2. **Directed NAd fine tracking** — during the session, a 1 Hz periodic LED blink
   is tracked with the nearest-advocate algorithm in a ±200 ms window centered on
   the PRBS estimate, giving sub-millisecond precision and drift tracking.

3. **Cross-validation gate** — PRBS and NAd offsets must agree within 25 ms;
   disagreement flags a quality issue rather than silently corrupting data.

**Precision proof:** the algorithm's precision is verified by a mathematical proof
test (`apps/collection/tests/test_sync_precision_proof.py`) that generates synthetic
LED data with a known offset, adds QPC timestamp quantization (~100 ns on Windows),
runs the full PRBS + NAd pipeline, and asserts the recovered offset is within
**< 2 ms** of the true offset. The proof uses `time.perf_counter_ns()` (QueryPerformanceCounter,
~100 ns resolution) instead of the default 15.6 ms Windows timer — a ~150,000x
improvement in timestamp resolution.

Run the proof:
```bash
python -m pytest apps/collection/tests/test_sync_precision_proof.py -v
```

---

## FFT cross-correlation sync (default method)

The current default sync algorithm is a fully-optimized FFT cross-correlation
(`apps/collection/fft_xcorr.py`) with three refinement stages:

1. **Coarse** — zero-padded FFT cross-correlation on a 0.1 ms grid.
2. **Sub-sample** — Gaussian interpolation of the correlation peak.
3. **Spline refinement** — cubic spline around the peak for final precision.

It operates directly on the LED brightness signal recorded by the camera and
the LED state column from the FSR stream, requiring no PRBS preamble.

Run offline sync on a recorded session:
```bash
python -c "from apps.collection.led_sync import fft_xcorr_sync; print(fft_xcorr_sync('P01','S01'))"
```

The legacy PRBS + NAd pipeline remains available via `apps/collection/prbs.py`
and `led_sync.run_led_sync()`.

---

## Folder map

| Folder | What it contains |
|--------|------------------|
| `apps/collection/` | Collection app: serial reader, camera, BIDS writer, PRBS sync, FFT xcorr sync, timer, pre-collect tests, web UI |
| `core/` | Shared BIDS infrastructure: naming, schema, paths, loader, merge, splits, joint-angle derivation |
| `data/sample/` | Curated BIDS sample (git-tracked) — one session, 76 runs |
| `data/splits/` | Cross-task / cross-subject / cross-session / cross-trial split specs (YAML) |

---

## Data structure (BIDS-inspired)

```
data/sample/sub-P01/ses-S01/
├── sub-P01_ses-S01_physio.json          # channel metadata
├── sub-P01_ses-S01_channels.tsv         # sensor channel definitions
├── sub-P01_ses-S01_led_sync.json        # sync offsets + quality
├── sub-P01_ses-S01_task-mvc_run-00_physio.csv
├── sub-P01_ses-S01_task-mvc_run-00_camera.csv
├── sub-P01_ses-S01_task-mvc_run-00_targets.csv
└── ... (76 runs per session)
```

Each run produces 3 CSVs sharing the same monotonic timestamp column:
- **physio** — FSR signals (100 Hz)
- **camera** — MediaPipe landmarks (30 Hz)
- **targets** — 15 joint angles (100 Hz)

Filenames follow: `sub-{x}_ses-{x}_task-{x}_run-{NN}_{suffix}.csv`

### Unified session record & ML data extraction

The 76 per-run CSVs can be merged into a single `sub-P01_ses-S01_record.csv`
(one row per 100 Hz tick, all sensors + camera + targets in one file) via:

```bash
python promote_to_sample.py          # merge per-run CSVs → unified record
```

From the unified record, extract ML-ready Parquet (raw samples, label column,
reaction-time trim, dual classification + regression targets):

```bash
python extract_ml_data.py            # → data/derived/sub-P01/ses-S01/*.parquet
```

Then apply cross-task / cross-subject / cross-session / cross-trial split specs:

```bash
python apply_splits.py               # → data/splits/results/*.json
```

> The unified record CSV (~387 MB) and derived Parquet are gitignored —
> regenerate them with the commands above.

---

## Testing

```bash
# Collection + core tests
python -m pytest apps/collection/tests/ core/tests/ -q
```

---

## Hardware

- **Sensor band**: 4× Force-Sensitive Resistors on a forearm band
- **Microcontroller**: Arduino Uno (serial over USB, 115200 baud)
- **Camera**: USB webcam (30 fps)
- **Sync LED**: single LED wired to an Arduino digital pin, visible in camera frame
- **Cost**: front-end built for under 20 JOD

---

## Credit

Student capstone project at Al-Balqa' Applied University, selected as one of seven
projects nationwide under the KAFD RISP program, in partnership with JODDB.

---

## License

MIT
