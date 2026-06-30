"""Column contracts for BIDS CSV files.

Three modalities per run:
  - _physio.csv  : FSR signals (dynamic channel count)
  - _camera.csv  : MediaPipe hand landmarks (21 × 3 = 63 columns)
  - _targets.csv : Derived joint angles (15 finger DOFs)
"""

from __future__ import annotations

# ── Dataset-wide version constants (single source of truth) ───────────────────
# Update docs/DATA_STRUCTURE.md if these change. Do NOT hardcode these values
# anywhere else (bids_writer.py, validation probes, etc. import from here).

BIDS_VERSION = "1.9.0"
SOFTWARE_VERSION = "1.0.0"

# ── Physio CSV (FSR data) ─────────────────────────────────────────────────────

PHYSIO_TIMESTAMP = "t_monotonic_ns"
PHYSIO_SAMPLE_IDX = "sample_idx"
PHYSIO_PHASE = "phase"
PHYSIO_PARTICIPANT = "participant_id"
PHYSIO_SESSION = "session_id"
PHYSIO_TASK = "task"
PHYSIO_RUN = "run"
PHYSIO_CUE_EVENT = "cue_event"
PHYSIO_LED_FSR = "led_fsr"
PHYSIO_QUALITY_FLAG = "quality_flag"  # 0 = ok, 1 = out-of-range / suspect

PHYSIO_BASE_COLUMNS = [
    PHYSIO_TIMESTAMP,
    PHYSIO_SAMPLE_IDX,
    PHYSIO_PHASE,
    PHYSIO_PARTICIPANT,
    PHYSIO_SESSION,
    PHYSIO_TASK,
    PHYSIO_RUN,
    PHYSIO_CUE_EVENT,
    PHYSIO_LED_FSR,
    PHYSIO_QUALITY_FLAG,
]


def physio_sensor_columns(n_sensors: int) -> list[str]:
    """Return FSR column names for a given sensor count.

    >>> physio_sensor_columns(4)
    ['fsr0', 'fsr1', 'fsr2', 'fsr3']
    """
    return [f"fsr{i}" for i in range(n_sensors)]


def physio_all_columns(n_sensors: int) -> list[str]:
    """Return all columns for a physio CSV (base + sensors).

    Column order:
        timestamp, sample_idx, phase, participant_id, session_id, task, run,
        fsr0..fsrN, cue_event, led_fsr, quality_flag
    """
    return (
        PHYSIO_BASE_COLUMNS[:7]
        + physio_sensor_columns(n_sensors)
        + PHYSIO_BASE_COLUMNS[7:]
    )


# ── Camera CSV (MediaPipe landmarks) ──────────────────────────────────────────

CAMERA_TIMESTAMP = "cam_ts_ns"
CAMERA_VALID = "mp_valid"
CAMERA_CONFIDENCE = "mp_confidence"
CAMERA_HANDEDNESS = "mp_handedness"
CAMERA_LED = "led_cam"
CAMERA_QUALITY_FLAG = "quality_flag"  # 0 = ok, 1 = out-of-range landmark

# 21 landmarks × 3 coords (x, y, z)
LANDMARK_COLUMNS = [
    f"mp_lm{i:02d}_{axis}" for i in range(21) for axis in ("x", "y", "z")
]  # 63 columns

CAMERA_BASE_COLUMNS = [
    CAMERA_TIMESTAMP,
    CAMERA_VALID,
    CAMERA_CONFIDENCE,
    CAMERA_HANDEDNESS,
]

CAMERA_ALL_COLUMNS = CAMERA_BASE_COLUMNS + LANDMARK_COLUMNS + [CAMERA_LED, CAMERA_QUALITY_FLAG]


# ── Targets CSV (joint angles — 15 finger DOFs) ───────────────────────────────

TARGETS_TIMESTAMP = "t_monotonic_ns"
TARGETS_QUALITY_FLAG = "quality_flag"  # 0 = ok, 1 = out-of-range angle

# 15 finger DOFs: thumb (3) + 4 fingers × 3 joints (12) = 15
TARGET_COLUMNS = [
    "target_thumb_cmc_flex",
    "target_thumb_mcp_flex",
    "target_thumb_ip_flex",
    "target_index_mcp_flex",
    "target_index_pip_flex",
    "target_index_dip_flex",
    "target_middle_mcp_flex",
    "target_middle_pip_flex",
    "target_middle_dip_flex",
    "target_ring_mcp_flex",
    "target_ring_pip_flex",
    "target_ring_dip_flex",
    "target_pinky_mcp_flex",
    "target_pinky_pip_flex",
    "target_pinky_dip_flex",
]  # 15 columns

TARGETS_ALL_COLUMNS = [TARGETS_TIMESTAMP] + TARGET_COLUMNS + [TARGETS_QUALITY_FLAG]


# ── Run manifest contract ─────────────────────────────────────────────────────
# Written atomically by BIDSRunWriter.close() as sub-..._run-NN_manifest.json.
# Consumers (core.loader.load_run) skip runs whose manifest is missing or whose
# "complete" flag is not True, so partial runs from a mid-session crash never
# contaminate the ML dataset.

MANIFEST_REQUIRED_KEYS = (
    "sub",
    "ses",
    "task",
    "run",
    "physio_rows",
    "camera_rows",
    "targets_rows",
    "bad_physio_count",
    "bad_camera_count",
    "bad_targets_count",
    "started_at",
    "finished_at",
    "complete",
)


# ── Events TSV (classification labels — post-hoc generated) ───────────────────

EVENTS_ONSET = "onset"
EVENTS_DURATION = "duration"
EVENTS_TRIAL_TYPE = "trial_type"
EVENTS_RUN = "run"

EVENTS_COLUMNS = [EVENTS_ONSET, EVENTS_DURATION, EVENTS_TRIAL_TYPE, EVENTS_RUN]


# ── Task taxonomy ─────────────────────────────────────────────────────────────

# Phase 1: Single-DOF isolation (15 tasks)
SINGLE_DOF_TASKS = [
    "thumbCmcIso", "thumbMcpIso", "thumbIpIso",
    "indexMcpIso", "indexPipIso", "indexDipIso",
    "middleMcpIso", "middlePipIso", "middleDipIso",
    "ringMcpIso", "ringPipIso", "ringDipIso",
    "pinkyMcpIso", "pinkyPipIso", "pinkyDipIso",
]

# Phase 2: Multi-DOF combinations (9 tasks — 6 SHAP + 3 additional)
MULTI_DOF_TASKS = [
    "powerGrip", "tripodGrip", "tipPinch",
    "lateralGrip", "sphericalGrip", "extensionGrip",
    "handOpen", "fingerSpread", "counting",
]

# Phase 3: Freeform
FREEFORM_TASKS = ["freeform"]

# Baseline
BASELINE_TASK = "mvc"

# All tasks
ALL_TASKS = [BASELINE_TASK] + SINGLE_DOF_TASKS + MULTI_DOF_TASKS + FREEFORM_TASKS

# Phase classification
def task_phase(task: str) -> str:
    """Return the phase name for a task.

    >>> task_phase("mvc")
    'baseline'
    >>> task_phase("thumbCmcIso")
    'single_dof'
    >>> task_phase("powerGrip")
    'multi_dof'
    >>> task_phase("freeform")
    'freeform'
    """
    if task == BASELINE_TASK:
        return "baseline"
    if task in SINGLE_DOF_TASKS:
        return "single_dof"
    if task in MULTI_DOF_TASKS:
        return "multi_dof"
    if task in FREEFORM_TASKS:
        return "freeform"
    return "unknown"


# ── Validation helpers ────────────────────────────────────────────────────────

def validate_physio_columns(columns: list[str], n_sensors: int | None = None) -> list[str]:
    """Check that a physio CSV has required columns. Returns list of missing columns."""
    required = list(PHYSIO_BASE_COLUMNS[:7])  # timestamp through run
    if n_sensors is not None:
        required = required + physio_sensor_columns(n_sensors)
    # cue_event, led_fsr, quality_flag are always required
    required = required + PHYSIO_BASE_COLUMNS[7:]
    missing = [c for c in required if c not in columns]
    return missing


def validate_camera_columns(columns: list[str]) -> list[str]:
    """Check that a camera CSV has required columns. Returns list of missing columns."""
    missing = [c for c in CAMERA_ALL_COLUMNS if c not in columns]
    return missing


def validate_targets_columns(columns: list[str]) -> list[str]:
    """Check that a targets CSV has required columns. Returns list of missing columns."""
    missing = [c for c in TARGETS_ALL_COLUMNS if c not in columns]
    return missing
