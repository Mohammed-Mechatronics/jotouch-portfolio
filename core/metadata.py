"""Loaders for BIDS metadata files.

Reads:
  - dataset_description.json   (dataset-level)
  - participants.tsv           (dataset-level)
  - sessions.tsv               (dataset-level)
  - sub-P01_ses-S01_physio.json (session-level)
  - sub-P01_ses-S01_channels.tsv (session-level)
  - sub-P01_ses-S01_led_sync.json (session-level)
  - sub-P01_ses-S01_precollect.json (session-level)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import paths


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class Participant:
    participant_id: str       # "sub-P01"
    age: str = "n/a"
    sex: str = "n/a"
    handedness: str = "n/a"
    forearm_circumference_mm: str = "n/a"
    forearm_length_mm: str = "n/a"


@dataclass
class SessionInfo:
    participant_id: str       # "sub-P01"
    session_id: str           # "ses-S01"
    acq_time: str = "n/a"
    band_placement: str = "n/a"
    band_tension: str = "n/a"
    sensor_count: str = "n/a"
    sampling_frequency_hz: str = "n/a"


@dataclass
class ChannelInfo:
    channel_name: str
    sensor_id: str = ""
    type: str = ""
    units: str = ""
    placement_description: str = ""
    target_muscle: str = ""


@dataclass
class PhysioMetadata:
    sampling_frequency: float | None = None
    manufacturer: str = ""
    manufacturers_model_name: str = ""
    software_version: str = ""
    placement_scheme: str = ""
    placement_description: str = ""
    band_placement: str = ""
    band_tension: str = ""
    sensor_count: int = 4
    camera_fps: float | None = None
    camera_manufacturer: str = ""
    camera_model: str = ""
    task_list: list[str] = field(default_factory=list)
    instructions: str = ""


@dataclass
class LedSyncMetadata:
    calibration_time: str = ""
    method: str = ""
    fsr_sampling_hz: float | None = None
    camera_fps: float | None = None
    sync_skew_raw_ms: dict[str, float | None] = field(default_factory=dict)
    sync_skew_corrected_ms: dict[str, float | None] = field(default_factory=dict)
    # Fields from sync_check.py write_bids_led_sync()
    passed: bool | None = None
    skew_ms: float | None = None
    abs_correlation: float | None = None
    n_samples: int | None = None
    n_matched_pairs: int | None = None
    mean_tolerance_ms: float | None = None
    std_tolerance_ms: float | None = None
    residuals_ms: list[float] = field(default_factory=list)
    # Linear correction: t_cam_corrected = a * t_cam + b
    a: float | None = None
    b: float | None = None
    offset_ms: float | None = None
    # Hybrid PRBS + NAd fields (additive, backward-compatible)
    prbs_offset_ms: float | None = None
    prbs_score: float | None = None
    nad_offset_ms: float | None = None
    nad_drift_ppm: float | None = None
    n_windows: int | None = None
    cross_validation_passed: bool | None = None
    # FFT xcorr fields (additive, backward-compatible)
    offset_s: float | None = None
    offset_ms: float | None = None
    score: float | None = None
    drift_ppm: float | None = None


@dataclass
class PrecollectMetadata:
    test_time: str = ""
    sensor_specific: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    regression_specific: dict[str, Any] = field(default_factory=dict)


# ── Dataset-level loaders ─────────────────────────────────────────────────────


def load_dataset_description(data_root: Path | None = None) -> dict:
    """Load dataset_description.json."""
    root = data_root or paths.SAMPLE_DIR
    path = root / "dataset_description.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_participants(data_root: Path | None = None) -> list[Participant]:
    """Load participants.tsv and return a list of Participant objects."""
    root = data_root or paths.SAMPLE_DIR
    path = root / "participants.tsv"
    if not path.exists():
        return []
    df = pd.read_csv(path, sep="\t")
    participants = []
    for _, row in df.iterrows():
        participants.append(Participant(
            participant_id=str(row.get("participant_id", "")),
            age=str(row.get("age", "n/a")),
            sex=str(row.get("sex", "n/a")),
            handedness=str(row.get("handedness", "n/a")),
            forearm_circumference_mm=str(row.get("forearm_circumference_mm", "n/a")),
            forearm_length_mm=str(row.get("forearm_length_mm", "n/a")),
        ))
    return participants


def load_sessions(data_root: Path | None = None) -> list[SessionInfo]:
    """Load sessions.tsv and return a list of SessionInfo objects."""
    root = data_root or paths.SAMPLE_DIR
    path = root / "sessions.tsv"
    if not path.exists():
        return []
    df = pd.read_csv(path, sep="\t")
    sessions = []
    for _, row in df.iterrows():
        sessions.append(SessionInfo(
            participant_id=str(row.get("participant_id", "")),
            session_id=str(row.get("session_id", "")),
            acq_time=str(row.get("acq_time", "n/a")),
            band_placement=str(row.get("band_placement", "n/a")),
            band_tension=str(row.get("band_tension", "n/a")),
            sensor_count=str(row.get("sensor_count", "n/a")),
            sampling_frequency_hz=str(row.get("sampling_frequency_hz", "n/a")),
        ))
    return sessions


def list_subjects(data_root: Path | None = None) -> list[str]:
    """List all subject directories (e.g. ['P01', 'P02'])."""
    root = data_root or paths.SAMPLE_DIR
    if not root.exists():
        return []
    subjects = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.startswith("sub-"):
            subjects.append(d.name[4:])  # strip "sub-" prefix
    return subjects


def list_sessions_for_subject(sub: str, data_root: Path | None = None) -> list[str]:
    """List all session directories for a subject (e.g. ['S01', 'S02'])."""
    subj_dir = paths.subject_dir(sub, data_root=data_root)
    if not subj_dir.exists():
        return []
    sessions = []
    for d in sorted(subj_dir.iterdir()):
        if d.is_dir() and d.name.startswith("ses-"):
            sessions.append(d.name[4:])  # strip "ses-" prefix
    return sessions


# ── Session-level loaders ─────────────────────────────────────────────────────


def load_physio_json(sub: str, ses: str, data_root: Path | None = None) -> PhysioMetadata:
    """Load sub-P01_ses-S01_physio.json."""
    sdir = paths.session_dir(sub, ses, data_root=data_root)
    # Find the physio.json file (name may vary slightly)
    candidates = list(sdir.glob(f"sub-{sub}_ses-{ses}_physio.json"))
    if not candidates:
        return PhysioMetadata()
    with open(candidates[0], encoding="utf-8") as f:
        data = json.load(f)
    return PhysioMetadata(
        sampling_frequency=data.get("SamplingFrequency"),
        manufacturer=data.get("Manufacturer", ""),
        manufacturers_model_name=data.get("ManufacturersModelName", ""),
        software_version=data.get("SoftwareVersion", ""),
        placement_scheme=data.get("PlacementScheme", ""),
        placement_description=data.get("PlacementDescription", ""),
        band_placement=data.get("BandPlacement", ""),
        band_tension=data.get("BandTension", ""),
        sensor_count=data.get("SensorCount", 4),
        camera_fps=data.get("CameraFPS"),
        camera_manufacturer=data.get("CameraManufacturer", ""),
        camera_model=data.get("CameraModel", ""),
        task_list=data.get("TaskList", []),
        instructions=data.get("Instructions", ""),
    )


def load_channels(sub: str, ses: str, data_root: Path | None = None) -> list[ChannelInfo]:
    """Load sub-P01_ses-S01_channels.tsv."""
    sdir = paths.session_dir(sub, ses, data_root=data_root)
    candidates = list(sdir.glob(f"sub-{sub}_ses-{ses}_channels.tsv"))
    if not candidates:
        return []
    df = pd.read_csv(candidates[0], sep="\t")
    channels = []
    for _, row in df.iterrows():
        channels.append(ChannelInfo(
            channel_name=str(row.get("channel_name", "")),
            sensor_id=str(row.get("sensor_id", "")),
            type=str(row.get("type", "")),
            units=str(row.get("units", "")),
            placement_description=str(row.get("placement_description", "")),
            target_muscle=str(row.get("target_muscle", "")),
        ))
    return channels


def load_led_sync(sub: str, ses: str, data_root: Path | None = None) -> LedSyncMetadata:
    """Load sub-P01_ses-S01_led_sync.json."""
    sdir = paths.session_dir(sub, ses, data_root=data_root)
    candidates = list(sdir.glob(f"sub-{sub}_ses-{ses}_led_sync.json"))
    if not candidates:
        return LedSyncMetadata()
    with open(candidates[0], encoding="utf-8") as f:
        data = json.load(f)
    return LedSyncMetadata(
        calibration_time=data.get("calibration_time", data.get("timestamp_utc", "")),
        method=data.get("method", ""),
        fsr_sampling_hz=data.get("fsr_sampling_hz"),
        camera_fps=data.get("camera_fps"),
        sync_skew_raw_ms=data.get("sync_skew_raw_ms", {}),
        sync_skew_corrected_ms=data.get("sync_skew_corrected_ms", {}),
        passed=data.get("passed"),
        skew_ms=data.get("skew_ms"),
        abs_correlation=data.get("abs_correlation"),
        n_samples=data.get("n_samples"),
        n_matched_pairs=data.get("n_matched_pairs"),
        mean_tolerance_ms=data.get("mean_tolerance_ms"),
        std_tolerance_ms=data.get("std_tolerance_ms"),
        residuals_ms=data.get("residuals_ms", []),
        a=data.get("a"),
        b=data.get("b"),
        offset_ms=data.get("offset_ms"),
        # Hybrid PRBS + NAd fields (backward-compatible: old JSON won't have these)
        prbs_offset_ms=data.get("prbs_offset_ms"),
        prbs_score=data.get("prbs_score"),
        nad_offset_ms=data.get("nad_offset_ms"),
        nad_drift_ppm=data.get("nad_drift_ppm"),
        n_windows=data.get("n_windows"),
        cross_validation_passed=data.get("cross_validation_passed"),
    )


def load_precollect(sub: str, ses: str, data_root: Path | None = None) -> PrecollectMetadata:
    """Load sub-P01_ses-S01_precollect.json."""
    sdir = paths.session_dir(sub, ses, data_root=data_root)
    candidates = list(sdir.glob(f"sub-{sub}_ses-{ses}_precollect.json"))
    if not candidates:
        return PrecollectMetadata()
    with open(candidates[0], encoding="utf-8") as f:
        data = json.load(f)
    return PrecollectMetadata(
        test_time=data.get("test_time", ""),
        sensor_specific=data.get("sensor_specific", {}),
        hardware=data.get("hardware", {}),
        regression_specific=data.get("regression_specific", {}),
    )
