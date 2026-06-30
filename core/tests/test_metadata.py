"""Tests for core.metadata — BIDS metadata file loading."""

from __future__ import annotations

import json
import pandas as pd
from pathlib import Path

from core.metadata import (
    load_dataset_description,
    load_participants,
    load_sessions,
    list_subjects,
    list_sessions_for_subject,
    load_physio_json,
    load_channels,
    load_led_sync,
    load_precollect,
)


def _make_bids_dataset(tmp_path: Path) -> Path:
    """Create a minimal BIDS dataset for testing."""
    # Dataset-level
    (tmp_path / "dataset_description.json").write_text(json.dumps({
        "Name": "Test Dataset",
        "BIDSVersion": "1.10.0",
    }))

    (tmp_path / "participants.tsv").write_text(
        "participant_id\tage\tsex\thandedness\tforearm_circumference_mm\tforearm_length_mm\n"
        "sub-P01\t28\tM\tR\t250\t230\n"
        "sub-P02\t25\tF\tL\t220\t210\n"
    )

    (tmp_path / "sessions.tsv").write_text(
        "participant_id\tsession_id\tacq_time\tband_placement\tband_tension\tsensor_count\tsampling_frequency_hz\n"
        "sub-P01\tses-S01\t2026-06-23T09:00:00Z\tforearm_2_3\tmedium\t4\t60\n"
    )

    # Session-level
    sdir = tmp_path / "sub-P01" / "ses-S01"
    sdir.mkdir(parents=True)

    (sdir / "sub-P01_ses-S01_physio.json").write_text(json.dumps({
        "SamplingFrequency": 60,
        "Manufacturer": "Interlink",
        "SensorCount": 4,
        "TaskList": ["mvc", "powerGrip", "freeform"],
    }))

    (sdir / "sub-P01_ses-S01_channels.tsv").write_text(
        "channel_name\tsensor_id\ttype\tunits\tplacement_description\ttarget_muscle\n"
        "fsr0\tFSR-001\tFSR\traw\tulnar_side\tflexor_carpi_ulnaris\n"
        "fsr1\tFSR-002\tFSR\traw\tventral_mid\tflexor_digitorum\n"
    )

    (sdir / "sub-P01_ses-S01_led_sync.json").write_text(json.dumps({
        "method": "LED_blink",
        "sync_skew_corrected_ms": {"mean": 32.5, "std": 13.2},
    }))

    (sdir / "sub-P01_ses-S01_precollect.json").write_text(json.dumps({
        "test_time": "2026-06-23T08:55:00Z",
        "sensor_specific": {"creep_drift": {"passed": True}},
    }))

    return tmp_path


class TestDatasetLevel:
    def test_load_dataset_description(self, tmp_path):
        root = _make_bids_dataset(tmp_path)
        desc = load_dataset_description(root)
        assert desc["Name"] == "Test Dataset"

    def test_load_participants(self, tmp_path):
        root = _make_bids_dataset(tmp_path)
        participants = load_participants(root)
        assert len(participants) == 2
        assert participants[0].participant_id == "sub-P01"
        assert participants[0].age == "28"
        assert participants[1].participant_id == "sub-P02"

    def test_load_sessions(self, tmp_path):
        root = _make_bids_dataset(tmp_path)
        sessions = load_sessions(root)
        assert len(sessions) == 1
        assert sessions[0].participant_id == "sub-P01"
        assert sessions[0].session_id == "ses-S01"

    def test_list_subjects(self, tmp_path):
        root = _make_bids_dataset(tmp_path)
        subjects = list_subjects(root)
        # Only sub-P01 has a directory; sub-P02 is only in participants.tsv
        assert subjects == ["P01"]

    def test_list_sessions_for_subject(self, tmp_path):
        root = _make_bids_dataset(tmp_path)
        sessions = list_sessions_for_subject("P01", data_root=root)
        assert sessions == ["S01"]


class TestSessionLevel:
    def test_load_physio_json(self, tmp_path):
        root = _make_bids_dataset(tmp_path)
        meta = load_physio_json("P01", "S01", data_root=root)
        assert meta.sampling_frequency == 60
        assert meta.manufacturer == "Interlink"
        assert meta.sensor_count == 4
        assert "mvc" in meta.task_list

    def test_load_channels(self, tmp_path):
        root = _make_bids_dataset(tmp_path)
        channels = load_channels("P01", "S01", data_root=root)
        assert len(channels) == 2
        assert channels[0].channel_name == "fsr0"
        assert channels[0].target_muscle == "flexor_carpi_ulnaris"

    def test_load_led_sync(self, tmp_path):
        root = _make_bids_dataset(tmp_path)
        sync = load_led_sync("P01", "S01", data_root=root)
        assert sync.method == "LED_blink"
        assert sync.sync_skew_corrected_ms["mean"] == 32.5

    def test_load_led_sync_with_correction_fields(self, tmp_path):
        """Load led_sync.json with the new sync_check.py schema."""
        root = _make_bids_dataset(tmp_path)
        sdir = root / "sub-P01" / "ses-S01"
        (sdir / "sub-P01_ses-S01_led_sync.json").write_text(json.dumps({
            "method": "peaks",
            "passed": True,
            "skew_ms": 12.5,
            "a": 1.001,
            "b": 500_000,
            "offset_ms": 12.5,
            "n_matched_pairs": 5,
            "timestamp_utc": "2026-01-01T00:00:00Z",
        }))
        sync = load_led_sync("P01", "S01", data_root=root)
        assert sync.method == "peaks"
        assert sync.passed is True
        assert sync.skew_ms == 12.5
        assert sync.a == 1.001
        assert sync.b == 500_000
        assert sync.n_matched_pairs == 5
        assert sync.calibration_time == "2026-01-01T00:00:00Z"

    def test_load_precollect(self, tmp_path):
        root = _make_bids_dataset(tmp_path)
        pre = load_precollect("P01", "S01", data_root=root)
        assert pre.test_time == "2026-06-23T08:55:00Z"
        assert pre.sensor_specific["creep_drift"]["passed"] is True

    def test_load_missing_metadata(self, tmp_path):
        meta = load_physio_json("P99", "S99", data_root=tmp_path)
        assert meta.sampling_frequency is None
        assert meta.sensor_count == 4  # default
