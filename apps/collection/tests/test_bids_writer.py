"""Tests for apps.collection.bids_writer — BIDS CSV writer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from apps.collection.bids_writer import (
    BIDSRunWriter,
    write_dataset_description,
    write_session_metadata,
    update_mvc_calibration,
    append_participants_tsv,
    append_sessions_tsv,
    manifest_path,
    quarantine_partial_run,
    sweep_orphan_partials,
    sweep_incomplete_sessions,
)
from core import naming, schema
from core.schema import (
    PHYSIO_QUALITY_FLAG,
    CAMERA_QUALITY_FLAG,
    TARGETS_QUALITY_FLAG,
    MANIFEST_REQUIRED_KEYS,
)


class TestBIDSRunWriter:
    def test_open_and_close(self, tmp_path):
        writer = BIDSRunWriter(
            sub="P01", ses="S01", task="powerGrip", run=1,
            n_sensors=4, data_root=tmp_path,
        )
        assert not writer.is_open
        writer.open()
        assert writer.is_open
        result = writer.close()
        assert not writer.is_open
        assert result["physio_rows"] == 0
        assert result["camera_rows"] == 0
        assert result["targets_rows"] == 0

    def test_write_physio(self, tmp_path):
        writer = BIDSRunWriter(
            sub="P01", ses="S01", task="powerGrip", run=1,
            n_sensors=4, data_root=tmp_path,
        )
        writer.open()
        writer.write_physio(1000, 0, [100, 200, 300, 400], phase="ACTIVE")
        writer.write_physio(2000, 1, [150, 250, 350, 450], phase="ACTIVE")
        result = writer.close()
        assert result["physio_rows"] == 2

        # Verify file content
        with open(writer.physio_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "t_monotonic_ns" in header
            assert "fsr0" in header
            assert "fsr3" in header
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0][0] == "1000"
            assert rows[1][0] == "2000"

    def test_write_camera(self, tmp_path):
        writer = BIDSRunWriter(
            sub="P01", ses="S01", task="powerGrip", run=1,
            n_sensors=4, data_root=tmp_path,
        )
        writer.open()
        landmarks = [0.1] * 63
        writer.write_camera(1000, landmarks, valid=True, confidence=0.95)
        result = writer.close()
        assert result["camera_rows"] == 1

        with open(writer.camera_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "cam_ts_ns" in header
            assert "mp_lm00_x" in header
            assert "mp_lm20_z" in header
            rows = list(reader)
            assert len(rows) == 1

    def test_write_targets(self, tmp_path):
        writer = BIDSRunWriter(
            sub="P01", ses="S01", task="powerGrip", run=1,
            n_sensors=4, data_root=tmp_path,
        )
        writer.open()
        writer.write_targets(1000, [10.0] * 15)
        writer.write_targets(2000, {"target_thumb_cmc_flex": 30.0})
        result = writer.close()
        assert result["targets_rows"] == 2

        with open(writer.targets_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "t_monotonic_ns" in header
            assert "target_thumb_cmc_flex" in header
            assert "target_pinky_dip_flex" in header
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0][1] == "10.0"  # first target column

    def test_filename_format(self, tmp_path):
        writer = BIDSRunWriter(
            sub="P01", ses="S01", task="thumbCmcIso", run=3,
            n_sensors=4, data_root=tmp_path,
        )
        writer.open()
        writer.close()
        assert writer.physio_path.name == "sub-P01_ses-S01_task-thumbCmcIso_run-03_physio.csv"
        assert writer.camera_path.name == "sub-P01_ses-S01_task-thumbCmcIso_run-03_camera.csv"
        assert writer.targets_path.name == "sub-P01_ses-S01_task-thumbCmcIso_run-03_targets.csv"

    def test_all_three_files_created(self, tmp_path):
        writer = BIDSRunWriter(
            sub="P01", ses="S01", task="mvc", run=0,
            n_sensors=4, data_root=tmp_path,
        )
        writer.open()
        writer.write_physio(1000, 0, [100, 200, 300, 400])
        writer.write_camera(1000, [0.1] * 63)
        writer.write_targets(1000, [10.0] * 15)
        writer.close()
        assert writer.physio_path.exists()
        assert writer.camera_path.exists()
        assert writer.targets_path.exists()


class TestSessionMetadata:
    def test_write_session_metadata(self, tmp_path):
        # Default sampling_frequency matches the Arduino firmware (100 Hz).
        write_session_metadata("P01", "S01", sensor_count=4, data_root=tmp_path)
        sdir = tmp_path / "sub-P01" / "ses-S01"
        assert (sdir / "sub-P01_ses-S01_physio.json").exists()
        assert (sdir / "sub-P01_ses-S01_channels.tsv").exists()

        with open(sdir / "sub-P01_ses-S01_physio.json") as f:
            meta = json.load(f)
        assert meta["SensorCount"] == 4
        # Must be 100 Hz — the Arduino firmware sends at 100 Hz and session.py
        # always calls write_session_metadata(sampling_frequency=100.0).
        assert meta["SamplingFrequency"] == 100.0

    def test_write_session_metadata_custom_frequency(self, tmp_path):
        # Callers can override the frequency (e.g. legacy 60 Hz hardware).
        write_session_metadata("P02", "S01", sensor_count=4,
                               sampling_frequency=60.0, data_root=tmp_path)
        sdir = tmp_path / "sub-P02" / "ses-S01"
        with open(sdir / "sub-P02_ses-S01_physio.json") as f:
            meta = json.load(f)
        assert meta["SamplingFrequency"] == 60.0

    def test_append_participants(self, tmp_path):
        append_participants_tsv("P01", age="28", sex="M", data_root=tmp_path)
        path = tmp_path / "participants.tsv"
        assert path.exists()
        with open(path) as f:
            content = f.read()
        assert "sub-P01" in content
        assert "28" in content

        # Append another participant
        append_participants_tsv("P02", age="25", sex="F", data_root=tmp_path)
        with open(path) as f:
            content = f.read()
        assert "sub-P02" in content

        # Append same participant again (should not duplicate)
        append_participants_tsv("P01", age="28", sex="M", data_root=tmp_path)
        with open(path) as f:
            lines = f.readlines()
        assert len([l for l in lines if "sub-P01" in l]) == 1

    def test_append_sessions(self, tmp_path):
        append_sessions_tsv("P01", "S01", sensor_count=4, data_root=tmp_path)
        path = tmp_path / "sessions.tsv"
        assert path.exists()
        with open(path) as f:
            content = f.read()
        assert "sub-P01" in content
        assert "ses-S01" in content

        # Append same session again (should not duplicate)
        append_sessions_tsv("P01", "S01", sensor_count=4, data_root=tmp_path)
        with open(path) as f:
            lines = f.readlines()
        assert len([l for l in lines if "ses-S01" in l]) == 1

    def test_append_sessions_default_frequency_is_100hz(self, tmp_path):
        """Default sampling_frequency_hz must be 100.0 (Arduino firmware rate)."""
        append_sessions_tsv("P01", "S01", data_root=tmp_path)
        path = tmp_path / "sessions.tsv"
        with open(path) as f:
            content = f.read()
        # The last field in the data row is sampling_frequency_hz
        assert "100.0" in content


class TestDatasetMetadata:
    def test_write_dataset_description(self, tmp_path):
        write_dataset_description(data_root=tmp_path)
        path = tmp_path / "dataset_description.json"
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["Name"] == "JoTouch FMG Hand Gesture Dataset"
        assert data["BIDSVersion"] == "1.9.0"
        assert data["DatasetType"] == "raw"

    def test_update_mvc_calibration(self, tmp_path):
        # Write a minimal MVC physio CSV and metadata, then update calibration.
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sdir.mkdir(parents=True, exist_ok=True)
        physio_csv = sdir / "sub-P01_ses-S01_task-mvc_run-00_physio.csv"
        with open(physio_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["t_monotonic_ns", "sample_idx", "phase", "participant_id", "session_id", "task", "run", "fsr0", "fsr1", "fsr2", "fsr3", "cue_event", "led_fsr"])
            writer.writerow([0, 0, "ACTIVE", "P01", "S01", "mvc", 0, 100, 110, 120, 130, "", 0])
            writer.writerow([10_000_000, 1, "ACTIVE", "P01", "S01", "mvc", 0, 200, 220, 240, 260, "", 1])
            writer.writerow([20_000_000, 2, "ACTIVE", "P01", "S01", "mvc", 0, 300, 330, 360, 390, "", 0])

        write_session_metadata("P01", "S01", data_root=tmp_path)
        update_mvc_calibration("P01", "S01", data_root=tmp_path)

        physio_json = sdir / "sub-P01_ses-S01_physio.json"
        with open(physio_json) as f:
            data = json.load(f)
        assert "BaselineADCVector" in data
        assert "MVCADCVector" in data
        assert data["MVCADCVector"] == [300, 330, 360, 390]


class TestBIDSValidation:
    def test_write_physio_warns_on_out_of_range_fsr(self, tmp_path, caplog):
        writer = BIDSRunWriter(
            sub="P01", ses="S01", task="powerGrip", run=1,
            n_sensors=4, data_root=tmp_path,
        )
        writer.open()
        with caplog.at_level("WARNING"):
            writer.write_physio(0, 0, [100, 200, 2000, 300], phase="ACTIVE")
        writer.close()
        assert "FSR value out of range" in caplog.text

    def test_write_camera_warns_on_out_of_range_x(self, tmp_path, caplog):
        """x > 1.0 is out of the normalised [0,1] range and should warn."""
        writer = BIDSRunWriter(
            sub="P01", ses="S01", task="powerGrip", run=1,
            n_sensors=4, data_root=tmp_path,
        )
        writer.open()
        bad_landmarks = [0.5] * 63
        bad_landmarks[0] = 2.0  # lm00_x > 1.0
        with caplog.at_level("WARNING"):
            writer.write_camera(0, bad_landmarks, valid=True, confidence=0.9)
        writer.close()
        assert "Landmark coordinate out of range" in caplog.text

    def test_write_camera_warns_on_out_of_range_y(self, tmp_path, caplog):
        """y < 0.0 is out of the normalised [0,1] range and should warn."""
        writer = BIDSRunWriter(
            sub="P01", ses="S01", task="powerGrip", run=1,
            n_sensors=4, data_root=tmp_path,
        )
        writer.open()
        bad_landmarks = [0.5] * 63
        bad_landmarks[1] = -0.1  # lm00_y < 0.0
        with caplog.at_level("WARNING"):
            writer.write_camera(0, bad_landmarks, valid=True, confidence=0.9)
        writer.close()
        assert "Landmark coordinate out of range" in caplog.text

    def test_write_camera_no_warn_for_large_z(self, tmp_path, caplog):
        """z is unconstrained depth — large or negative z must NOT warn."""
        writer = BIDSRunWriter(
            sub="P01", ses="S01", task="powerGrip", run=1,
            n_sensors=4, data_root=tmp_path,
        )
        writer.open()
        # All x/y in [0,1], but z values well outside [-1,1]
        landmarks = []
        for _ in range(21):
            landmarks += [0.5, 0.5, -5.0]  # z = -5.0 is valid relative depth
        with caplog.at_level("WARNING"):
            writer.write_camera(0, landmarks, valid=True, confidence=0.9)
        writer.close()
        assert "Landmark coordinate out of range" not in caplog.text

    def test_write_targets_warns_on_out_of_range_angle(self, tmp_path, caplog):
        writer = BIDSRunWriter(
            sub="P01", ses="S01", task="powerGrip", run=1,
            n_sensors=4, data_root=tmp_path,
        )
        writer.open()
        with caplog.at_level("WARNING"):
            writer.write_targets(0, [10.0, 200.0, -5.0] + [0.0] * 12)
        writer.close()
        assert "Joint angle out of range" in caplog.text


# ── Phase 2: safe open, manifest, quality_flag, double-close ──────────────────


class TestSafeOpen:
    def test_explicit_run_overwrites_silently_is_rejected(self, tmp_path):
        """Re-running an explicit run number must NOT silently destroy data."""
        # First write
        w1 = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                           n_sensors=4, data_root=tmp_path)
        w1.open(); w1.write_physio(1, 0, [1, 2, 3, 4]); w1.close()
        assert w1.physio_path.exists()
        # Second open with the SAME explicit run must fail (mode "x")
        w2 = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                           n_sensors=4, data_root=tmp_path)
        with pytest.raises(FileExistsError):
            w2.open()
        # Original data preserved
        with open(w1.physio_path) as f:
            assert sum(1 for _ in f) >= 2  # header + at least 1 row

    def test_auto_run_number_picks_next_free(self, tmp_path):
        """run=None auto-increments to the next free run number."""
        w1 = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=None,
                           n_sensors=4, data_root=tmp_path)
        w1.open(); w1.close()
        assert w1.run == 0  # first run for this task
        w2 = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=None,
                           n_sensors=4, data_root=tmp_path)
        w2.open(); w2.close()
        assert w2.run == 1  # second run auto-increments


class TestManifest:
    def test_close_writes_manifest(self, tmp_path):
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [100, 200, 300, 400])
        w.write_camera(1, [0.1] * 63, valid=True, confidence=0.9)
        w.write_targets(1, [10.0] * 15)
        w.close()
        mpath = manifest_path(w.sub, w.ses, w.task, w.run, data_root=tmp_path)
        assert mpath.exists()
        with open(mpath) as f:
            m = json.load(f)
        for key in MANIFEST_REQUIRED_KEYS:
            assert key in m, f"manifest missing key: {key}"
        assert m["physio_rows"] == 1
        assert m["camera_rows"] == 1
        assert m["targets_rows"] == 1
        assert m["complete"] is True
        assert m["bad_physio_count"] == 0

    def test_complete_sentinel_written(self, tmp_path):
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open(); w.close()
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sentinels = list(sdir.glob("sub-P01_ses-S01_task-powerGrip_run-01.complete"))
        assert len(sentinels) == 1

    def test_double_close_safe(self, tmp_path):
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [1, 2, 3, 4])
        r1 = w.close()
        r2 = w.close()  # must not raise, must not corrupt manifest
        assert r1 == r2
        # Manifest still valid
        mpath = manifest_path(w.sub, w.ses, w.task, w.run, data_root=tmp_path)
        with open(mpath) as f:
            m = json.load(f)
        assert m["physio_rows"] == 1

    def test_manifest_not_written_on_crash(self, tmp_path):
        """If close() is never called (crash), no manifest/sentinel exists."""
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [1, 2, 3, 4])
        # Simulate crash: do NOT call close()
        mpath = manifest_path(w.sub, w.ses, w.task, w.run, data_root=tmp_path)
        assert not mpath.exists()
        sdir = tmp_path / "sub-P01" / "ses-S01"
        assert not list(sdir.glob("*.complete"))


class TestCameraQuality:
    """WS-5: manifest.json must include camera_quality with achieved_fps,
    mean_led_cam, led_visible_pct, total_frames, valid_pct."""

    def test_camera_quality_in_result_dict(self, tmp_path):
        """close() result must include camera_quality with all 5 fields."""
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        # Write 3 camera frames at 30Hz intervals with LED on/off
        w.write_camera(1_000_000_000, [0.1] * 63, valid=True, confidence=0.9, led_cam=200)
        w.write_camera(1_033_333_333, [0.1] * 63, valid=True, confidence=0.9, led_cam=0)
        w.write_camera(1_066_666_666, [0.1] * 63, valid=False, confidence=0.0, led_cam=200)
        result = w.close()
        assert "camera_quality" in result, "close() result must include camera_quality"
        cq = result["camera_quality"]
        assert "achieved_fps" in cq
        assert "mean_led_cam" in cq
        assert "led_visible_pct" in cq
        assert "total_frames" in cq
        assert "valid_pct" in cq

    def test_camera_quality_in_manifest(self, tmp_path):
        """manifest.json must include camera_quality with all 5 fields."""
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_camera(1_000_000_000, [0.1] * 63, valid=True, confidence=0.9, led_cam=200)
        w.write_camera(1_033_333_333, [0.1] * 63, valid=True, confidence=0.9, led_cam=0)
        w.close()
        mpath = manifest_path(w.sub, w.ses, w.task, w.run, data_root=tmp_path)
        with open(mpath) as f:
            m = json.load(f)
        assert "camera_quality" in m, "manifest must include camera_quality"
        cq = m["camera_quality"]
        assert "achieved_fps" in cq
        assert "mean_led_cam" in cq
        assert "led_visible_pct" in cq
        assert "total_frames" in cq
        assert "valid_pct" in cq

    def test_camera_quality_values_correct(self, tmp_path):
        """camera_quality values must be computed correctly from the data."""
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        # 3 frames: 2 valid, 1 invalid; LED: 200, 0, 200
        w.write_camera(1_000_000_000, [0.1] * 63, valid=True, confidence=0.9, led_cam=200)
        w.write_camera(1_033_333_333, [0.1] * 63, valid=True, confidence=0.9, led_cam=0)
        w.write_camera(1_066_666_666, [0.1] * 63, valid=False, confidence=0.0, led_cam=200)
        result = w.close()
        cq = result["camera_quality"]
        assert cq["total_frames"] == 3
        # valid_pct = 2/3 = 66.67%
        assert abs(cq["valid_pct"] - 66.67) < 0.1 or abs(cq["valid_pct"] - (2/3*100)) < 0.1
        # mean_led_cam = (200 + 0 + 200) / 3 = 133.33
        assert abs(cq["mean_led_cam"] - 133.33) < 0.1
        # led_visible_pct = 2/3 = 66.67% (threshold=10, so 200>10 and 0<10)
        assert abs(cq["led_visible_pct"] - 66.67) < 0.1 or abs(cq["led_visible_pct"] - (2/3*100)) < 0.1
        # achieved_fps: 2 intervals over 66.67ms = 30fps
        assert cq["achieved_fps"] > 0


class TestQualityFlag:
    def test_physio_has_quality_flag_column(self, tmp_path):
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [100, 200, 300, 400])
        w.close()
        with open(w.physio_path) as f:
            header = next(csv.reader(f))
        assert PHYSIO_QUALITY_FLAG in header

    def test_physio_quality_flag_set_on_bad_value(self, tmp_path):
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [100, 2000, 300, 400])  # 2000 > 1023
        w.close()
        with open(w.physio_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row[PHYSIO_QUALITY_FLAG] == "1"
        mpath = manifest_path(w.sub, w.ses, w.task, w.run, data_root=tmp_path)
        with open(mpath) as f:
            m = json.load(f)
        assert m["bad_physio_count"] == 1

    def test_physio_quality_flag_zero_on_good_value(self, tmp_path):
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [100, 200, 300, 400])
        w.close()
        with open(w.physio_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row[PHYSIO_QUALITY_FLAG] == "0"

    def test_camera_has_quality_flag_column(self, tmp_path):
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_camera(1, [0.1] * 63, valid=True, confidence=0.9)
        w.close()
        with open(w.camera_path) as f:
            header = next(csv.reader(f))
        assert CAMERA_QUALITY_FLAG in header

    def test_camera_quality_flag_set_on_bad_landmark(self, tmp_path):
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        bad = [0.5] * 63; bad[0] = 2.0  # x > 1
        w.write_camera(1, bad, valid=True, confidence=0.9)
        w.close()
        with open(w.camera_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row[CAMERA_QUALITY_FLAG] == "1"

    def test_targets_has_quality_flag_column(self, tmp_path):
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_targets(1, [10.0] * 15)
        w.close()
        with open(w.targets_path) as f:
            header = next(csv.reader(f))
        assert TARGETS_QUALITY_FLAG in header

    def test_targets_quality_flag_set_on_bad_angle(self, tmp_path):
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_targets(1, [10.0, 200.0] + [0.0] * 13)  # 200 > 180
        w.close()
        with open(w.targets_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row[TARGETS_QUALITY_FLAG] == "1"


class TestPhysioPadding:
    def test_under_length_pads_with_zero(self, tmp_path):
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [100, 200])  # only 2 of 4 sensors
        w.close()
        with open(w.physio_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["fsr0"] == "100"
        assert row["fsr1"] == "200"
        assert row["fsr2"] == "0"
        assert row["fsr3"] == "0"

    def test_eight_sensors(self, tmp_path):
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=8, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, list(range(100, 108)))
        w.close()
        with open(w.physio_path) as f:
            header = next(csv.reader(f))
        assert "fsr7" in header
        assert "fsr8" not in header


class TestNonMonotonicWarning:
    def test_physio_warns_on_backwards_timestamp(self, tmp_path, caplog):
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(2000, 0, [1, 2, 3, 4])
        with caplog.at_level("WARNING"):
            w.write_physio(1000, 1, [5, 6, 7, 8])  # backwards
        w.close()
        assert "Non-monotonic physio timestamp" in caplog.text


class TestTargetsDictMissingKeys:
    def test_missing_keys_default_to_zero(self, tmp_path):
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_targets(1, {"target_thumb_cmc_flex": 30.0})  # 14 missing
        w.close()
        with open(w.targets_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["target_thumb_cmc_flex"] == "30.0"
        assert row["target_pinky_dip_flex"] == "0.0"  # missing → default


class TestMVCUsesRestPhase:
    def test_mvc_uses_rest_phase_as_baseline(self, tmp_path):
        """update_mvc_calibration must use REST-phase rows as baseline,
        not the 'first 1s of ACTIVE' hack."""
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sdir.mkdir(parents=True)
        physio_csv = sdir / "sub-P01_ses-S01_task-mvc_run-00_physio.csv"
        # REST rows: low values (baseline). RECORD rows: high values (MVC).
        with open(physio_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["t_monotonic_ns","sample_idx","phase","participant_id",
                         "session_id","task","run","fsr0","fsr1","fsr2","fsr3",
                         "cue_event","led_fsr","quality_flag"])
            # 10 REST samples (baseline ~100)
            for i in range(10):
                wr.writerow([i*10_000_000, i, "REST", "P01","S01","mvc",0,
                             100,110,120,130,"","",0])
            # 10 RECORD samples (MVC ~800)
            for i in range(10, 20):
                wr.writerow([i*10_000_000, i, "RECORD", "P01","S01","mvc",0,
                             800,810,820,830,"","",0])
        # Write physio.json so update_mvc_calibration has somewhere to write
        write_session_metadata("P01", "S01", sensor_count=4, data_root=tmp_path)
        update_mvc_calibration("P01", "S01", data_root=tmp_path)
        with open(sdir / "sub-P01_ses-S01_physio.json") as f:
            meta = json.load(f)
        # Baseline must come from REST rows (~100), MVC from RECORD rows (~800)
        assert meta["BaselineADCVector"] == [100, 110, 120, 130]
        assert meta["MVCADCVector"] == [800, 810, 820, 830]


class TestVersionConstantsUsed:
    def test_dataset_description_uses_schema_bids_version(self, tmp_path):
        write_dataset_description(data_root=tmp_path)
        with open(tmp_path / "dataset_description.json") as f:
            d = json.load(f)
        assert d["BIDSVersion"] == schema.BIDS_VERSION

    def test_session_metadata_uses_schema_software_version(self, tmp_path):
        write_session_metadata("P01", "S01", data_root=tmp_path)
        with open(tmp_path / "sub-P01" / "ses-S01" / "sub-P01_ses-S01_physio.json") as f:
            m = json.load(f)
        assert m["SoftwareVersion"] == schema.SOFTWARE_VERSION


# ── Phase 2b: aborted flag, quarantine, run-number reuse ──────────────────────


class TestAbortedFlag:
    def test_close_aborted_writes_complete_false(self, tmp_path):
        """close(aborted=True) writes manifest with complete=false, aborted=true."""
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [100, 200, 300, 400])
        w.close(aborted=True)
        mpath = manifest_path(w.sub, w.ses, w.task, w.run, data_root=tmp_path)
        with open(mpath) as f:
            m = json.load(f)
        assert m["complete"] is False
        assert m["aborted"] is True

    def test_close_normal_writes_complete_true(self, tmp_path):
        """close() with no aborted flag writes complete=true, aborted=false."""
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [100, 200, 300, 400])
        w.close()
        mpath = manifest_path(w.sub, w.ses, w.task, w.run, data_root=tmp_path)
        with open(mpath) as f:
            m = json.load(f)
        assert m["complete"] is True
        assert m["aborted"] is False

    def test_aborted_run_has_no_complete_sentinel(self, tmp_path):
        """Aborted runs must NOT get a .complete sentinel."""
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [100, 200, 300, 400])
        w.close(aborted=True)
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sentinels = list(sdir.glob("*.complete"))
        assert len(sentinels) == 0


class TestQuarantine:
    def test_quarantine_moves_csvs_to_partial_dir(self, tmp_path):
        """quarantine_partial_run moves the 3 CSVs + manifest to _partial/."""
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [100, 200, 300, 400])
        w.write_camera(1, [0.1] * 63, valid=True, confidence=0.9)
        w.write_targets(1, [10.0] * 15)
        w.close(aborted=True)

        # Before quarantine: files are in the session dir
        assert w.physio_path.exists()
        quarantine_partial_run(w.sub, w.ses, w.task, w.run, data_root=tmp_path)

        # After quarantine: CSVs moved to _partial/
        assert not w.physio_path.exists()
        assert not w.camera_path.exists()
        assert not w.targets_path.exists()
        partial_dir = tmp_path / "sub-P01" / "ses-S01" / "_partial"
        assert partial_dir.exists()
        # At least one CSV should be in the attempt subfolder
        attempt_dirs = list(partial_dir.iterdir())
        assert len(attempt_dirs) == 1
        assert "attempt1" in attempt_dirs[0].name
        csvs = list(attempt_dirs[0].glob("*.csv"))
        assert len(csvs) == 3

    def test_quarantine_frees_run_number(self, tmp_path):
        """After quarantine, the run number can be reused (mode 'x' succeeds)."""
        w1 = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                           n_sensors=4, data_root=tmp_path)
        w1.open()
        w1.write_physio(1, 0, [100, 200, 300, 400])
        w1.close(aborted=True)
        quarantine_partial_run(w1.sub, w1.ses, w1.task, w1.run, data_root=tmp_path)

        # Now run-01 should be reusable
        w2 = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                           n_sensors=4, data_root=tmp_path)
        w2.open()  # should NOT raise FileExistsError
        w2.write_physio(1, 0, [500, 600, 700, 800])
        w2.close()
        assert w2.physio_path.exists()

    def test_quarantine_double_fail_uses_attempt2(self, tmp_path):
        """If the same run fails twice, second quarantine uses attempt2/."""
        # First failed attempt
        w1 = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                           n_sensors=4, data_root=tmp_path)
        w1.open()
        w1.write_physio(1, 0, [100, 200, 300, 400])
        w1.close(aborted=True)
        quarantine_partial_run(w1.sub, w1.ses, w1.task, w1.run, data_root=tmp_path)

        # Second failed attempt (same run number, now freed)
        w2 = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                           n_sensors=4, data_root=tmp_path)
        w2.open()
        w2.write_physio(1, 0, [500, 600, 700, 800])
        w2.close(aborted=True)
        quarantine_partial_run(w2.sub, w2.ses, w2.task, w2.run, data_root=tmp_path)

        partial_dir = tmp_path / "sub-P01" / "ses-S01" / "_partial"
        attempt_dirs = sorted(partial_dir.iterdir())
        assert len(attempt_dirs) == 2
        assert "attempt1" in attempt_dirs[0].name
        assert "attempt2" in attempt_dirs[1].name

    def test_quarantine_preserves_manifest(self, tmp_path):
        """The manifest (with complete=false) is moved to _partial/ too."""
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [100, 200, 300, 400])
        w.close(aborted=True)
        mpath = manifest_path(w.sub, w.ses, w.task, w.run, data_root=tmp_path)
        assert mpath.exists()
        quarantine_partial_run(w.sub, w.ses, w.task, w.run, data_root=tmp_path)
        # Manifest moved to _partial/
        assert not mpath.exists()
        partial_dir = tmp_path / "sub-P01" / "ses-S01" / "_partial"
        manifests = list(partial_dir.rglob("*manifest*"))
        assert len(manifests) == 1

    def test_quarantine_no_manifest_crash_recovery(self, tmp_path):
        """Quarantine works even if no manifest exists (crash scenario)."""
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [100, 200, 300, 400])
        # Simulate crash: do NOT call close() — no manifest written
        # Force-close the file handles so quarantine can move them
        w._physio_file.close()
        w._camera_file.close()
        w._targets_file.close()
        w._physio_file = None
        w._camera_file = None
        w._targets_file = None

        # Should not raise even though no manifest exists
        quarantine_partial_run(w.sub, w.ses, w.task, w.run, data_root=tmp_path)
        assert not w.physio_path.exists()
        partial_dir = tmp_path / "sub-P01" / "ses-S01" / "_partial"
        csvs = list(partial_dir.rglob("*.csv"))
        assert len(csvs) == 3

    def test_quarantine_does_not_touch_completed_runs(self, tmp_path):
        """Quarantine of run-01 must not affect run-02 (completed)."""
        # run-01: completed
        w1 = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                           n_sensors=4, data_root=tmp_path)
        w1.open(); w1.write_physio(1, 0, [100, 200, 300, 400]); w1.close()
        # run-02: aborted
        w2 = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=2,
                           n_sensors=4, data_root=tmp_path)
        w2.open(); w2.write_physio(1, 0, [500, 600, 700, 800]); w2.close(aborted=True)
        quarantine_partial_run(w2.sub, w2.ses, w2.task, w2.run, data_root=tmp_path)

        # run-01 still intact
        assert w1.physio_path.exists()
        mpath1 = manifest_path(w1.sub, w1.ses, w1.task, w1.run, data_root=tmp_path)
        assert mpath1.exists()
        with open(mpath1) as f:
            m = json.load(f)
        assert m["complete"] is True


class TestSweepOrphanPartials:
    def test_sweep_quarantines_orphan_no_manifest(self, tmp_path):
        """An orphan run (CSVs but no manifest) is quarantined by sweep."""
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [100, 200, 300, 400])
        # Simulate crash: close file handles without calling close() — no manifest
        w._physio_file.close()
        w._camera_file.close()
        w._targets_file.close()
        w._physio_file = None
        w._camera_file = None
        w._targets_file = None

        assert w.physio_path.exists()
        swept = sweep_orphan_partials("P01", "S01", data_root=tmp_path)
        assert swept >= 1
        assert not w.physio_path.exists()
        partial_dir = tmp_path / "sub-P01" / "ses-S01" / "_partial"
        csvs = list(partial_dir.rglob("*.csv"))
        assert len(csvs) == 3

    def test_sweep_quarantines_partial_manifest(self, tmp_path):
        """A run with complete=false manifest is quarantined by sweep."""
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [100, 200, 300, 400])
        w.close(aborted=True)

        swept = sweep_orphan_partials("P01", "S01", data_root=tmp_path)
        assert swept >= 1
        assert not w.physio_path.exists()

    def test_sweep_preserves_completed_runs(self, tmp_path):
        """Completed runs (complete=true manifest + sentinel) are NOT swept."""
        w = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [100, 200, 300, 400])
        w.close()

        swept = sweep_orphan_partials("P01", "S01", data_root=tmp_path)
        assert swept == 0
        assert w.physio_path.exists()

    def test_sweep_mixed_completed_and_orphan(self, tmp_path):
        """Sweep only quarantines orphans, leaving completed runs intact."""
        w1 = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=1,
                           n_sensors=4, data_root=tmp_path)
        w1.open(); w1.write_physio(1, 0, [100, 200, 300, 400]); w1.close()
        w2 = BIDSRunWriter(sub="P01", ses="S01", task="powerGrip", run=2,
                           n_sensors=4, data_root=tmp_path)
        w2.open(); w2.write_physio(1, 0, [500, 600, 700, 800])
        w2._physio_file.close(); w2._camera_file.close(); w2._targets_file.close()
        w2._physio_file = None; w2._camera_file = None; w2._targets_file = None

        swept = sweep_orphan_partials("P01", "S01", data_root=tmp_path)
        assert swept == 1
        assert w1.physio_path.exists()
        assert not w2.physio_path.exists()

    def test_sweep_empty_session_returns_zero(self, tmp_path):
        """Sweep on a session with no run files returns 0."""
        swept = sweep_orphan_partials("P01", "S01", data_root=tmp_path)
        assert swept == 0

    def test_sweep_nonexistent_session_returns_zero(self, tmp_path):
        """Sweep on a non-existent session dir returns 0 without error."""
        swept = sweep_orphan_partials("P99", "S99", data_root=tmp_path)
        assert swept == 0


class TestSweepIncompleteSessions:
    """Tests for sweep_incomplete_sessions — quarantine sessions that haven't
    completed their CHOSEN protocol (respecting n_reps + include_freeform)."""

    @staticmethod
    def _write_full_protocol(sub: str, ses: str, data_root: Path) -> None:
        """Write a complete 76-run protocol session (all tasks, all reps)."""
        from apps.collection.protocol import build_protocol
        protocol = build_protocol()
        for spec in protocol:
            w = BIDSRunWriter(
                sub=sub, ses=ses, task=spec.task, run=spec.run,
                n_sensors=4, data_root=data_root,
            )
            w.open()
            w.write_physio(1, 0, [100, 200, 300, 400])
            w.close()

    def test_quarantines_session_with_zero_runs(self, tmp_path):
        """A session dir with metadata but no runs is quarantined."""
        write_session_metadata("P01", "S01", sensor_count=4, data_root=tmp_path)
        sdir = tmp_path / "sub-P01" / "ses-S01"
        assert sdir.exists()

        swept = sweep_incomplete_sessions(data_root=tmp_path)
        assert swept == 1
        assert not sdir.exists()
        dest = tmp_path / "_incomplete" / "sub-P01" / "ses-S01"
        assert dest.exists()
        assert any(dest.iterdir())

    def test_preserves_session_with_full_protocol(self, tmp_path):
        """A session with ALL 76 protocol runs completed is NOT quarantined."""
        self._write_full_protocol("P01", "S01", tmp_path)

        swept = sweep_incomplete_sessions(data_root=tmp_path)
        assert swept == 0
        # Session dir still exists with all runs
        sdir = tmp_path / "sub-P01" / "ses-S01"
        assert sdir.exists()
        manifests = list(sdir.glob("*_manifest.json"))
        assert len(manifests) == 76

    def test_quarantines_session_with_partial_protocol(self, tmp_path):
        """A session with only 1/76 runs (just mvc) IS quarantined."""
        w = BIDSRunWriter(sub="P01", ses="S01", task="mvc", run=0,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [100, 200, 300, 400])
        w.close()  # completed, but only 1 of 76
        # Write physio.json with default config (76-run protocol) so the
        # sweep knows the expected protocol and can detect incompleteness.
        write_session_metadata("P01", "S01", sensor_count=4, data_root=tmp_path)

        swept = sweep_incomplete_sessions(data_root=tmp_path)
        assert swept == 1
        assert not w.physio_path.exists()

    def test_quarantines_session_missing_one_task(self, tmp_path):
        """A session with 73/76 runs (missing freeform) IS quarantined."""
        from apps.collection.protocol import build_protocol
        from core.schema import FREEFORM_TASKS
        protocol = build_protocol()
        for spec in protocol:
            if spec.task in FREEFORM_TASKS:
                continue  # skip freeform — session will be 73/76
            w = BIDSRunWriter(
                sub="P01", ses="S01", task=spec.task, run=spec.run,
                n_sensors=4, data_root=tmp_path,
            )
            w.open()
            w.write_physio(1, 0, [100, 200, 300, 400])
            w.close()
        # Write physio.json with default config (include_freeform=True → 76)
        write_session_metadata("P01", "S01", sensor_count=4, data_root=tmp_path)

        swept = sweep_incomplete_sessions(data_root=tmp_path)
        assert swept == 1  # missing freeform → incomplete

    def test_quarantines_session_with_only_aborted_run(self, tmp_path):
        """A session with only an aborted run (complete=false) is quarantined."""
        w = BIDSRunWriter(sub="P01", ses="S01", task="mvc", run=0,
                          n_sensors=4, data_root=tmp_path)
        w.open()
        w.write_physio(1, 0, [100, 200, 300, 400])
        w.close(aborted=True)

        swept = sweep_incomplete_sessions(data_root=tmp_path)
        assert swept == 1
        assert not w.physio_path.exists()

    def test_mixed_complete_and_incomplete_subjects(self, tmp_path):
        """P01 has full protocol, P02 has only metadata → only P02 quarantined."""
        self._write_full_protocol("P01", "S01", tmp_path)
        write_session_metadata("P02", "S01", sensor_count=4, data_root=tmp_path)

        swept = sweep_incomplete_sessions(data_root=tmp_path)
        assert swept == 1
        # P01 preserved
        assert (tmp_path / "sub-P01" / "ses-S01").exists()
        # P02 quarantined
        assert not (tmp_path / "sub-P02" / "ses-S01").exists()
        assert (tmp_path / "_incomplete" / "sub-P02" / "ses-S01").exists()

    def test_removes_from_participants_tsv(self, tmp_path):
        """Quarantined subjects are removed from participants.tsv."""
        append_participants_tsv("P01", data_root=tmp_path)
        append_participants_tsv("P02", data_root=tmp_path)
        write_session_metadata("P02", "S01", sensor_count=4, data_root=tmp_path)

        swept = sweep_incomplete_sessions(data_root=tmp_path)
        assert swept == 1

        tsv = (tmp_path / "participants.tsv").read_text()
        assert "sub-P01" in tsv
        assert "sub-P02" not in tsv

    def test_removes_from_sessions_tsv(self, tmp_path):
        """Quarantined sessions are removed from sessions.tsv."""
        append_sessions_tsv("P01", "S01", data_root=tmp_path)
        append_sessions_tsv("P02", "S01", data_root=tmp_path)
        write_session_metadata("P02", "S01", sensor_count=4, data_root=tmp_path)

        swept = sweep_incomplete_sessions(data_root=tmp_path)
        assert swept == 1

        tsv = (tmp_path / "sessions.tsv").read_text()
        assert "sub-P01" in tsv
        assert "sub-P02" not in tsv

    def test_frees_subject_label_for_reuse(self, tmp_path):
        """After quarantine, next_subject_label returns the freed label."""
        self._write_full_protocol("P01", "S01", tmp_path)
        write_session_metadata("P02", "S01", sensor_count=4, data_root=tmp_path)

        assert naming.next_subject_label(data_root=tmp_path) == "P03"

        swept = sweep_incomplete_sessions(data_root=tmp_path)
        assert swept == 1

        assert naming.next_subject_label(data_root=tmp_path) == "P02"

    def test_empty_data_root_returns_zero(self, tmp_path):
        """Sweep on an empty data root returns 0."""
        swept = sweep_incomplete_sessions(data_root=tmp_path)
        assert swept == 0

    def test_preserves_partial_dir_in_quarantine(self, tmp_path):
        """The _partial/ subdirectory is moved along with the session."""
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sdir.mkdir(parents=True)
        (sdir / "sub-P01_ses-S01_channels.tsv").write_text("test")
        partial = sdir / "_partial" / "task-mvc_run-07_attempt1"
        partial.mkdir(parents=True)
        (partial / "sub-P01_ses-S01_task-mvc_run-07_camera.csv").write_text("data")

        swept = sweep_incomplete_sessions(data_root=tmp_path)
        assert swept == 1
        dest_partial = tmp_path / "_incomplete" / "sub-P01" / "ses-S01" / "_partial"
        assert dest_partial.exists()
        assert (dest_partial / "task-mvc_run-07_attempt1" / "sub-P01_ses-S01_task-mvc_run-07_camera.csv").exists()

    def test_preserves_session_complete_without_freeform(self, tmp_path):
        """A session with include_freeform=False that completed all 25
        non-freeform runs STAYS in data/raw/.

        The completeness check must respect the session's chosen protocol
        (n_reps + include_freeform), not the hardcoded 76-run default.
        """
        from apps.collection.protocol import build_protocol
        from core.schema import FREEFORM_TASKS
        protocol = build_protocol(n_reps=1, include_freeform=False)
        assert len(protocol) == 25  # mvc + 15 single_dof + 9 multi_dof
        for spec in protocol:
            w = BIDSRunWriter(
                sub="P01", ses="S01", task=spec.task, run=spec.run,
                n_sensors=4, data_root=tmp_path,
            )
            w.open()
            w.write_physio(1, 0, [100, 200, 300, 400])
            w.close()
        # Persist the chosen protocol config so the sweep can read it
        write_session_metadata("P01", "S01", sensor_count=4, data_root=tmp_path,
                               n_reps=1, include_freeform=False)

        swept = sweep_incomplete_sessions(data_root=tmp_path)
        assert swept == 0, "Session completed its chosen (no-freeform) protocol"
        assert (tmp_path / "sub-P01" / "ses-S01").exists()
        assert not (tmp_path / "_incomplete").exists()

    def test_preserves_session_complete_with_n_reps_1(self, tmp_path):
        """A session with n_reps=1, include_freeform=True that completed
        all 26 runs (mvc + 15 + 9 + 1 freeform) STAYS in data/raw/."""
        from apps.collection.protocol import build_protocol
        protocol = build_protocol(n_reps=1, include_freeform=True)
        assert len(protocol) == 26
        for spec in protocol:
            w = BIDSRunWriter(
                sub="P01", ses="S01", task=spec.task, run=spec.run,
                n_sensors=4, data_root=tmp_path,
            )
            w.open()
            w.write_physio(1, 0, [100, 200, 300, 400])
            w.close()
        write_session_metadata("P01", "S01", sensor_count=4, data_root=tmp_path,
                               n_reps=1, include_freeform=True)

        swept = sweep_incomplete_sessions(data_root=tmp_path)
        assert swept == 0, "Session completed its chosen (n_reps=1) protocol"
        assert (tmp_path / "sub-P01" / "ses-S01").exists()

    def test_preserves_session_when_manifest_complete_but_sentinel_missing(self, tmp_path):
        """A run whose manifest says complete=true but whose .complete sentinel
        is missing (e.g. crash between manifest write and sentinel touch) must
        STILL count as completed.  The manifest is the authoritative source;
        the sentinel is a convenience for filesystem scanning.
        """
        from apps.collection.protocol import build_protocol
        from core.schema import FREEFORM_TASKS
        protocol = build_protocol(n_reps=1, include_freeform=False)
        for spec in protocol:
            w = BIDSRunWriter(
                sub="P01", ses="S01", task=spec.task, run=spec.run,
                n_sensors=4, data_root=tmp_path,
            )
            w.open()
            w.write_physio(1, 0, [100, 200, 300, 400])
            w.close()
        write_session_metadata("P01", "S01", sensor_count=4, data_root=tmp_path,
                               n_reps=1, include_freeform=False)
        # Delete one sentinel to simulate a crash between manifest write and
        # sentinel touch.
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sentinels = sorted(sdir.glob("*.complete"))
        assert sentinels, "Expected at least one .complete sentinel"
        sentinels[0].unlink()

        swept = sweep_incomplete_sessions(data_root=tmp_path)
        assert swept == 0, (
            "Session completed its chosen protocol — missing one sentinel "
            "must not cause quarantine (manifest is authoritative)"
        )
        assert (tmp_path / "sub-P01" / "ses-S01").exists()

    def test_preserves_session_when_physio_json_missing(self, tmp_path):
        """If physio.json is missing or corrupted (can't read the chosen
        protocol config), a session with completed runs must NOT be
        quarantined.  Falling back to the 76-run default would wrongly
        quarantine a session that completed its chosen (smaller) protocol.
        Data preservation takes priority over protocol-completeness when
        the protocol config is unknown.
        """
        from apps.collection.protocol import build_protocol
        from core.schema import FREEFORM_TASKS
        protocol = build_protocol(n_reps=1, include_freeform=False)
        for spec in protocol:
            w = BIDSRunWriter(
                sub="P01", ses="S01", task=spec.task, run=spec.run,
                n_sensors=4, data_root=tmp_path,
            )
            w.open()
            w.write_physio(1, 0, [100, 200, 300, 400])
            w.close()
        # Write physio.json, then corrupt it so the config can't be read.
        write_session_metadata("P01", "S01", sensor_count=4, data_root=tmp_path,
                               n_reps=1, include_freeform=False)
        sdir = tmp_path / "sub-P01" / "ses-S01"
        physio = sorted(sdir.glob("*_physio.json"))[0]
        physio.write_text("{CORRUPTED JSON")

        swept = sweep_incomplete_sessions(data_root=tmp_path)
        assert swept == 0, (
            "Session has completed runs but physio.json is corrupted — "
            "must NOT quarantine (data preservation > protocol-completeness "
            "when the chosen protocol config is unknown)"
        )
        assert (tmp_path / "sub-P01" / "ses-S01").exists()

