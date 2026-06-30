"""Tests for core.schema — column contracts and task taxonomy."""

from __future__ import annotations

from core.schema import (
    PHYSIO_BASE_COLUMNS,
    LANDMARK_COLUMNS,
    TARGET_COLUMNS,
    SINGLE_DOF_TASKS,
    MULTI_DOF_TASKS,
    FREEFORM_TASKS,
    BASELINE_TASK,
    ALL_TASKS,
    task_phase,
    physio_sensor_columns,
    physio_all_columns,
    validate_physio_columns,
    validate_targets_columns,
    PHYSIO_QUALITY_FLAG,
    PHYSIO_CUE_EVENT,
    PHYSIO_LED_FSR,
    BIDS_VERSION,
    SOFTWARE_VERSION,
    MANIFEST_REQUIRED_KEYS,
)


class TestColumnContracts:
    def test_physio_sensor_columns(self):
        cols = physio_sensor_columns(4)
        assert cols == ["fsr0", "fsr1", "fsr2", "fsr3"]

    def test_physio_sensor_columns_8(self):
        cols = physio_sensor_columns(8)
        assert len(cols) == 8
        assert cols[0] == "fsr0"
        assert cols[-1] == "fsr7"

    def test_physio_all_columns(self):
        cols = physio_all_columns(4)
        assert "t_monotonic_ns" in cols
        assert "fsr0" in cols
        assert "fsr3" in cols
        assert "led_fsr" in cols

    def test_landmark_columns_count(self):
        assert len(LANDMARK_COLUMNS) == 63  # 21 landmarks × 3 coords

    def test_landmark_columns_format(self):
        assert "mp_lm00_x" in LANDMARK_COLUMNS
        assert "mp_lm20_z" in LANDMARK_COLUMNS

    def test_target_columns_count(self):
        assert len(TARGET_COLUMNS) == 15  # 15 finger DOFs

    def test_target_columns_content(self):
        assert "target_thumb_cmc_flex" in TARGET_COLUMNS
        assert "target_pinky_dip_flex" in TARGET_COLUMNS
        # No wrist targets
        assert not any("wrist" in c for c in TARGET_COLUMNS)


class TestTaskTaxonomy:
    def test_single_dof_count(self):
        assert len(SINGLE_DOF_TASKS) == 15

    def test_multi_dof_count(self):
        assert len(MULTI_DOF_TASKS) == 9

    def test_freeform_count(self):
        assert len(FREEFORM_TASKS) == 1

    def test_all_tasks_count(self):
        assert len(ALL_TASKS) == 26  # 1 baseline + 15 + 9 + 1

    def test_task_phase_baseline(self):
        assert task_phase("mvc") == "baseline"

    def test_task_phase_single_dof(self):
        assert task_phase("thumbCmcIso") == "single_dof"
        assert task_phase("pinkyDipIso") == "single_dof"

    def test_task_phase_multi_dof(self):
        assert task_phase("powerGrip") == "multi_dof"
        assert task_phase("counting") == "multi_dof"

    def test_task_phase_freeform(self):
        assert task_phase("freeform") == "freeform"

    def test_task_phase_unknown(self):
        assert task_phase("nonexistent") == "unknown"


class TestValidation:
    def test_validate_physio_missing(self):
        missing = validate_physio_columns(["t_monotonic_ns", "fsr0"], n_sensors=4)
        assert "sample_idx" in missing
        assert "fsr1" in missing

    def test_validate_physio_ok(self):
        cols = physio_all_columns(4)
        missing = validate_physio_columns(cols, n_sensors=4)
        assert missing == []

    def test_validate_targets_missing(self):
        missing = validate_targets_columns(["t_monotonic_ns"])
        assert len(missing) == 16  # 15 target columns + quality_flag


class TestQualityFlag:
    def test_quality_flag_constant(self):
        assert PHYSIO_QUALITY_FLAG == "quality_flag"

    def test_quality_flag_in_physio_columns(self):
        cols = physio_all_columns(4)
        assert PHYSIO_QUALITY_FLAG in cols

    def test_validate_physio_requires_quality_flag(self):
        # Build columns WITHOUT quality_flag → must be reported missing
        cols = (
            PHYSIO_BASE_COLUMNS[:7]
            + physio_sensor_columns(4)
            + [PHYSIO_CUE_EVENT, PHYSIO_LED_FSR]  # omit quality_flag
        )
        missing = validate_physio_columns(cols, n_sensors=4)
        assert PHYSIO_QUALITY_FLAG in missing


class TestVersionConstants:
    def test_bids_version_is_string(self):
        assert isinstance(BIDS_VERSION, str)
        assert BIDS_VERSION  # non-empty

    def test_software_version_is_string(self):
        assert isinstance(SOFTWARE_VERSION, str)
        assert SOFTWARE_VERSION

    def test_versions_match_documented_contract(self):
        # Single source of truth — must not drift between code and docs.
        # Update docs/DATA_STRUCTURE.md if these change.
        assert BIDS_VERSION == "1.9.0"


class TestManifestContract:
    def test_manifest_keys_present(self):
        # These keys are written by BIDSRunWriter.close() and read by
        # core.loader.load_run() to gate partial runs.
        required = set(MANIFEST_REQUIRED_KEYS)
        assert "physio_rows" in required
        assert "camera_rows" in required
        assert "targets_rows" in required
        assert "bad_physio_count" in required
        assert "bad_camera_count" in required
        assert "bad_targets_count" in required
        assert "started_at" in required
        assert "finished_at" in required
        assert "complete" in required
