"""Tests for core.naming — BIDS filename parsing and generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.naming import (
    BIDSRunName,
    build_filename,
    parse_filename,
    is_physio_file,
    is_camera_file,
    is_targets_file,
    list_runs_in_session,
    validate_label,
    next_run_number,
    list_subjects,
    next_subject_label,
    next_session_label,
)


class TestBuildFilename:
    def test_basic_physio(self):
        assert build_filename("P01", "S01", "thumbCmcIso", 1, "physio") == \
            "sub-P01_ses-S01_task-thumbCmcIso_run-01_physio.csv"

    def test_baseline_run00(self):
        assert build_filename("P01", "S01", "mvc", 0, "targets") == \
            "sub-P01_ses-S01_task-mvc_run-00_targets.csv"

    def test_run_padding(self):
        assert "run-09" in build_filename("P01", "S01", "powerGrip", 9, "physio")
        assert "run-10" in build_filename("P01", "S01", "powerGrip", 10, "physio")

    def test_invalid_suffix_raises(self):
        import pytest
        with pytest.raises(ValueError):
            build_filename("P01", "S01", "powerGrip", 1, "invalid")


class TestParseFilename:
    def test_parse_physio(self):
        parsed = parse_filename("sub-P01_ses-S01_task-thumbCmcIso_run-01_physio.csv")
        assert parsed is not None
        assert parsed.sub == "P01"
        assert parsed.ses == "S01"
        assert parsed.task == "thumbCmcIso"
        assert parsed.run == 1
        assert parsed.suffix == "physio"
        assert parsed.extension == "csv"

    def test_parse_baseline(self):
        parsed = parse_filename("sub-P01_ses-S01_task-mvc_run-00_targets.csv")
        assert parsed is not None
        assert parsed.run == 0
        assert parsed.is_baseline is True

    def test_parse_non_bids_returns_none(self):
        assert parse_filename("P01_grip_2026-05-16_133247.csv") is None
        assert parse_filename("random_file.txt") is None
        assert parse_filename("sub-P01_ses-S01_physio.json") is None  # metadata, not run

    def test_parse_path_object(self):
        parsed = parse_filename(Path("sub-P02_ses-S03_task-powerGrip_run-05_camera.csv"))
        assert parsed is not None
        assert parsed.sub == "P02"
        assert parsed.ses == "S03"
        assert parsed.suffix == "camera"


class TestFileChecks:
    def test_is_physio(self):
        assert is_physio_file("sub-P01_ses-S01_task-mvc_run-00_physio.csv") is True
        assert is_physio_file("sub-P01_ses-S01_task-mvc_run-00_camera.csv") is False

    def test_is_camera(self):
        assert is_camera_file("sub-P01_ses-S01_task-mvc_run-00_camera.csv") is True
        assert is_camera_file("sub-P01_ses-S01_task-mvc_run-00_physio.csv") is False

    def test_is_targets(self):
        assert is_targets_file("sub-P01_ses-S01_task-mvc_run-00_targets.csv") is True
        assert is_targets_file("sub-P01_ses-S01_task-mvc_run-00_physio.csv") is False


class TestListRuns:
    def test_empty_directory(self, tmp_path):
        runs = list_runs_in_session(tmp_path)
        assert runs == []

    def test_with_files(self, tmp_path):
        for suffix in ("physio", "camera", "targets"):
            fname = build_filename("P01", "S01", "powerGrip", 1, suffix)
            (tmp_path / fname).write_text("dummy")
        runs = list_runs_in_session(tmp_path)
        assert len(runs) == 1  # one run, 3 files → 1 entry
        assert runs[0].task == "powerGrip"
        assert runs[0].run == 1


class TestValidateLabel:
    def test_valid_camel_case(self):
        assert validate_label("powerGrip", "task") == "powerGrip"
        assert validate_label("thumbCmcIso", "task") == "thumbCmcIso"

    def test_valid_alphanumeric(self):
        assert validate_label("P01", "sub") == "P01"
        assert validate_label("S01", "ses") == "S01"

    def test_rejects_space(self):
        with pytest.raises(ValueError):
            validate_label("P 01", "sub")
        with pytest.raises(ValueError):
            validate_label("power grip", "task")

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            validate_label("", "sub")

    def test_rejects_special_chars(self):
        with pytest.raises(ValueError):
            validate_label("P-01", "sub")  # hyphen collides with BIDS separator
        with pytest.raises(ValueError):
            validate_label("power_grip", "task")  # underscore collides with BIDS separator

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError):
            validate_label("P01", "unknown")


class TestNextRunNumber:
    def test_empty_session_returns_zero(self, tmp_path):
        assert next_run_number("P01", "S01", "powerGrip", data_root=tmp_path) == 0

    def test_after_existing_run_returns_max_plus_one(self, tmp_path):
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sdir.mkdir(parents=True)
        for run in (0, 1, 2):
            fname = build_filename("P01", "S01", "powerGrip", run, "physio")
            (sdir / fname).write_text("dummy")
        assert next_run_number("P01", "S01", "powerGrip", data_root=tmp_path) == 3

    def test_only_counts_matching_task(self, tmp_path):
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sdir.mkdir(parents=True)
        # powerGrip has run-05; thumbCmcIso has run-01
        (sdir / build_filename("P01", "S01", "powerGrip", 5, "physio")).write_text("d")
        (sdir / build_filename("P01", "S01", "thumbCmcIso", 1, "physio")).write_text("d")
        assert next_run_number("P01", "S01", "powerGrip", data_root=tmp_path) == 6
        assert next_run_number("P01", "S01", "thumbCmcIso", data_root=tmp_path) == 2

    def test_ignores_non_bids_files(self, tmp_path):
        sdir = tmp_path / "sub-P01" / "ses-S01"
        sdir.mkdir(parents=True)
        (sdir / "random.txt").write_text("noise")
        (sdir / "sub-P01_ses-S01_physio.json").write_text("{}")  # metadata, not a run
        assert next_run_number("P01", "S01", "powerGrip", data_root=tmp_path) == 0


class TestListSubjects:
    def test_empty_data_root(self, tmp_path):
        assert list_subjects(data_root=tmp_path) == []

    def test_lists_subject_labels(self, tmp_path):
        (tmp_path / "sub-P01").mkdir()
        (tmp_path / "sub-P02").mkdir()
        (tmp_path / "random_dir").mkdir()  # not a subject
        result = list_subjects(data_root=tmp_path)
        assert result == ["P01", "P02"]

    def test_ignores_files(self, tmp_path):
        (tmp_path / "sub-P01").mkdir()
        (tmp_path / "dataset_description.json").write_text("{}")
        (tmp_path / "participants.tsv").write_text("header\n")
        result = list_subjects(data_root=tmp_path)
        assert result == ["P01"]

    def test_sorted_output(self, tmp_path):
        (tmp_path / "sub-P03").mkdir()
        (tmp_path / "sub-P01").mkdir()
        (tmp_path / "sub-P02").mkdir()
        result = list_subjects(data_root=tmp_path)
        assert result == ["P01", "P02", "P03"]


class TestNextSubjectLabel:
    def test_empty_data_root_returns_P01(self, tmp_path):
        assert next_subject_label(data_root=tmp_path) == "P01"

    def test_after_existing_returns_next(self, tmp_path):
        (tmp_path / "sub-P01").mkdir()
        (tmp_path / "sub-P02").mkdir()
        assert next_subject_label(data_root=tmp_path) == "P03"

    def test_skips_gaps(self, tmp_path):
        (tmp_path / "sub-P01").mkdir()
        (tmp_path / "sub-P05").mkdir()
        assert next_subject_label(data_root=tmp_path) == "P06"

    def test_with_non_matching_labels(self, tmp_path):
        (tmp_path / "sub-P01").mkdir()
        (tmp_path / "sub-TEST").mkdir()  # non-numeric, ignored for P{NN} pattern
        result = next_subject_label(data_root=tmp_path)
        # Should still find P01 and increment to P02
        assert result == "P02"


class TestNextSessionLabel:
    def test_no_subject_returns_S01(self, tmp_path):
        assert next_session_label("P01", data_root=tmp_path) == "S01"

    def test_subject_no_sessions_returns_S01(self, tmp_path):
        (tmp_path / "sub-P01").mkdir()
        assert next_session_label("P01", data_root=tmp_path) == "S01"

    def test_after_existing_sessions_returns_next(self, tmp_path):
        (tmp_path / "sub-P01" / "ses-S01").mkdir(parents=True)
        (tmp_path / "sub-P01" / "ses-S02").mkdir(parents=True)
        assert next_session_label("P01", data_root=tmp_path) == "S03"

    def test_skips_gaps(self, tmp_path):
        (tmp_path / "sub-P01" / "ses-S01").mkdir(parents=True)
        (tmp_path / "sub-P01" / "ses-S05").mkdir(parents=True)
        assert next_session_label("P01", data_root=tmp_path) == "S06"

    def test_ignores_other_subjects(self, tmp_path):
        (tmp_path / "sub-P01" / "ses-S01").mkdir(parents=True)
        (tmp_path / "sub-P02" / "ses-S01").mkdir(parents=True)
        (tmp_path / "sub-P02" / "ses-S02").mkdir(parents=True)
        # P01 has only S01, so next is S02 (not affected by P02's S02)
        assert next_session_label("P01", data_root=tmp_path) == "S02"

    def test_with_non_matching_labels(self, tmp_path):
        (tmp_path / "sub-P01" / "ses-S01").mkdir(parents=True)
        (tmp_path / "sub-P01" / "ses-BASELINE").mkdir(parents=True)
        result = next_session_label("P01", data_root=tmp_path)
        # Should find S01 and increment to S02
        assert result == "S02"
