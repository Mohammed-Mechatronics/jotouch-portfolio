"""Tests for apps.collection.cues — visual cues for tasks."""

from __future__ import annotations

import pytest

from apps.collection.cues import CUES, get_cue, format_cue_for_display, format_countdown
from core.schema import ALL_TASKS


class TestCues:
    def test_all_tasks_have_cues(self):
        """Every task in the taxonomy should have a cue defined."""
        for task in ALL_TASKS:
            assert task in CUES, f"Missing cue for task: {task}"

    def test_cue_count(self):
        assert len(CUES) == 26  # 1 baseline + 15 + 9 + 1

    def test_get_cue_existing(self):
        cue = get_cue("powerGrip")
        assert cue.task == "powerGrip"
        assert cue.display_name == "Power Grip"
        assert "cylinder" in cue.instruction.lower()

    def test_get_cue_missing(self):
        cue = get_cue("nonexistent")
        assert cue.task == "nonexistent"
        assert cue.display_name == "nonexistent"

    def test_mvc_cue(self):
        cue = get_cue("mvc")
        assert "squeeze" in cue.instruction.lower()

    def test_freeform_cue(self):
        cue = get_cue("freeform")
        assert "60" in cue.instruction

    def test_single_dof_cues_have_isolation(self):
        from core.schema import SINGLE_DOF_TASKS
        for task in SINGLE_DOF_TASKS:
            cue = get_cue(task)
            assert "only" in cue.instruction.lower(), f"{task} cue should emphasize isolation"


class TestFormatCue:
    def test_format_for_display(self):
        cue = get_cue("powerGrip")
        text = format_cue_for_display(cue, 46, 1, 3)
        assert "Power Grip" in text
        assert "46" in text
        assert "Rep 1/3" in text

    def test_format_countdown(self):
        assert format_countdown(3.0) == "Starting in 3s..."
        assert format_countdown(0) == "GO!"
        assert format_countdown(-1) == "GO!"
