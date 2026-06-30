"""Tests for apps.collection.protocol — 3-phase task protocol."""

from __future__ import annotations

import pytest

from apps.collection.protocol import (
    build_protocol,
    protocol_summary,
    runs_for_phase,
    runs_for_task,
    RunSpec,
    RunState,
)


class TestBuildProtocol:
    def test_default_76_runs(self):
        runs = build_protocol()
        assert len(runs) == 76  # 1 + 45 + 27 + 3

    def test_baseline_first(self):
        runs = build_protocol()
        assert runs[0].task == "mvc"
        assert runs[0].run == 0
        assert runs[0].is_baseline

    def test_phase_ordering(self):
        runs = build_protocol()
        phases = [r.phase for r in runs]
        # baseline -> single_dof -> multi_dof -> freeform
        assert phases[0] == "baseline"
        assert "single_dof" in phases[1:46]
        assert "multi_dof" in phases[46:73]
        assert "freeform" in phases[73:]

    def test_run_numbers_sequential(self):
        runs = build_protocol()
        run_numbers = [r.run for r in runs]
        assert run_numbers == list(range(76))

    def test_reps_per_task(self):
        runs = build_protocol()
        # Each single-DOF task should have 3 reps
        thumb_runs = runs_for_task(runs, "thumbCmcIso")
        assert len(thumb_runs) == 3
        assert [r.rep for r in thumb_runs] == [1, 2, 3]

    def test_freeform_duration(self):
        runs = build_protocol()
        freeform = runs_for_phase(runs, "freeform")
        assert len(freeform) == 3
        for r in freeform:
            assert r.record_duration == 60.0
            assert r.is_freeform

    def test_no_freeform(self):
        runs = build_protocol(include_freeform=False)
        assert len(runs) == 73  # 76 - 3
        assert all(r.phase != "freeform" for r in runs)

    def test_custom_reps(self):
        runs = build_protocol(n_reps=5)
        # 1 baseline + 15*5 + 9*5 + 1*5 = 1 + 75 + 45 + 5 = 126
        assert len(runs) == 126

    def test_prep_duration_override_applies_to_structured_runs(self):
        """A custom prep_duration applies to single/multi-DOF runs, while
        MVC and freeform keep their longer dedicated prep durations."""
        runs = build_protocol(prep_duration=8.0)
        single_dof = runs_for_phase(runs, "single_dof")
        multi_dof = runs_for_phase(runs, "multi_dof")
        baseline = [r for r in runs if r.is_baseline]
        freeform = runs_for_phase(runs, "freeform")
        assert all(r.prep_duration == 8.0 for r in single_dof)
        assert all(r.prep_duration == 8.0 for r in multi_dof)
        # MVC and freeform keep their own longer prep (not the override)
        assert baseline[0].prep_duration == 5.0
        assert all(r.prep_duration == 5.0 for r in freeform)

    def test_rest_duration_override_applies_to_structured_runs(self):
        runs = build_protocol(rest_duration=4.0)
        single_dof = runs_for_phase(runs, "single_dof")
        assert all(r.rest_duration == 4.0 for r in single_dof)


class TestProtocolSummary:
    def test_summary_has_all_phases(self):
        runs = build_protocol()
        summary = protocol_summary(runs)
        assert summary["total_runs"] == 76
        assert "baseline" in summary["phases"]
        assert "single_dof" in summary["phases"]
        assert "multi_dof" in summary["phases"]
        assert "freeform" in summary["phases"]

    def test_summary_counts(self):
        runs = build_protocol()
        summary = protocol_summary(runs)
        assert summary["phases"]["baseline"]["n_runs"] == 1
        assert summary["phases"]["single_dof"]["n_runs"] == 45
        assert summary["phases"]["multi_dof"]["n_runs"] == 27
        assert summary["phases"]["freeform"]["n_runs"] == 3

    def test_summary_tasks(self):
        runs = build_protocol()
        summary = protocol_summary(runs)
        assert summary["phases"]["baseline"]["n_tasks"] == 1
        assert summary["phases"]["single_dof"]["n_tasks"] == 15
        assert summary["phases"]["multi_dof"]["n_tasks"] == 9
        assert summary["phases"]["freeform"]["n_tasks"] == 1


class TestRunSpec:
    def test_is_baseline(self):
        spec = RunSpec("mvc", 0, 1, "baseline", 5.0, 5.0, 3.0)
        assert spec.is_baseline

    def test_is_freeform(self):
        spec = RunSpec("freeform", 73, 1, "freeform", 60.0, 5.0, 5.0)
        assert spec.is_freeform

    def test_label(self):
        spec = RunSpec("powerGrip", 46, 1, "multi_dof", 10.0, 3.0, 3.0)
        assert "powerGrip" in spec.label
        assert "run 46" in spec.label
