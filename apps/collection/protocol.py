"""3-phase regression collection protocol.

Defines the full 76-run protocol:
  Baseline:  1 task  (mvc)                    x 1 rep  =  1 run
  Phase 1:  15 tasks (single-DOF isolation)   x 3 reps = 45 runs
  Phase 2:   9 tasks (multi-DOF combinations) x 3 reps = 27 runs
  Phase 3:   1 task  (freeform)               x 3 reps =  3 runs
  Total: 76 runs

Each run has a structured timing pattern:
  PREP (3s countdown) -> RECORD (variable) -> REST (3s)
  Freeform runs: PREP (5s) -> RECORD (60s) -> REST (5s)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from core.schema import (
    BASELINE_TASK,
    SINGLE_DOF_TASKS,
    MULTI_DOF_TASKS,
    FREEFORM_TASKS,
    ALL_TASKS,
    task_phase,
)


class RunState(Enum):
    """States within a single run."""
    IDLE = "IDLE"
    PREP = "PREP"          # countdown before recording
    RECORD = "RECORD"      # active data recording
    REST = "REST"          # rest between reps
    DONE = "DONE"          # run complete


@dataclass(frozen=True)
class RunSpec:
    """Specification for a single run in the protocol."""

    task: str
    run: int               # run number (0 = baseline)
    rep: int               # rep number within task (1, 2, 3)
    phase: str             # "baseline", "single_dof", "multi_dof", "freeform"
    record_duration: float # seconds
    prep_duration: float   # seconds
    rest_duration: float   # seconds

    @property
    def is_baseline(self) -> bool:
        return self.run == 0

    @property
    def is_freeform(self) -> bool:
        return self.phase == "freeform"

    @property
    def label(self) -> str:
        """Human-readable label for UI display."""
        return f"{self.task} (run {self.run:02d}, rep {self.rep})"


# ── Timing defaults ───────────────────────────────────────────────────────────

DEFAULT_PREP_S = 6.0       # countdown before recording (time to read the cue)
DEFAULT_REST_S = 3.0       # rest between reps
DEFAULT_RECORD_S = 10.0    # recording duration for structured tasks
FREEFORM_RECORD_S = 60.0   # recording duration for freeform
FREEFORM_PREP_S = 5.0      # longer prep for freeform
FREEFORM_REST_S = 5.0      # longer rest for freeform
MVC_RECORD_S = 5.0         # MVC baseline: 5s max voluntary contraction
MVC_PREP_S = 5.0           # longer prep for MVC


def build_protocol(
    *,
    n_reps: int = 3,
    record_duration: float = DEFAULT_RECORD_S,
    prep_duration: float = DEFAULT_PREP_S,
    rest_duration: float = DEFAULT_REST_S,
    include_freeform: bool = True,
) -> list[RunSpec]:
    """Build the full 76-run protocol.

    Parameters
    ----------
    n_reps : int
        Number of repetitions per task (default 3).
    record_duration : float
        Recording duration for structured tasks (seconds).
    prep_duration : float
        Prep countdown duration (seconds).
    rest_duration : float
        Rest between reps (seconds).
    include_freeform : bool
        Whether to include freeform runs.

    Returns
    -------
    list[RunSpec]
        Ordered list of all runs in the protocol.
    """
    runs: list[RunSpec] = []
    run_counter = 0

    # Baseline: MVC (run-00)
    runs.append(RunSpec(
        task=BASELINE_TASK,
        run=run_counter,
        rep=1,
        phase="baseline",
        record_duration=MVC_RECORD_S,
        prep_duration=MVC_PREP_S,
        rest_duration=DEFAULT_REST_S,
    ))
    run_counter += 1

    # Phase 1: Single-DOF isolation (15 tasks x n_reps)
    for task in SINGLE_DOF_TASKS:
        for rep in range(1, n_reps + 1):
            runs.append(RunSpec(
                task=task,
                run=run_counter,
                rep=rep,
                phase="single_dof",
                record_duration=record_duration,
                prep_duration=prep_duration,
                rest_duration=rest_duration,
            ))
            run_counter += 1

    # Phase 2: Multi-DOF combinations (9 tasks x n_reps)
    for task in MULTI_DOF_TASKS:
        for rep in range(1, n_reps + 1):
            runs.append(RunSpec(
                task=task,
                run=run_counter,
                rep=rep,
                phase="multi_dof",
                record_duration=record_duration,
                prep_duration=prep_duration,
                rest_duration=rest_duration,
            ))
            run_counter += 1

    # Phase 3: Freeform (1 task x n_reps, 60s each)
    if include_freeform:
        for rep in range(1, n_reps + 1):
            runs.append(RunSpec(
                task=FREEFORM_TASKS[0],
                run=run_counter,
                rep=rep,
                phase="freeform",
                record_duration=FREEFORM_RECORD_S,
                prep_duration=FREEFORM_PREP_S,
                rest_duration=FREEFORM_REST_S,
            ))
            run_counter += 1

    return runs


def protocol_summary(runs: list[RunSpec]) -> dict:
    """Return a summary of the protocol for display/logging."""
    phases = {}
    for run in runs:
        if run.phase not in phases:
            phases[run.phase] = {"tasks": set(), "runs": 0, "duration_s": 0.0}
        phases[run.phase]["tasks"].add(run.task)
        phases[run.phase]["runs"] += 1
        phases[run.phase]["duration_s"] += run.prep_duration + run.record_duration + run.rest_duration

    return {
        "total_runs": len(runs),
        "total_duration_s": sum(r.prep_duration + r.record_duration + r.rest_duration for r in runs),
        "phases": {
            phase: {
                "n_tasks": len(data["tasks"]),
                "n_runs": data["runs"],
                "duration_s": round(data["duration_s"], 1),
                "tasks": sorted(data["tasks"]),
            }
            for phase, data in phases.items()
        },
    }


def runs_for_phase(runs: list[RunSpec], phase: str) -> list[RunSpec]:
    """Filter runs to a specific phase."""
    return [r for r in runs if r.phase == phase]


def runs_for_task(runs: list[RunSpec], task: str) -> list[RunSpec]:
    """Filter runs to a specific task."""
    return [r for r in runs if r.task == task]
