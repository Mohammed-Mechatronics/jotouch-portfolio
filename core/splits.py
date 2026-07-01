"""Declarative split system for JoTouch ML data.

Supports three generalization levels via a single YAML spec format:

    cross_trial    — leave-one-rep-out within subject/session
    cross_session  — leave-one-session-out within subject
    cross_subject  — leave-one-subject-out (LOSO)
    cross_task     — leave-one-task-out

The spec defines **rules** (not lists). The engine evaluates rules against
the actual metadata to produce fold assignments. This makes the system:

- **Adaptive**: auto-detects available subjects/sessions/reps
- **Scalable**: spec is constant size regardless of data volume
- **UI-friendly**: maps directly to dropdowns in a form
- **Sweepable**: multiple specs in a directory, iterate over all

Spec format (YAML)::

    name: cross_trial
    description: "Leave-one-rep-out within subject/session"
    generalization_level: cross_trial

    unit:
      group_by: [sub, ses, task, rep]

    strategy:
      type: leave_one_out
      leave_one_out_column: rep

    filters:
      exclude_tasks: [mvc, freeform]
      exclude_phases: [PREP]
      include_rest_as_class: true

    stratify_by: task
    splits: [train, test]
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class SplitSpec:
    """Declarative split specification loaded from YAML."""

    name: str
    description: str = ""
    generalization_level: str = ""

    # Unit: the granularity at which we split (no leakage within a unit)
    unit_group_by: list[str] = field(default_factory=lambda: ["sub", "ses", "task", "rep"])

    # Strategy: how groups are assigned to splits
    strategy_type: str = "leave_one_out"  # leave_one_out | random | temporal | custom
    strategy_column: str = "rep"  # column to rotate/shuffle
    test_size: float = 0.2  # for random strategy
    random_seed: int = 42  # for random strategy

    # Filters
    exclude_tasks: list[str] = field(default_factory=lambda: ["mvc", "freeform"])
    exclude_phases: list[str] = field(default_factory=lambda: ["PREP"])
    include_rest_as_class: bool = True

    # Stratification
    stratify_by: str = "task"

    # Output splits
    splits: list[str] = field(default_factory=lambda: ["train", "test"])

    @classmethod
    def from_yaml(cls, path: Path) -> "SplitSpec":
        """Load a split spec from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        unit = data.get("unit", {})
        strategy = data.get("strategy", {})
        filters = data.get("filters", {})

        return cls(
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            generalization_level=data.get("generalization_level", ""),
            unit_group_by=unit.get("group_by", ["sub", "ses", "task", "rep"]),
            strategy_type=strategy.get("type", "leave_one_out"),
            strategy_column=strategy.get("leave_one_out_column", strategy.get("column", "rep")),
            test_size=strategy.get("test_size", 0.2),
            random_seed=strategy.get("random_seed", 42),
            exclude_tasks=filters.get("exclude_tasks", ["mvc", "freeform"]),
            exclude_phases=filters.get("exclude_phases", ["PREP"]),
            include_rest_as_class=filters.get("include_rest_as_class", True),
            stratify_by=data.get("stratify_by", "task"),
            splits=data.get("splits", ["train", "test"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a dict (for JSON output)."""
        return {
            "name": self.name,
            "description": self.description,
            "generalization_level": self.generalization_level,
            "unit": {"group_by": self.unit_group_by},
            "strategy": {
                "type": self.strategy_type,
                "leave_one_out_column": self.strategy_column,
                "test_size": self.test_size,
                "random_seed": self.random_seed,
            },
            "filters": {
                "exclude_tasks": self.exclude_tasks,
                "exclude_phases": self.exclude_phases,
                "include_rest_as_class": self.include_rest_as_class,
            },
            "stratify_by": self.stratify_by,
            "splits": self.splits,
        }

    def content_hash(self) -> str:
        """Stable hash of the spec content for reproducibility."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return "sha256:" + hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class SplitFold:
    """One fold of a split result."""

    name: str
    train_groups: list[dict[str, Any]] = field(default_factory=list)
    test_groups: list[dict[str, Any]] = field(default_factory=list)
    val_groups: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


@dataclass
class SplitResult:
    """Result of evaluating a split spec against metadata."""

    spec: SplitSpec
    folds: list[SplitFold] = field(default_factory=list)
    generated_at: str = ""
    data_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for JSON output."""
        return {
            "spec": self.spec.to_dict(),
            "spec_hash": self.spec.content_hash(),
            "generated_at": self.generated_at,
            "data_snapshot": self.data_snapshot,
            "n_folds": len(self.folds),
            "folds": [
                {
                    "name": f.name,
                    "train_groups": f.train_groups,
                    "test_groups": f.test_groups,
                    "val_groups": f.val_groups,
                    "stats": f.stats,
                }
                for f in self.folds
            ],
        }


# ── Split engine ─────────────────────────────────────────────────────────────


class SplitEngine:
    """Evaluates a SplitSpec against run metadata to produce folds.

    The engine is stateless — it takes metadata and a spec, returns folds.
    """

    @staticmethod
    def build_run_metadata(record_df: "pd.DataFrame") -> "pd.DataFrame":
        """Extract one row per run from a merged session record.

        Returns a DataFrame with columns: sub, ses, task, run, run_idx, rep,
        phase (list), n_samples, n_record_samples.
        """
        import pandas as pd

        if record_df.empty:
            return pd.DataFrame()

        # Group by (sub, ses, task, run, run_idx, rep) and aggregate
        groups = record_df.groupby(
            ["participant_id", "session_id", "task", "run", "run_idx", "rep"],
            observed=True,
        )

        rows = []
        for (sub, ses, task, run, run_idx, rep), group in groups:
            phases = group["phase"].unique().tolist() if "phase" in group.columns else []
            n_samples = len(group)
            n_record = int((group["phase"] == "RECORD").sum()) if "phase" in group.columns else n_samples
            rows.append({
                "sub": sub,
                "ses": ses,
                "task": task,
                "run": run,
                "run_idx": run_idx,
                "rep": rep,
                "phases": phases,
                "n_samples": n_samples,
                "n_record_samples": n_record,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def generate(spec: SplitSpec, metadata_df: "pd.DataFrame") -> SplitResult:
        """Generate split folds from metadata according to the spec.

        Parameters
        ----------
        spec : SplitSpec
            The declarative split specification.
        metadata_df : pd.DataFrame
            One row per run, with columns: sub, ses, task, run, run_idx, rep,
            n_samples, n_record_samples.

        Returns
        -------
        SplitResult
        """
        import pandas as pd

        result = SplitResult(
            spec=spec,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        if metadata_df.empty:
            return result

        # Data snapshot
        result.data_snapshot = {
            "n_subjects": int(metadata_df["sub"].nunique()),
            "n_sessions": int(metadata_df.groupby(["sub", "ses"]).ngroups),
            "n_tasks": int(metadata_df["task"].nunique()),
            "n_runs": int(len(metadata_df)),
            "reps_per_task": metadata_df.groupby("task")["rep"].nunique().to_dict(),
        }

        # Apply filters
        df = metadata_df.copy()
        if spec.exclude_tasks:
            df = df[~df["task"].isin(spec.exclude_tasks)]

        if df.empty:
            logger.warning("All runs filtered out by exclude_tasks")
            return result

        # Create groups based on unit_group_by
        group_cols = spec.unit_group_by
        groups = df.groupby(group_cols, observed=True)

        # Build group list
        group_list = []
        for group_key, group_data in groups:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_dict = dict(zip(group_cols, group_key))
            group_dict["n_runs"] = len(group_data)
            group_dict["n_samples"] = int(group_data["n_samples"].sum())
            group_dict["n_record_samples"] = int(group_data["n_record_samples"].sum())
            group_dict["_run_keys"] = group_data[["task", "run"]].to_dict("records")
            group_list.append(group_dict)

        # Apply strategy
        if spec.strategy_type == "leave_one_out":
            folds = SplitEngine._leave_one_out(
                group_list, spec, group_cols
            )
        elif spec.strategy_type == "random":
            folds = SplitEngine._random_split(
                group_list, spec, group_cols
            )
        elif spec.strategy_type == "temporal":
            folds = SplitEngine._temporal_split(
                group_list, spec, group_cols
            )
        else:
            logger.warning("Unknown strategy type: %s, falling back to leave_one_out", spec.strategy_type)
            folds = SplitEngine._leave_one_out(
                group_list, spec, group_cols
            )

        result.folds = folds
        return result

    @staticmethod
    def _leave_one_out(
        groups: list[dict[str, Any]],
        spec: SplitSpec,
        group_cols: list[str],
    ) -> list[SplitFold]:
        """Leave-one-out strategy: rotate the strategy_column."""
        col = spec.strategy_column

        # Get unique values of the rotation column
        unique_vals = sorted(set(g.get(col) for g in groups if g.get(col) is not None))

        if not unique_vals:
            logger.warning("No unique values for leave_one_out column: %s", col)
            return []

        folds = []
        for val in unique_vals:
            train_groups = [g for g in groups if g.get(col) != val]
            test_groups = [g for g in groups if g.get(col) == val]

            # Clean up group dicts (remove internal keys)
            train_clean = [SplitEngine._clean_group(g) for g in train_groups]
            test_clean = [SplitEngine._clean_group(g) for g in test_groups]

            fold = SplitFold(
                name=f"fold_{val}",
                train_groups=train_clean,
                test_groups=test_clean,
                stats={
                    "n_train_groups": len(train_clean),
                    "n_test_groups": len(test_clean),
                    "n_train_runs": sum(g["n_runs"] for g in train_clean),
                    "n_test_runs": sum(g["n_runs"] for g in test_clean),
                    "n_train_samples": sum(g["n_samples"] for g in train_clean),
                    "n_test_samples": sum(g["n_samples"] for g in test_clean),
                    "n_train_record_samples": sum(g["n_record_samples"] for g in train_clean),
                    "n_test_record_samples": sum(g["n_record_samples"] for g in test_clean),
                },
            )
            folds.append(fold)

        return folds

    @staticmethod
    def _random_split(
        groups: list[dict[str, Any]],
        spec: SplitSpec,
        group_cols: list[str],
    ) -> list[SplitFold]:
        """Random split strategy."""
        import random

        rng = random.Random(spec.random_seed)
        shuffled = list(groups)
        rng.shuffle(shuffled)

        n_test = max(1, int(len(shuffled) * spec.test_size))
        test_groups = shuffled[:n_test]
        train_groups = shuffled[n_test:]

        train_clean = [SplitEngine._clean_group(g) for g in train_groups]
        test_clean = [SplitEngine._clean_group(g) for g in test_groups]

        fold = SplitFold(
            name="fold_1",
            train_groups=train_clean,
            test_groups=test_clean,
            stats={
                "n_train_groups": len(train_clean),
                "n_test_groups": len(test_clean),
                "n_train_runs": sum(g["n_runs"] for g in train_clean),
                "n_test_runs": sum(g["n_runs"] for g in test_clean),
                "n_train_samples": sum(g["n_samples"] for g in train_clean),
                "n_test_samples": sum(g["n_samples"] for g in test_clean),
            },
        )
        return [fold]

    @staticmethod
    def _temporal_split(
        groups: list[dict[str, Any]],
        spec: SplitSpec,
        group_cols: list[str],
    ) -> list[SplitFold]:
        """Temporal split: first N-1 reps → train, last rep → test."""
        col = spec.strategy_column
        unique_vals = sorted(set(g.get(col) for g in groups if g.get(col) is not None))

        if len(unique_vals) < 2:
            return SplitEngine._leave_one_out(groups, spec, group_cols)

        # Use last value as test, rest as train
        test_val = unique_vals[-1]
        train_groups = [g for g in groups if g.get(col) != test_val]
        test_groups = [g for g in groups if g.get(col) == test_val]

        train_clean = [SplitEngine._clean_group(g) for g in train_groups]
        test_clean = [SplitEngine._clean_group(g) for g in test_groups]

        fold = SplitFold(
            name="fold_1",
            train_groups=train_clean,
            test_groups=test_clean,
            stats={
                "n_train_groups": len(train_clean),
                "n_test_groups": len(test_clean),
                "n_train_runs": sum(g["n_runs"] for g in train_clean),
                "n_test_runs": sum(g["n_runs"] for g in test_clean),
            },
        )
        return [fold]

    @staticmethod
    def _clean_group(g: dict[str, Any]) -> dict[str, Any]:
        """Remove internal keys from a group dict."""
        return {k: v for k, v in g.items() if not k.startswith("_")}


# ── I/O helpers ──────────────────────────────────────────────────────────────


def save_split_result(result: SplitResult, path: Path) -> None:
    """Save a split result to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)


def load_split_result(path: Path) -> dict[str, Any]:
    """Load a split result from JSON (returns raw dict)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_split_specs(splits_dir: Path | None = None) -> list[Path]:
    """List all split spec YAML files in the splits directory."""
    sdir = splits_dir or Path(__file__).resolve().parent.parent / "data" / "splits"
    if not sdir.exists():
        return []
    return sorted(sdir.glob("*.yaml")) + sorted(sdir.glob("*.yml"))
