"""Extract ML-ready data from the curated sample dataset.

Reads the single merged session record CSV from ``data/sample/`` and writes
ML-ready Parquet files to ``data/derived/``.

**Format**: Raw samples (one row per 100Hz tick). No windowing or feature
extraction is done here — that happens at model-training time. This preserves
maximum flexibility: classification, regression, and temporal models can all
use the same Parquet file.

**Label column**: ``label`` is set to the task name during RECORD phase and
"rest" during REST phase (if ``include_rest_as_class`` is True). PREP phase
rows are excluded entirely.

**Reaction time trim**: The first ``reaction_trim_ms`` milliseconds of each
RECORD phase are trimmed to remove the subject's reaction delay (the time
between the cue appearing and the subject starting the movement). This is
configurable via CLI.

**Dual classification + regression**: The same Parquet file supports both:
  - Classification: use ``label`` column as target
  - Regression: use ``target_*`` columns (15 joint angles) as targets

Output structure::

    data/derived/sub-P01/ses-S01/
    └── sub-P01_ses-S01_record.parquet

Usage::

    python extract_ml_data.py                     # extract from data/sample/
    python extract_ml_data.py --dry-run           # report shapes without saving
    python extract_ml_data.py --reaction-trim-ms 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from core import paths
from core.loader import load_all_session_records
from core.schema import TARGET_COLUMNS, FREEFORM_TASKS, BASELINE_TASK


# ── Label derivation ─────────────────────────────────────────────────────────


def derive_labels(
    df: pd.DataFrame,
    *,
    include_rest_as_class: bool = True,
    reaction_trim_ms: int = 200,
    exclude_tasks: list[str] | None = None,
) -> pd.DataFrame:
    """Add a ``label`` column and apply reaction time trimming.

    - RECORD phase: label = task name (after trimming first N ms)
    - REST phase: label = "rest" (if include_rest_as_class)
    - PREP phase: rows are excluded (returned as None label, filtered later)
    - Tasks in ``exclude_tasks`` are labeled None (excluded from ML)

    The reaction time trim removes the first ``reaction_trim_ms`` of each
    RECORD phase within each run, since the subject hasn't started moving yet.
    """
    df = df.copy()

    if exclude_tasks is None:
        exclude_tasks = list(FREEFORM_TASKS) + [BASELINE_TASK]

    # Default: no label
    df["label"] = None

    if "phase" not in df.columns:
        return df

    # RECORD phase: label = task (unless excluded)
    record_mask = df["phase"] == "RECORD"
    excluded_mask = df["task"].isin(exclude_tasks)
    df.loc[record_mask & ~excluded_mask, "label"] = df.loc[record_mask & ~excluded_mask, "task"]

    # REST phase: label = "rest"
    if include_rest_as_class:
        rest_mask = df["phase"] == "REST"
        df.loc[rest_mask, "label"] = "rest"

    # Reaction time trim: mark first N ms of each RECORD phase as None
    if reaction_trim_ms > 0 and record_mask.any():
        trim_ns = reaction_trim_ms * 1_000_000
        for (task, run), group in df[record_mask].groupby(["task", "run"], observed=True):
            if group.empty:
                continue
            first_ts = group["t_monotonic_ns"].min()
            trim_mask = (df["task"] == task) & (df["run"] == run) & (df["phase"] == "RECORD") & (df["t_monotonic_ns"] < first_ts + trim_ns)
            df.loc[trim_mask, "label"] = None

    return df


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract ML-ready Parquet from curated sample data.")
    parser.add_argument("--data-root", type=str, default=None, help="Override source data root (default: data/sample/)")
    parser.add_argument("--dry-run", action="store_true", help="Report shapes without saving")
    parser.add_argument("--reaction-trim-ms", type=int, default=200,
                        help="Trim first N ms of RECORD phase for reaction time (default: 200)")
    parser.add_argument("--no-rest-class", action="store_true",
                        help="Exclude REST phase (don't include 'rest' as a class)")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else paths.SAMPLE_DIR
    print("JoTouch ML Data Extraction (raw samples Parquet)")
    print("=" * 60)
    print(f"Source: {data_root}")

    records = load_all_session_records(data_root=data_root)
    if not records:
        print(f"No session records found in {data_root}")
        return 1

    print(f"Sessions: {len(records)}")

    derived = paths.DERIVED_DIR
    total_samples = 0
    total_record = 0
    total_rest = 0

    for (sub, ses), record_df in records.items():
        print(f"\n--- {sub}/{ses} ---")
        print(f"  Raw rows: {len(record_df)}")

        # Derive labels
        labeled = derive_labels(
            record_df,
            include_rest_as_class=not args.no_rest_class,
            reaction_trim_ms=args.reaction_trim_ms,
        )

        # Filter: exclude PREP phase and unlabeled rows (from reaction trim)
        filtered = labeled[labeled["label"].notna()].copy()
        print(f"  After label filter (excl PREP + reaction trim): {len(filtered)}")

        # Count by label
        label_counts = filtered["label"].value_counts()
        print(f"  Labels: {len(label_counts)} classes")
        for label, count in label_counts.items():
            print(f"    {label}: {count}")

        # Verify target columns exist
        available_targets = [c for c in TARGET_COLUMNS if c in filtered.columns]
        print(f"  Target columns: {len(available_targets)}/15")

        total_samples += len(record_df)
        total_record += int((filtered["label"] != "rest").sum()) if "rest" in label_counts.index else len(filtered)
        total_rest += int((filtered["label"] == "rest").sum()) if "rest" in label_counts.index else 0

        if args.dry_run:
            print(f"  (dry-run — not saved)")
            continue

        # Write Parquet
        out_dir = derived / f"sub-{sub}" / f"ses-{ses}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"sub-{sub}_ses-{ses}_record.parquet"
        print(f"  Writing {out_path.name}...")
        filtered.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"  Saved: {out_path}")

    print(f"\n{'=' * 60}")
    print(f"Total: {len(records)} sessions, {total_samples} raw samples")
    print(f"  Record samples: {total_record}")
    print(f"  Rest samples: {total_rest}")
    if args.dry_run:
        print("(dry-run — no files saved)")
    else:
        print(f"Output: {derived}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
