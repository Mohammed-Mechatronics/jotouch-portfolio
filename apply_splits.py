"""Apply split specifications to derived ML data.

Reads split spec YAML files from ``data/splits/`` and generates fold
assignments for each session's derived Parquet data. Writes split result
JSON files to ``data/splits/results/``.

The split result JSON contains group assignments (which runs go to
train/test), not sample indices. The experiment code uses this to filter
the Parquet at load time.

Usage::

    python apply_splits.py                          # apply all specs to all sessions
    python apply_splits.py --spec cross_trial       # apply one spec
    python apply_splits.py --sub P01 --ses S01      # apply to one session
    python apply_splits.py --write-folds            # also write per-fold Parquet files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from core import paths
from core.loader import load_all_session_records
from core.splits import (
    SplitSpec,
    SplitEngine,
    save_split_result,
    list_split_specs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply split specs to derived ML data.")
    parser.add_argument("--spec", type=str, default=None, help="Specific spec name (without .yaml)")
    parser.add_argument("--sub", type=str, default=None, help="Specific subject")
    parser.add_argument("--ses", type=str, default=None, help="Specific session")
    parser.add_argument("--write-folds", action="store_true",
                        help="Also write per-fold Parquet files (train.parquet, test.parquet)")
    parser.add_argument("--data-root", type=str, default=None, help="Override source data root")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else paths.SAMPLE_DIR
    splits_dir = paths.SPLITS_DIR

    print("JoTouch Split Generation")
    print("=" * 60)
    print(f"Specs dir: {splits_dir}")
    print(f"Data root: {data_root}")

    # Load split specs
    spec_paths = list_split_specs(splits_dir)
    if args.spec:
        spec_paths = [p for p in spec_paths if p.stem == args.spec]

    if not spec_paths:
        print(f"No split specs found in {splits_dir}")
        return 1

    print(f"Specs: {[p.stem for p in spec_paths]}")

    # Load session records
    records = load_all_session_records(data_root=data_root)
    if not records:
        print(f"No session records found in {data_root}")
        return 1

    # Filter to specific session if requested
    if args.sub and args.ses:
        key = (args.sub, args.ses)
        if key not in records:
            print(f"Session {args.sub}/{args.ses} not found")
            return 1
        records = {key: records[key]}

    print(f"Sessions: {list(records.keys())}")

    results_dir = splits_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for spec_path in spec_paths:
        spec = SplitSpec.from_yaml(spec_path)
        print(f"\n--- Spec: {spec.name} ---")
        print(f"  Level: {spec.generalization_level}")
        print(f"  Unit: {spec.unit_group_by}")
        print(f"  Strategy: {spec.strategy_type} on {spec.strategy_column}")

        for (sub, ses), record_df in records.items():
            print(f"\n  {sub}/{ses}:")

            # Build run metadata from the record
            metadata_df = SplitEngine.build_run_metadata(record_df)
            print(f"    Runs: {len(metadata_df)}")

            # Generate folds
            result = SplitEngine.generate(spec, metadata_df)
            print(f"    Folds: {len(result.folds)}")
            print(f"    Snapshot: {result.data_snapshot}")

            for fold in result.folds:
                print(f"    {fold.name}: train={fold.stats.get('n_train_runs', 0)} runs "
                      f"({fold.stats.get('n_train_record_samples', 0)} RECORD samples), "
                      f"test={fold.stats.get('n_test_runs', 0)} runs "
                      f"({fold.stats.get('n_test_record_samples', 0)} RECORD samples)")

            # Save result JSON
            result_path = results_dir / f"sub-{sub}" / f"ses-{ses}" / f"{spec.name}_result.json"
            save_split_result(result, result_path)
            print(f"    Saved: {result_path}")

            # Optionally write per-fold Parquet files
            if args.write_folds and result.folds:
                derived_dir = paths.DERIVED_DIR / f"sub-{sub}" / f"ses-{ses}"
                if not derived_dir.exists():
                    print(f"    (no derived data at {derived_dir} — skipping fold Parquet)")
                    continue

                parquet_path = derived_dir / f"sub-{sub}_ses-{ses}_record.parquet"
                if not parquet_path.exists():
                    print(f"    (no Parquet at {parquet_path} — skipping fold Parquet)")
                    continue

                parquet_df = pd.read_parquet(parquet_path)
                fold_dir = derived_dir / "folds" / spec.name
                fold_dir.mkdir(parents=True, exist_ok=True)

                for fold in result.folds:
                    # Build a set of (task, run) tuples for train and test
                    train_keys = set()
                    for g in fold.train_groups:
                        for rk in g.get("run_keys", []):
                            train_keys.add((rk["task"], rk["run"]))

                    test_keys = set()
                    for g in fold.test_groups:
                        for rk in g.get("run_keys", []):
                            test_keys.add((rk["task"], rk["run"]))

                    # Filter Parquet
                    train_mask = parquet_df.apply(
                        lambda r: (r["task"], r["run"]) in train_keys, axis=1
                    )
                    test_mask = parquet_df.apply(
                        lambda r: (r["task"], r["run"]) in test_keys, axis=1
                    )

                    train_df = parquet_df[train_mask]
                    test_df = parquet_df[test_mask]

                    train_path = fold_dir / f"{fold.name}_train.parquet"
                    test_path = fold_dir / f"{fold.name}_test.parquet"
                    train_df.to_parquet(train_path, index=False)
                    test_df.to_parquet(test_path, index=False)
                    print(f"    {fold.name}: train={len(train_df)}, test={len(test_df)} → {fold_dir.name}/")

    print(f"\n{'=' * 60}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
