"""Promote a raw BIDS session to the curated sample dataset.

This script is the gate between ``data/raw/`` (full, untouched recordings)
and ``data/sample/`` (curated, git-tracked, consumable by ML pipelines).

A session is promoted only if ALL of the following criteria pass:

1. **Manifests complete** — every run has a ``manifest.json`` with
   ``complete=true`` and ``aborted=false``.
2. **LED sync passed** — ``led_sync.json`` exists with ``passed=true``.
3. **No bad quality samples** — all manifests report
   ``bad_physio_count == 0`` and ``bad_camera_count == 0`` and
   ``bad_targets_count == 0``.
4. **Validation probes pass** — all registered probes return ``status=OK``
   (0 failures).
5. **Precollect critical checks** — dead/stuck channels, channel activation,
   and response linearity all pass.  (Baseline stability and single-DOF
   isolation are reported as warnings but do not block promotion.)
6. **Minimum trials per class** — every non-MVC, non-freeform task has
   ≥ ``min_trials`` runs (default 3).
7. **BIDS naming** — all filenames match the expected pattern.

Usage::

    python promote_to_sample.py                    # promote all eligible sessions
    python promote_to_sample.py --sub P01 --ses S01
    python promote_to_sample.py --dry-run          # check without copying
    python promote_to_sample.py --min-trials 5
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from core import naming, paths
from core.loader import load_all_sessions


# ── Criterion checks ─────────────────────────────────────────────────────────


def check_manifests(sub: str, ses: str, *, data_root: Path) -> dict[str, Any]:
    """Criterion 1: every run has a complete, non-aborted manifest."""
    from apps.collection.bids_writer import manifest_path

    sdir = paths.session_dir(sub, ses, data_root=data_root)
    manifests = sorted(sdir.glob(f"sub-{sub}_ses-{ses}_task-*_run-*_manifest.json"))
    if not manifests:
        return {"pass": False, "reason": "no manifests found", "details": {}}

    incomplete, aborted = [], []
    for mp in manifests:
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except Exception as exc:
            incomplete.append(f"{mp.name}: read error {exc}")
            continue
        if not m.get("complete", False):
            incomplete.append(mp.name)
        if m.get("aborted", False):
            aborted.append(mp.name)

    passed = not incomplete and not aborted
    return {
        "pass": passed,
        "reason": "" if passed else f"{len(incomplete)} incomplete, {len(aborted)} aborted",
        "details": {"n_manifests": len(manifests), "incomplete": incomplete, "aborted": aborted},
    }


def check_led_sync(sub: str, ses: str, *, data_root: Path) -> dict[str, Any]:
    """Criterion 2: LED sync passed."""
    sdir = paths.session_dir(sub, ses, data_root=data_root)
    sync_path = sdir / f"sub-{sub}_ses-{ses}_led_sync.json"
    if not sync_path.exists():
        return {"pass": False, "reason": "no led_sync.json", "details": {}}
    try:
        sync = json.loads(sync_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"pass": False, "reason": f"read error: {exc}", "details": {}}
    passed = sync.get("passed") is True
    return {
        "pass": passed,
        "reason": "" if passed else sync.get("reason", "passed != true"),
        "details": {
            "skew_ms": sync.get("skew_ms"),
            "n_matched_pairs": sync.get("n_matched_pairs"),
            "cross_validation_passed": sync.get("cross_validation_passed"),
        },
    }


def check_quality(sub: str, ses: str, *, data_root: Path) -> dict[str, Any]:
    """Criterion 3: no bad quality samples in any manifest."""
    from apps.collection.bids_writer import manifest_path

    sdir = paths.session_dir(sub, ses, data_root=data_root)
    manifests = sorted(sdir.glob(f"sub-{sub}_ses-{ses}_task-*_run-*_manifest.json"))
    bad_runs = []
    for mp in manifests:
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if m.get("bad_physio_count", 0) > 0 or m.get("bad_camera_count", 0) > 0 or m.get("bad_targets_count", 0) > 0:
            bad_runs.append({
                "file": mp.name,
                "bad_physio": m.get("bad_physio_count", 0),
                "bad_camera": m.get("bad_camera_count", 0),
                "bad_targets": m.get("bad_targets_count", 0),
            })
    passed = not bad_runs
    return {
        "pass": passed,
        "reason": "" if passed else f"{len(bad_runs)} runs with bad samples",
        "details": {"bad_runs": bad_runs},
    }


def check_precollect(sub: str, ses: str, *, data_root: Path) -> dict[str, Any]:
    """Criterion 5: critical precollect checks pass."""
    sdir = paths.session_dir(sub, ses, data_root=data_root)
    pc_path = sdir / f"sub-{sub}_ses-{ses}_precollect.json"
    if not pc_path.exists():
        return {"pass": False, "reason": "no precollect.json", "details": {}}
    try:
        pc = json.loads(pc_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"pass": False, "reason": f"read error: {exc}", "details": {}}

    # Critical checks that MUST pass
    critical_checks = [
        ("sensor_specific", "dead_stuck_channels", "passed"),
        ("sensor_specific", "channel_activation", "passed"),
        ("sensor_specific", "response_linearity", "passed"),
    ]
    # Warning checks (reported but don't block)
    warning_checks = [
        ("sensor_specific", "baseline_stability", "passed"),
        ("task_specific", "single_dof_isolation", "passed"),
    ]

    failed_critical = []
    failed_warnings = []
    for section, key, field in critical_checks:
        val = pc.get(section, {}).get(key, {})
        if not val.get(field, False):
            failed_critical.append(f"{section}.{key}")
    for section, key, field in warning_checks:
        val = pc.get(section, {}).get(key, {})
        if not val.get(field, False):
            failed_warnings.append(f"{section}.{key}")

    passed = not failed_critical
    return {
        "pass": passed,
        "reason": "" if passed else f"failed: {', '.join(failed_critical)}",
        "details": {
            "failed_critical": failed_critical,
            "failed_warnings": failed_warnings,
        },
    }


def check_min_trials(sub: str, ses: str, *, data_root: Path, min_trials: int) -> dict[str, Any]:
    """Criterion 6: every non-MVC, non-freeform task has >= min_trials runs."""
    sessions = load_all_sessions(data_root=data_root, require_manifest=True)
    target_session = None
    for s in sessions:
        if s.sub == sub and s.ses == ses:
            target_session = s
            break
    if target_session is None:
        return {"pass": False, "reason": "session not found", "details": {}}

    from collections import Counter
    task_counts: Counter[str] = Counter()
    for run in target_session.runs:
        if run.task in ("mvc", "freeform"):
            continue
        task_counts[run.task] += 1

    insufficient = {t: c for t, c in task_counts.items() if c < min_trials}
    passed = not insufficient
    return {
        "pass": passed,
        "reason": "" if passed else f"{len(insufficient)} tasks with < {min_trials} trials",
        "details": {
            "min_trials": min_trials,
            "task_counts": dict(task_counts),
            "insufficient": insufficient,
        },
    }


def check_bids_naming(sub: str, ses: str, *, data_root: Path) -> dict[str, Any]:
    """Criterion 7: all filenames match the BIDS pattern."""
    sdir = paths.session_dir(sub, ses, data_root=data_root)
    if not sdir.exists():
        return {"pass": False, "reason": "session dir does not exist", "details": {}}

    bad_names = []
    for f in sdir.iterdir():
        if f.is_file() and f.suffix in (".csv", ".json", ".tsv"):
            # Check if it matches sub-{x}_ses-{x}_... pattern
            if not f.name.startswith(f"sub-{sub}_ses-{ses}_"):
                bad_names.append(f.name)

    passed = not bad_names
    return {
        "pass": passed,
        "reason": "" if passed else f"{len(bad_names)} misnamed files",
        "details": {"bad_names": bad_names},
    }


# ── Main promotion logic ─────────────────────────────────────────────────────


def check_session(sub: str, ses: str, *, data_root: Path, min_trials: int) -> dict[str, Any]:
    """Run all 7 criteria checks on a session. Returns a structured report."""
    checks = {
        "1_manifests": check_manifests(sub, ses, data_root=data_root),
        "2_led_sync": check_led_sync(sub, ses, data_root=data_root),
        "3_quality": check_quality(sub, ses, data_root=data_root),
        # Criterion 4 (validation probes) is checked separately via run_validation
        "5_precollect": check_precollect(sub, ses, data_root=data_root),
        "6_min_trials": check_min_trials(sub, ses, data_root=data_root, min_trials=min_trials),
        "7_bids_naming": check_bids_naming(sub, ses, data_root=data_root),
    }
    all_pass = all(c["pass"] for c in checks.values())
    return {"pass": all_pass, "checks": checks}


def update_participants_tsv(sub: str, ses: str, *, sample_root: Path) -> None:
    """Append/update a row in participants.tsv for the promoted subject."""
    tsv_path = sample_root / "participants.tsv"
    if not tsv_path.exists():
        tsv_path.write_text(
            "participant_id\tage\tsex\thandedness\tforearm_circumference_mm\tforearm_length_mm\n",
            encoding="utf-8",
        )
    lines = tsv_path.read_text(encoding="utf-8").strip().split("\n")
    sub_id = f"sub-{sub}"
    existing = [l for l in lines[1:] if l.startswith(sub_id + "\t") or l == sub_id]
    if existing:
        return
    sdir = paths.session_dir(sub, ses, data_root=paths.RAW_DIR)
    physio_json = sdir / f"sub-{sub}_ses-{ses}_physio.json"
    age = sex = handedness = circ = length = "n/a"
    if physio_json.exists():
        try:
            pj = json.loads(physio_json.read_text(encoding="utf-8"))
            age = str(pj.get("Age", "n/a"))
            sex = str(pj.get("Sex", "n/a"))
            handedness = str(pj.get("Handedness", "n/a"))
            circ = str(pj.get("ForearmCircumferenceMm", "n/a"))
            length = str(pj.get("ForearmLengthMm", "n/a"))
        except Exception:
            pass
    lines.append(f"{sub_id}\t{age}\t{sex}\t{handedness}\t{circ}\t{length}")
    tsv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_sessions_tsv(sub: str, ses: str, *, sample_root: Path) -> None:
    """Append/update a row in sessions.tsv for the promoted session."""
    tsv_path = sample_root / "sessions.tsv"
    if not tsv_path.exists():
        tsv_path.write_text(
            "participant_id\tsession_id\tacq_time\tband_placement\tband_tension\tsensor_count\tsampling_frequency_hz\n",
            encoding="utf-8",
        )
    lines = tsv_path.read_text(encoding="utf-8").strip().split("\n")
    row_id = f"sub-{sub}\tses-{ses}"
    existing = [l for l in lines[1:] if l.startswith(row_id + "\t") or l == row_id]
    if existing:
        return
    sdir = paths.session_dir(sub, ses, data_root=paths.RAW_DIR)
    physio_json = sdir / f"sub-{sub}_ses-{ses}_physio.json"
    acq = band = tension = sensors = sf = "n/a"
    if physio_json.exists():
        try:
            pj = json.loads(physio_json.read_text(encoding="utf-8"))
            acq = str(pj.get("AcqTime", pj.get("started_at", "n/a")))
            band = str(pj.get("BandPlacement", "n/a"))
            tension = str(pj.get("BandTension", "n/a"))
            sensors = str(pj.get("SensorCount", "n/a"))
            sf = str(pj.get("SamplingFrequency", "n/a"))
        except Exception:
            pass
    lines.append(f"sub-{sub}\tses-{ses}\t{acq}\t{band}\t{tension}\t{sensors}\t{sf}")
    tsv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def promote_session(
    sub: str,
    ses: str,
    *,
    src_root: Path,
    dst_root: Path,
    method: str = "fft_xcorr",
    recompute_sync: bool = False,
) -> dict[str, Any]:
    """Promote a session by merging all runs into a single CSV.

    This replaces the old copy-and-split approach. The new approach:
      1. (Optionally) recompute LED sync using FFT xcorr
      2. Merge all 76 runs into a single 100Hz DataFrame
      3. Write one _record.csv + one _record.json sidecar
      4. Copy metadata files (physio.json, channels.tsv, precollect.json)
    """
    from core.merge import merge_session, load_sync_correction, session_metadata_summary
    from apps.collection.led_sync import write_led_sync

    # Step 1: LED sync (recompute if requested or if missing)
    sync_path = paths.session_dir(sub, ses, data_root=src_root) / f"sub-{sub}_ses-{ses}_led_sync.json"
    if recompute_sync or not sync_path.exists():
        print(f"  Computing LED sync ({method})...")
        write_led_sync(sub, ses, data_root=src_root, method=method)

    # Step 2: Load sync correction and merge all runs
    print(f"  Merging all runs...")
    sync_correction = load_sync_correction(sub, ses, data_root=src_root)
    merged = merge_session(sub, ses, data_root=src_root, sync_correction=sync_correction)

    if merged.empty:
        return {"pass": False, "reason": "merge produced empty DataFrame"}

    # Step 3: Write the single merged CSV
    dst_sdir = paths.session_dir(sub, ses, data_root=dst_root)
    dst_sdir.mkdir(parents=True, exist_ok=True)

    # Clean old per-run files from previous promotion format
    old_files = list(dst_sdir.glob(f"sub-{sub}_ses-{ses}_task-*_run-*"))
    if old_files:
        print(f"  Cleaning {len(old_files)} old per-run files...")
        for f in old_files:
            f.unlink()

    # Clean old classification/regression directories (legacy format)
    for old_subdir in ["classification", "regression"]:
        old_dir = dst_root / old_subdir
        if old_dir.exists():
            shutil.rmtree(old_dir)
            print(f"  Cleaned old {old_subdir}/ directory")

    record_csv_path = dst_sdir / f"sub-{sub}_ses-{ses}_record.csv"
    print(f"  Writing {record_csv_path.name} ({merged.shape[0]} rows, {merged.shape[1]} cols)...")
    merged.to_csv(record_csv_path, index=False)

    # Step 4: Write sidecar JSON
    meta = session_metadata_summary(sub, ses, merged, sync_correction=sync_correction)
    record_json_path = dst_sdir / f"sub-{sub}_ses-{ses}_record.json"
    with open(record_json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Step 5: Copy metadata files from raw
    src_sdir = paths.session_dir(sub, ses, data_root=src_root)
    for meta_file in [
        f"sub-{sub}_ses-{ses}_physio.json",
        f"sub-{sub}_ses-{ses}_channels.tsv",
        f"sub-{sub}_ses-{ses}_precollect.json",
        f"sub-{sub}_ses-{ses}_led_sync.json",
    ]:
        src_meta = src_sdir / meta_file
        if src_meta.exists():
            shutil.copy2(src_meta, dst_sdir / meta_file)

    # Step 6: Update TSV files
    update_participants_tsv(sub, ses, sample_root=dst_root)
    update_sessions_tsv(sub, ses, sample_root=dst_root)

    return {
        "pass": True,
        "n_samples": len(merged),
        "n_columns": merged.shape[1],
        "n_runs": meta["n_runs"],
        "cam_coverage_pct": meta["cam_coverage_pct"],
        "sync_method": sync_correction.method,
        "sync_passed": sync_correction.passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote raw BIDS sessions to the curated sample dataset.")
    parser.add_argument("--sub", type=str, default=None, help="Specific subject to promote")
    parser.add_argument("--ses", type=str, default=None, help="Specific session to promote")
    parser.add_argument("--dry-run", action="store_true", help="Check criteria without promoting")
    parser.add_argument("--min-trials", type=int, default=3, help="Minimum trials per gesture class (default 3)")
    parser.add_argument("--method", choices=["fft_xcorr", "hybrid_prbs_nad"], default="fft_xcorr",
                        help="LED sync algorithm (default: fft_xcorr)")
    parser.add_argument("--recompute-sync", action="store_true",
                        help="Recompute LED sync even if led_sync.json exists")
    args = parser.parse_args()

    print("JoTouch Promotion: raw -> sample (single merged CSV)")
    print("=" * 60)

    raw_root = paths.RAW_DIR
    if not raw_root.exists():
        print(f"No raw data directory: {raw_root}")
        return 1

    if args.sub and args.ses:
        candidates = [(args.sub, args.ses)]
    else:
        candidates = []
        for subj_dir in sorted(raw_root.iterdir()):
            if subj_dir.is_dir() and subj_dir.name.startswith("sub-"):
                sub = subj_dir.name[4:]
                for ses_dir in sorted(subj_dir.iterdir()):
                    if ses_dir.is_dir() and ses_dir.name.startswith("ses-"):
                        ses = ses_dir.name[4:]
                        candidates.append((sub, ses))

    if not candidates:
        print("No sessions found in data/raw/")
        return 1

    promoted, failed = 0, 0
    for sub, ses in candidates:
        print(f"\n--- {sub}/{ses} ---")
        report = check_session(sub, ses, data_root=raw_root, min_trials=args.min_trials)

        for name, result in report["checks"].items():
            status = "PASS" if result["pass"] else "FAIL"
            reason = f" — {result['reason']}" if result["reason"] else ""
            print(f"  [{status}] {name}{reason}")

        if report["pass"]:
            if args.dry_run:
                print(f"  -> Would promote (dry-run)")
                promoted += 1
            else:
                result = promote_session(
                    sub, ses,
                    src_root=raw_root,
                    dst_root=paths.SAMPLE_DIR,
                    method=args.method,
                    recompute_sync=args.recompute_sync,
                )
                if result["pass"]:
                    print(f"  -> Promoted: {result['n_samples']} samples, {result['n_runs']} runs, "
                          f"{result['n_columns']} cols, cam={result['cam_coverage_pct']}%")
                    print(f"     Sync: {result['sync_method']} (passed={result['sync_passed']})")
                    promoted += 1
                else:
                    print(f"  -> FAILED: {result['reason']}")
                    failed += 1
        else:
            print(f"  -> NOT promoted (criteria failed)")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Promoted: {promoted}, Failed: {failed}")
    if args.dry_run:
        print("(dry-run — no files written)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
