"""BIDS filename parsing and generation.

BIDS naming convention:
    sub-{label}_ses-{label}_task-{label}_run-{index}_{suffix}.csv

Examples:
    sub-P01_ses-S01_task-thumbCmcIso_run-01_physio.csv
    sub-P01_ses-S01_task-powerGrip_run-01_camera.csv
    sub-P01_ses-S01_task-mvc_run-00_targets.csv

Suffixes:
    physio  — FSR signals
    camera  — MediaPipe hand landmarks
    targets — derived joint angles (15 DOFs)
    events  — classification event labels (post-hoc generated)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Suffixes
SUFFIX_PHYSIO = "physio"
SUFFIX_CAMERA = "camera"
SUFFIX_TARGETS = "targets"
SUFFIX_EVENTS = "events"

VALID_SUFFIXES = {SUFFIX_PHYSIO, SUFFIX_CAMERA, SUFFIX_TARGETS, SUFFIX_EVENTS}

# Allowed label kinds for validate_label()
_VALID_LABEL_KINDS = {"sub", "ses", "task"}
# BIDS entity labels are alphanumeric only — no spaces, hyphens, or
# underscores (those collide with the BIDS filename separators "-" and "_").
_LABEL_RE = re.compile(r"^[A-Za-z0-9]+$")


def validate_label(label: str, kind: str) -> str:
    """Validate a BIDS entity label (sub / ses / task).

    BIDS labels must be alphanumeric (``[A-Za-z0-9]+``). Spaces, hyphens, and
    underscores are rejected because they collide with the BIDS filename
    separators (``-`` and ``_``) and would make ``parse_filename`` fail to
    round-trip the generated filename.

    Parameters
    ----------
    label : str
        The label value (e.g. ``"P01"``, ``"powerGrip"``).
    kind : str
        One of ``"sub"``, ``"ses"``, ``"task"``. Used only for the error
        message; the validation rules are identical for all kinds.

    Returns
    -------
    str
        The validated label (unchanged).

    Raises
    ------
    ValueError
        If the label is empty, contains non-alphanumeric characters, or
        ``kind`` is not one of the allowed kinds.
    """
    if kind not in _VALID_LABEL_KINDS:
        raise ValueError(
            f"Invalid label kind '{kind}'. Must be one of {sorted(_VALID_LABEL_KINDS)}"
        )
    if not isinstance(label, str) or not label:
        raise ValueError(f"Empty {kind} label")
    if not _LABEL_RE.match(label):
        raise ValueError(
            f"Invalid {kind} label '{label}'. BIDS labels must be alphanumeric "
            f"([A-Za-z0-9]+) — no spaces, hyphens, or underscores."
        )
    return label

# BIDS filename regex
# sub-P01_ses-S01_task-thumbCmcIso_run-01_physio.csv
BIDS_RE = re.compile(
    r"^sub-(?P<sub>[A-Za-z0-9]+)"
    r"_ses-(?P<ses>[A-Za-z0-9]+)"
    r"_task-(?P<task>[A-Za-z0-9]+)"
    r"_run-(?P<run>\d+)"
    r"_(?P<suffix>[A-Za-z0-9]+)"
    r"\.(?P<ext>csv|tsv)$"
)

# For metadata files (no task/run): sub-P01_ses-S01_physio.json
META_RE = re.compile(
    r"^sub-(?P<sub>[A-Za-z0-9]+)"
    r"_ses-(?P<ses>[A-Za-z0-9]+)"
    r"_(?P<suffix>[A-Za-z0-9]+)"
    r"\.(?P<ext>json|tsv)$"
)


@dataclass(frozen=True)
class BIDSRunName:
    """Parsed BIDS run filename."""

    sub: str        # without "sub-" prefix, e.g. "P01"
    ses: str        # without "ses-" prefix, e.g. "S01"
    task: str       # camelCase, e.g. "thumbCmcIso"
    run: int        # 0-99 (0 = MVC baseline)
    suffix: str     # "physio", "camera", "targets", "events"
    extension: str  # "csv" or "tsv"

    @property
    def filename(self) -> str:
        return build_filename(self.sub, self.ses, self.task, self.run, self.suffix, self.extension)

    @property
    def is_baseline(self) -> bool:
        return self.run == 0


def build_filename(
    sub: str,
    ses: str,
    task: str,
    run: int,
    suffix: str,
    extension: str = "csv",
) -> str:
    """Build a BIDS-compliant filename.

    >>> build_filename("P01", "S01", "thumbCmcIso", 1, "physio")
    'sub-P01_ses-S01_task-thumbCmcIso_run-01_physio.csv'
    """
    if suffix not in VALID_SUFFIXES:
        raise ValueError(f"Invalid suffix '{suffix}'. Must be one of {VALID_SUFFIXES}")
    return f"sub-{sub}_ses-{ses}_task-{task}_run-{run:02d}_{suffix}.{extension}"


def parse_filename(path: str | Path) -> BIDSRunName | None:
    """Parse a BIDS run filename.

    Returns ``None`` if the filename does not match the BIDS run pattern.
    Metadata files (without task/run) are not matched here.
    """
    name = path.name if isinstance(path, Path) else str(path)
    m = BIDS_RE.match(name)
    if not m:
        return None
    return BIDSRunName(
        sub=m.group("sub"),
        ses=m.group("ses"),
        task=m.group("task"),
        run=int(m.group("run")),
        suffix=m.group("suffix"),
        extension=m.group("ext"),
    )


def is_physio_file(path: str | Path) -> bool:
    """Check if a file is a physio (FSR) CSV."""
    parsed = parse_filename(path)
    return parsed is not None and parsed.suffix == SUFFIX_PHYSIO


def is_camera_file(path: str | Path) -> bool:
    """Check if a file is a camera (MediaPipe landmarks) CSV."""
    parsed = parse_filename(path)
    return parsed is not None and parsed.suffix == SUFFIX_CAMERA


def is_targets_file(path: str | Path) -> bool:
    """Check if a file is a targets (joint angles) CSV."""
    parsed = parse_filename(path)
    return parsed is not None and parsed.suffix == SUFFIX_TARGETS


def is_events_file(path: str | Path) -> bool:
    """Check if a file is an events (classification labels) TSV."""
    parsed = parse_filename(path)
    return parsed is not None and parsed.suffix == SUFFIX_EVENTS


def list_runs_in_session(session_path: Path) -> list[BIDSRunName]:
    """List all BIDS run files in a session directory.

    Returns a sorted list of parsed filenames. Each run appears once
    (the first suffix found alphabetically).
    """
    if not session_path.exists():
        return []
    seen: set[tuple[str, int]] = set()
    runs: list[BIDSRunName] = []
    for path in sorted(session_path.iterdir()):
        parsed = parse_filename(path)
        if parsed is None:
            continue
        key = (parsed.task, parsed.run)
        if key in seen:
            continue
        seen.add(key)
        runs.append(parsed)
    return runs


def run_files_for(session_path: Path, task: str, run: int) -> dict[str, Path | None]:
    """Return the physio/camera/targets file paths for a given task+run.

    Returns a dict with keys 'physio', 'camera', 'targets' mapping to
    Path or None if the file does not exist.
    """
    result: dict[str, Path | None] = {}
    for suffix in (SUFFIX_PHYSIO, SUFFIX_CAMERA, SUFFIX_TARGETS):
        fname = build_filename(
            sub=_extract_sub_from_path(session_path),
            ses=_extract_ses_from_path(session_path),
            task=task,
            run=run,
            suffix=suffix,
        )
        path = session_path / fname
        result[suffix] = path if path.exists() else None
    return result


def _extract_sub_from_path(session_path: Path) -> str:
    """Extract subject label from a session directory path like .../sub-P01/ses-S01."""
    for part in session_path.parts:
        if part.startswith("sub-"):
            return part[4:]
    return "unknown"


def _extract_ses_from_path(session_path: Path) -> str:
    """Extract session label from a session directory path like .../sub-P01/ses-S01."""
    for part in session_path.parts:
        if part.startswith("ses-"):
            return part[4:]
    return "unknown"


def next_run_number(
    sub: str,
    ses: str,
    task: str,
    *,
    data_root: Path | None = None,
) -> int:
    """Return the next free run number for ``(sub, ses, task)``.

    Scans the session directory for existing BIDS run files of the given task
    and returns ``max(existing runs) + 1``. If no runs exist, returns ``0``
    (the baseline/MVC run number).

    Used by the collection UI so operators never type a run number and re-runs
    never silently overwrite existing data (the writer opens with mode ``x``
    when ``run`` is explicitly provided, and auto-increments when ``run=None``).

    Parameters
    ----------
    sub, ses, task : str
        Subject, session, and task labels.
    data_root : Path | None
        Data root (defaults to ``core.paths.RAW_DIR``).

    Returns
    -------
    int
        The next free run number (0 if none exist).
    """
    from core import paths

    root = data_root if data_root is not None else paths.RAW_DIR
    sdir = paths.session_dir(sub, ses, data_root=root)
    if not sdir.exists():
        return 0
    max_run = -1
    for path in sdir.iterdir():
        parsed = parse_filename(path)
        if parsed is None or parsed.task != task:
            continue
        if parsed.run > max_run:
            max_run = parsed.run
    return max_run + 1


# ── Subject / session discovery (for UI auto-increment) ───────────────────────

import re as _re

_SUB_LABEL_RE = _re.compile(r"^sub-(.+)$")
_SES_LABEL_RE = _re.compile(r"^ses-(.+)$")
_P_NUM_RE = _re.compile(r"^P(\d+)$")
_S_NUM_RE = _re.compile(r"^S(\d+)$")


def list_subjects(*, data_root: Path | None = None) -> list[str]:
    """List existing subject labels (without ``sub-`` prefix) in the data root.

    Scans ``data_root/`` for directories named ``sub-{label}`` and returns the
    labels sorted alphabetically. Files and non-``sub-`` directories are
    ignored.

    Used by the collection UI to populate the "Existing Subject" dropdown.

    Parameters
    ----------
    data_root : Path | None
        Data root (defaults to ``core.paths.RAW_DIR``).

    Returns
    -------
    list[str]
        Sorted list of subject labels (e.g. ``["P01", "P02"]``).
    """
    from core import paths

    root = data_root if data_root is not None else paths.RAW_DIR
    if not root.exists():
        return []
    subjects: list[str] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        m = _SUB_LABEL_RE.match(path.name)
        if m:
            subjects.append(m.group(1))
    return subjects


def next_subject_label(*, data_root: Path | None = None) -> str:
    """Return the next free subject label following the ``P{NN}`` convention.

    Scans existing ``sub-P{NN}`` directories and returns ``P{max+1:02d}``.
    If no matching subjects exist, returns ``"P01"``. Subject labels that
    don't match the ``P{NN}`` pattern are ignored for the increment
    calculation but don't cause an error.

    Used by the collection UI so operators never type a subject ID and
    never accidentally reuse an existing subject.

    Parameters
    ----------
    data_root : Path | None
        Data root (defaults to ``core.paths.RAW_DIR``).

    Returns
    -------
    str
        The next free subject label (e.g. ``"P02"``).
    """
    subjects = list_subjects(data_root=data_root)
    max_num = 0
    for label in subjects:
        m = _P_NUM_RE.match(label)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return f"P{max_num + 1:02d}"


def next_session_label(sub: str, *, data_root: Path | None = None) -> str:
    """Return the next free session label for a subject following ``S{NN}``.

    Scans ``data_root/sub-{sub}/`` for ``ses-S{NN}`` directories and returns
    ``S{max+1:02d}``. If the subject directory doesn't exist or has no
    matching sessions, returns ``"S01"``. Session labels that don't match
    the ``S{NN}`` pattern are ignored for the increment calculation.

    Used by the collection UI so operators never type a session ID and
    never accidentally reuse an existing session (which would cause
    run-number drift on re-runs).

    Parameters
    ----------
    sub : str
        Subject label (without ``sub-`` prefix).
    data_root : Path | None
        Data root (defaults to ``core.paths.RAW_DIR``).

    Returns
    -------
    str
        The next free session label (e.g. ``"S02"``).
    """
    from core import paths

    root = data_root if data_root is not None else paths.RAW_DIR
    sdir = root / f"sub-{sub}"
    if not sdir.exists():
        return "S01"
    max_num = 0
    for path in sdir.iterdir():
        if not path.is_dir():
            continue
        m = _SES_LABEL_RE.match(path.name)
        if m:
            inner = _S_NUM_RE.match(m.group(1))
            if inner:
                max_num = max(max_num, int(inner.group(1)))
    return f"S{max_num + 1:02d}"
