"""Single data root resolver for the JoTouch portfolio.

All apps and scripts resolve data paths through this module so there is
one place to change if the data layout moves.
"""

from __future__ import annotations

from pathlib import Path

# Repository root = parent of the core/ package
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent

# Top-level data directory
DATA_ROOT = REPO_ROOT / "data"

# Subdirectories
SAMPLE_DIR = DATA_ROOT / "sample"   # tracked in git — curated demo subset
RAW_DIR = DATA_ROOT / "raw"         # gitignored — full collection writes here
DERIVED_DIR = DATA_ROOT / "derived"  # gitignored — ML-ready Parquet files
SPLITS_DIR = DATA_ROOT / "splits"    # tracked in git — split spec YAML files
PROCESSED_DIR = DATA_ROOT / "processed"  # gitignored — legacy (deprecated)
RESULTS_DIR = DATA_ROOT / "results"      # gitignored — validation reports

# Legacy archive
ARCHIVE_DIR = REPO_ROOT / "_archive" / "sample_data_legacy"


def sample_dir() -> Path:
    """Return the curated sample data directory (tracked in git)."""
    return SAMPLE_DIR


def raw_dir() -> Path:
    """Return the raw data directory (gitignored, written by collection app)."""
    return RAW_DIR


def processed_dir() -> Path:
    """Return the processed data directory (legacy, deprecated — use derived_dir)."""
    return PROCESSED_DIR


def derived_dir() -> Path:
    """Return the derived data directory (ML-ready Parquet files)."""
    return DERIVED_DIR


def splits_dir() -> Path:
    """Return the splits directory (split spec YAML files, git-tracked)."""
    return SPLITS_DIR


def results_dir() -> Path:
    """Return the results directory (validation reports)."""
    return RESULTS_DIR


def subject_dir(sub: str, *, data_root: Path | None = None) -> Path:
    """Return the subject directory, e.g. ``data/sample/sub-P01``.

    Parameters
    ----------
    sub : str
        Subject label without the ``sub-`` prefix (e.g. ``"P01"``).
    data_root : Path, optional
        Override the data root (defaults to ``data/sample/``).
    """
    root = data_root or SAMPLE_DIR
    return root / f"sub-{sub}"


def session_dir(sub: str, ses: str, *, data_root: Path | None = None) -> Path:
    """Return the session directory, e.g. ``data/sample/sub-P01/ses-S01``.

    Parameters
    ----------
    sub : str
        Subject label without prefix (e.g. ``"P01"``).
    ses : str
        Session label without prefix (e.g. ``"S01"``).
    data_root : Path, optional
        Override the data root.
    """
    return subject_dir(sub, data_root=data_root) / f"ses-{ses}"
