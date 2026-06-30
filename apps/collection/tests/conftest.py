"""Conftest for apps.collection tests — add repo root to sys.path."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _reset_manager():
    """Reset the global SessionManager between tests to prevent state leakage."""
    from apps.collection.api import _manager
    _manager.reset()
    yield
    _manager.reset()


@pytest.fixture()
def fast_sleep():
    """Patch time.sleep in precollect so sampling loops complete instantly.

    Use this fixture in test_precollect.py tests only — NOT in session tests
    where the sleep-paced MockSensorReader must run at its intended 100 Hz.
    """
    with patch("apps.collection.precollect.time.sleep"):
        yield
