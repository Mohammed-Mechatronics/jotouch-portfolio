"""Tests for core/joint_angles.py — MediaPipe landmarks → 15 joint angles."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.joint_angles import (
    JOINT_DEFINITIONS,
    JOINT_NAMES,
    landmarks_to_joint_angles,
    landmarks_to_joint_angles_list,
    _angle_3d,
    _normalize_landmarks,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _flat_hand_landmarks() -> list[float]:
    """Generate 63 floats representing a flat, extended hand.

    All fingers straight, pointing in +x direction from wrist.
    Each finger is a straight line, so all joint angles should be ~0°.
    """
    lms = []
    # Wrist at origin
    lms += [0.0, 0.0, 0.0]
    # Thumb: straight line along +x, slightly below
    lms += [0.05, -0.02, 0.0]  # CMC
    lms += [0.10, -0.03, 0.0]  # MCP
    lms += [0.15, -0.04, 0.0]  # IP
    lms += [0.20, -0.05, 0.0]  # Tip
    # Index: straight line along +x
    lms += [0.10, 0.02, 0.0]   # MCP
    lms += [0.15, 0.02, 0.0]   # PIP
    lms += [0.20, 0.02, 0.0]   # DIP
    lms += [0.25, 0.02, 0.0]   # Tip
    # Middle: straight line along +x, slightly higher
    lms += [0.10, 0.05, 0.0]   # MCP
    lms += [0.16, 0.05, 0.0]   # PIP
    lms += [0.22, 0.05, 0.0]   # DIP
    lms += [0.28, 0.05, 0.0]   # Tip
    # Ring: straight line along +x
    lms += [0.10, 0.08, 0.0]   # MCP
    lms += [0.15, 0.08, 0.0]   # PIP
    lms += [0.20, 0.08, 0.0]   # DIP
    lms += [0.25, 0.08, 0.0]   # Tip
    # Pinky: straight line along +x
    lms += [0.10, 0.11, 0.0]   # MCP
    lms += [0.14, 0.11, 0.0]   # PIP
    lms += [0.18, 0.11, 0.0]   # DIP
    lms += [0.22, 0.11, 0.0]   # Tip
    return lms


def _clenched_fist_landmarks() -> list[float]:
    """Generate 63 floats representing a clenched fist.

    Fingers are curled back toward the wrist, creating sharp angles
    at MCP, PIP, and DIP joints. Joint angles should be high (>>0°).
    """
    lms = []
    # Wrist at origin
    lms += [0.0, 0.0, 0.0]
    # Thumb: curled inward
    lms += [0.05, -0.02, 0.0]  # CMC
    lms += [0.08, -0.05, 0.0]  # MCP
    lms += [0.05, -0.08, 0.0]  # IP (bent back)
    lms += [0.02, -0.10, 0.0]  # Tip
    # Index: MCP forward, PIP/DIP curled back down
    lms += [0.10, 0.02, 0.0]   # MCP
    lms += [0.12, -0.02, 0.0]  # PIP (bent)
    lms += [0.08, -0.05, 0.0]  # DIP (bent more)
    lms += [0.04, -0.07, 0.0]  # Tip
    # Middle: same curl pattern
    lms += [0.10, 0.05, 0.0]   # MCP
    lms += [0.13, 0.01, 0.0]   # PIP
    lms += [0.09, -0.03, 0.0]  # DIP
    lms += [0.05, -0.06, 0.0]  # Tip
    # Ring
    lms += [0.10, 0.08, 0.0]   # MCP
    lms += [0.12, 0.04, 0.0]   # PIP
    lms += [0.08, 0.00, 0.0]   # DIP
    lms += [0.04, -0.03, 0.0]  # Tip
    # Pinky
    lms += [0.10, 0.11, 0.0]   # MCP
    lms += [0.11, 0.08, 0.0]   # PIP
    lms += [0.08, 0.05, 0.0]   # DIP
    lms += [0.05, 0.02, 0.0]   # Tip
    return lms


# ---------------------------------------------------------------------------
# _angle_3d tests
# ---------------------------------------------------------------------------

class TestAngle3D:
    def test_straight_line_180(self):
        """Three collinear points → 180° interior angle."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        c = [2.0, 0.0, 0.0]
        angle = _angle_3d(a, b, c)
        assert angle == pytest.approx(180.0, abs=0.5)

    def test_right_angle_90(self):
        """Perpendicular vectors → 90° interior angle."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        c = [0.0, 1.0, 0.0]
        angle = _angle_3d(a, b, c)
        assert angle == pytest.approx(90.0, abs=0.5)

    def test_zero_vector_returns_180(self):
        """Degenerate (zero-length vector) → 180° (treated as extended)."""
        a = [0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        c = [1.0, 0.0, 0.0]
        angle = _angle_3d(a, b, c)
        assert angle == 180.0

    def test_3d_angle(self):
        """Angle in 3D space (not just 2D)."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        c = [0.0, 0.0, 1.0]
        angle = _angle_3d(a, b, c)
        assert angle == pytest.approx(90.0, abs=0.5)


# ---------------------------------------------------------------------------
# _normalize_landmarks tests
# ---------------------------------------------------------------------------

class TestNormalizeLandmarks:
    def test_flat_list_63(self):
        """Flat list of 63 floats → 21 (x,y,z) tuples."""
        flat = [float(i) for i in range(63)]
        points = _normalize_landmarks(flat)
        assert len(points) == 21
        assert points[0] == (0.0, 1.0, 2.0)
        assert points[20] == (60.0, 61.0, 62.0)

    def test_list_of_tuples(self):
        """List of 21 [x,y,z] tuples → 21 (x,y,z) tuples."""
        input_data = [[float(i), float(i + 1), float(i + 2)] for i in range(0, 63, 3)]
        points = _normalize_landmarks(input_data)
        assert len(points) == 21
        assert points[0] == (0.0, 1.0, 2.0)

    def test_empty_returns_empty(self):
        assert _normalize_landmarks([]) == []

    def test_short_flat_list_returns_empty(self):
        """Flat list with < 63 floats → empty (not enough data)."""
        assert _normalize_landmarks([1.0, 2.0, 3.0]) == []


# ---------------------------------------------------------------------------
# landmarks_to_joint_angles tests
# ---------------------------------------------------------------------------

class TestLandmarksToJointAngles:
    def test_returns_15_joints(self):
        """Output has exactly 15 joint angles."""
        lms = _flat_hand_landmarks()
        angles = landmarks_to_joint_angles(lms)
        assert len(angles) == 15

    def test_joint_names_match_schema(self):
        """Joint names match core.schema.TARGET_COLUMNS."""
        from core.schema import TARGET_COLUMNS
        lms = _flat_hand_landmarks()
        angles = landmarks_to_joint_angles(lms)
        for name in TARGET_COLUMNS:
            assert name in angles, f"Missing joint: {name}"

    def test_flat_hand_low_flexion(self):
        """Flat hand → low flexion angles (close to 0°).

        PIP and DIP joints should be ~0° (fingers are straight).
        MCP joints may be higher because fingers fan out from the wrist
        at different angles — this is anatomically correct.
        """
        lms = _flat_hand_landmarks()
        angles = landmarks_to_joint_angles(lms)
        # PIP and DIP joints (finger straight) should be < 10°
        for name, value in angles.items():
            if "pip" in name or "dip" in name:
                assert value < 10.0, f"{name} = {value}°, expected < 10° for flat hand"
        # MCP joints can be higher due to finger fan-out from wrist
        for name, value in angles.items():
            if "mcp" in name:
                assert value < 60.0, f"{name} = {value}°, expected < 60° for flat hand"

    def test_clenched_fist_high_flexion(self):
        """Clenched fist → higher flexion angles than flat hand."""
        flat_angles = landmarks_to_joint_angles(_flat_hand_landmarks())
        fist_angles = landmarks_to_joint_angles(_clenched_fist_landmarks())
        # At least 10 of 15 joints should be more flexed in fist
        more_flexed = sum(
            1 for name in flat_angles
            if fist_angles[name] > flat_angles[name] + 5.0
        )
        assert more_flexed >= 10, (
            f"Only {more_flexed}/15 joints more flexed in fist vs flat"
        )

    def test_values_in_range(self):
        """All angles should be in [0, 180]."""
        import random
        rng = random.Random(42)
        for _ in range(20):
            lms = [rng.uniform(-0.5, 0.5) for _ in range(63)]
            angles = landmarks_to_joint_angles(lms)
            for name, value in angles.items():
                assert 0.0 <= value <= 180.0, f"{name} = {value}, out of [0, 180]"

    def test_empty_landmarks_returns_zeros(self):
        """Empty landmarks → all zeros."""
        angles = landmarks_to_joint_angles([])
        assert len(angles) == 15
        assert all(v == 0.0 for v in angles.values())

    def test_short_landmarks_returns_zeros(self):
        """Landmarks with < 21 points → all zeros."""
        angles = landmarks_to_joint_angles([1.0, 2.0, 3.0])
        assert all(v == 0.0 for v in angles.values())


# ---------------------------------------------------------------------------
# landmarks_to_joint_angles_list tests
# ---------------------------------------------------------------------------

class TestLandmarksToList:
    def test_returns_15_floats(self):
        lms = _flat_hand_landmarks()
        angles_list = landmarks_to_joint_angles_list(lms)
        assert len(angles_list) == 15
        assert all(isinstance(a, float) for a in angles_list)

    def test_order_matches_schema(self):
        """List order matches core.schema.TARGET_COLUMNS."""
        from core.schema import TARGET_COLUMNS
        lms = _flat_hand_landmarks()
        angles_dict = landmarks_to_joint_angles(lms)
        angles_list = landmarks_to_joint_angles_list(lms)
        for i, name in enumerate(TARGET_COLUMNS):
            assert angles_list[i] == angles_dict[name]


# ---------------------------------------------------------------------------
# JOINT_DEFINITIONS tests
# ---------------------------------------------------------------------------

class TestJointDefinitions:
    def test_15_joints_defined(self):
        assert len(JOINT_DEFINITIONS) == 15

    def test_all_indices_valid(self):
        """All landmark indices must be 0-20."""
        for name, a, b, c in JOINT_DEFINITIONS:
            assert 0 <= a <= 20, f"{name}: a={a} out of range"
            assert 0 <= b <= 20, f"{name}: vertex={b} out of range"
            assert 0 <= c <= 20, f"{name}: c={c} out of range"

    def test_vertex_is_middle(self):
        """Vertex (b) must be between a and c (anatomically the joint)."""
        for name, a, b, c in JOINT_DEFINITIONS:
            # Vertex should be different from a and c
            assert b != a, f"{name}: vertex == a"
            assert b != c, f"{name}: vertex == c"

    def test_joint_names_unique(self):
        names = [name for name, _, _, _ in JOINT_DEFINITIONS]
        assert len(names) == len(set(names)), "Duplicate joint names"
