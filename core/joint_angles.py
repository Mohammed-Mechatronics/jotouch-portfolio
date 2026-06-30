"""Convert MediaPipe Hand landmarks (21 × 3D) to 15 finger joint angles.

Each joint angle is computed using the 3D vector dot product method:

    angle = arccos( (BA · BC) / (|BA| × |BC|) )

where B is the vertex (the joint itself), and A, C are the adjacent
landmarks on either side of the joint.

The result is the **interior angle** at the joint in degrees (0-180).
A fully extended finger yields ~180° at each joint; a fully flexed
finger yields ~30-60°.

To convert to **flexion angle** (0° = flat, 90° = fully bent), use
``flexion = 180 - interior_angle``. This module returns flexion angles
so that 0° = flat hand and higher values = more flexion.

Landmark indices (MediaPipe Hands):
    0  = WRIST
    1  = THUMB_CMC       2  = THUMB_MCP       3  = THUMB_IP        4  = THUMB_TIP
    5  = INDEX_MCP       6  = INDEX_PIP       7  = INDEX_DIP       8  = INDEX_TIP
    9  = MIDDLE_MCP     10  = MIDDLE_PIP     11  = MIDDLE_DIP     12  = MIDDLE_TIP
    13 = RING_MCP       14  = RING_PIP       15  = RING_DIP       16  = RING_TIP
    17 = PINKY_MCP      18  = PINKY_PIP      19  = PINKY_DIP      20 = PINKY_TIP

References:
    - MediaPipe Hands: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
    - ntu-rris/google-mediapipe (195 stars): https://github.com/ntu-rris/google-mediapipe
    - Stack Overflow #76336785: https://stackoverflow.com/questions/76336785
"""

from __future__ import annotations

import math
from typing import Sequence

# ---------------------------------------------------------------------------
# Joint definitions: (name, landmark_a, vertex, landmark_c)
# The angle is computed at the vertex between vectors to A and C.
# ---------------------------------------------------------------------------
JOINT_DEFINITIONS: list[tuple[str, int, int, int]] = [
    # Thumb (3 joints)
    ("target_thumb_cmc_flex",  0, 1, 2),   # WRIST → THUMB_CMC → THUMB_MCP
    ("target_thumb_mcp_flex",  1, 2, 3),   # THUMB_CMC → THUMB_MCP → THUMB_IP
    ("target_thumb_ip_flex",   2, 3, 4),   # THUMB_MCP → THUMB_IP → THUMB_TIP
    # Index (3 joints)
    ("target_index_mcp_flex",  0, 5, 6),   # WRIST → INDEX_MCP → INDEX_PIP
    ("target_index_pip_flex",  5, 6, 7),   # INDEX_MCP → INDEX_PIP → INDEX_DIP
    ("target_index_dip_flex",  6, 7, 8),   # INDEX_PIP → INDEX_DIP → INDEX_TIP
    # Middle (3 joints)
    ("target_middle_mcp_flex", 0, 9, 10),  # WRIST → MIDDLE_MCP → MIDDLE_PIP
    ("target_middle_pip_flex", 9, 10, 11), # MIDDLE_MCP → MIDDLE_PIP → MIDDLE_DIP
    ("target_middle_dip_flex", 10, 11, 12),# MIDDLE_PIP → MIDDLE_DIP → MIDDLE_TIP
    # Ring (3 joints)
    ("target_ring_mcp_flex",   0, 13, 14), # WRIST → RING_MCP → RING_PIP
    ("target_ring_pip_flex",   13, 14, 15),# RING_MCP → RING_PIP → RING_DIP
    ("target_ring_dip_flex",   14, 15, 16),# RING_PIP → RING_DIP → RING_TIP
    # Pinky (3 joints)
    ("target_pinky_mcp_flex",  0, 17, 18), # WRIST → PINKY_MCP → PINKY_PIP
    ("target_pinky_pip_flex",  17, 18, 19),# PINKY_MCP → PINKY_PIP → PINKY_DIP
    ("target_pinky_dip_flex",  18, 19, 20),# PINKY_PIP → PINKY_DIP → PINKY_TIP
]

# Map joint name → (a, vertex, c) for quick lookup
_JOINT_MAP: dict[str, tuple[int, int, int]] = {
    name: (a, b, c) for name, a, b, c in JOINT_DEFINITIONS
}

# Ordered list of joint names matching schema.TARGET_COLUMNS
JOINT_NAMES: list[str] = [name for name, _, _, _ in JOINT_DEFINITIONS]


def _angle_3d(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
) -> float:
    """Compute the interior angle at vertex ``b`` between points ``a`` and ``c``.

    Uses the 3D dot product method:
        cos(θ) = (BA · BC) / (|BA| × |BC|)

    Args:
        a: 3D coordinates [x, y, z] of point A.
        b: 3D coordinates [x, y, z] of vertex B.
        c: 3D coordinates [x, y, z] of point C.

    Returns:
        Interior angle in degrees (0-180).
    """
    # Vectors from vertex B to A and C
    ba = [a[i] - b[i] for i in range(3)]
    bc = [c[i] - b[i] for i in range(3)]

    # Dot product and magnitudes
    dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)

    if mag_ba < 1e-9 or mag_bc < 1e-9:
        return 180.0  # Degenerate — treat as fully extended

    cos_angle = dot / (mag_ba * mag_bc)
    # Clamp to [-1, 1] to avoid NaN from floating-point errors
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return math.degrees(math.acos(cos_angle))


def landmarks_to_joint_angles(
    landmarks: Sequence[float] | Sequence[Sequence[float]],
) -> dict[str, float]:
    """Convert 21 MediaPipe hand landmarks to 15 joint flexion angles.

    Args:
        landmarks: Either:
            - A flat list of 63 floats (21 landmarks × x, y, z), as
              produced by ``CameraReader.get_frame()["landmarks"]``.
            - A list of 21 [x, y, z] tuples/lists.
            - A list of 21 objects with ``.x``, ``.y``, ``.z`` attributes
              (raw MediaPipe landmark objects).

    Returns:
        Dict mapping joint names (matching ``core.schema.TARGET_COLUMNS``)
        to flexion angles in degrees.

        Flexion angle = 180 - interior_angle, so:
        - Flat hand → ~0° (fully extended)
        - Closed fist → ~90-150° (fully flexed)

        Values are clamped to [0, 180].
    """
    # Normalize input to list of (x, y, z) tuples
    points = _normalize_landmarks(landmarks)

    if len(points) < 21:
        # Not enough landmarks — return all zeros
        return {name: 0.0 for name in JOINT_NAMES}

    result: dict[str, float] = {}
    for name, a_idx, vertex_idx, c_idx in JOINT_DEFINITIONS:
        interior = _angle_3d(points[a_idx], points[vertex_idx], points[c_idx])
        # Convert interior angle to flexion angle
        # Interior ~180° = flat → flexion = 0°
        # Interior ~30° = fully bent → flexion = 150°
        flexion = 180.0 - interior
        # Clamp to [0, 180]
        flexion = max(0.0, min(180.0, flexion))
        result[name] = round(flexion, 2)

    return result


def landmarks_to_joint_angles_list(
    landmarks: Sequence[float] | Sequence[Sequence[float]],
) -> list[float]:
    """Same as ``landmarks_to_joint_angles`` but returns an ordered list.

    The order matches ``core.schema.TARGET_COLUMNS`` (15 values).
    """
    angles = landmarks_to_joint_angles(landmarks)
    return [angles[name] for name in JOINT_NAMES]


def _normalize_landmarks(
    landmarks: Sequence[float] | Sequence[Sequence[float]] | Sequence[object],
) -> list[tuple[float, float, float]]:
    """Normalize various landmark input formats to a list of (x, y, z) tuples.

    Handles:
    - Flat list of 63 floats (21 × 3)
    - List of 21 [x, y, z] lists/tuples
    - List of 21 objects with .x, .y, .z attributes (MediaPipe Landmark)
    """
    if not landmarks:
        return []

    # Check first element type
    first = landmarks[0]

    if isinstance(first, (int, float)):
        # Flat list of 63 floats — group into triples
        flat = list(landmarks)
        if len(flat) < 63:
            return []
        return [(flat[i], flat[i + 1], flat[i + 2]) for i in range(0, 63, 3)]

    if hasattr(first, "x") and hasattr(first, "y"):
        # MediaPipe Landmark objects
        return [(lm.x, lm.y, lm.z) for lm in landmarks[:21]]

    if isinstance(first, (list, tuple)) and len(first) >= 3:
        # List of [x, y, z] tuples
        return [(lm[0], lm[1], lm[2]) for lm in landmarks[:21]]

    return []
