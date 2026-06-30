"""Visual cues for the collection protocol.

Provides text-based cues for the terminal-based collection UI. Each task has:
  - A human-readable name
  - A description of the movement
  - An instruction for the subject

The cues are displayed during the PREP phase to guide the subject.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskCue:
    """Visual cue for a single task."""

    task: str
    display_name: str
    description: str
    instruction: str


# ── Cue definitions for all 26 tasks ──────────────────────────────────────────

CUES: dict[str, TaskCue] = {
    # Baseline
    "mvc": TaskCue(
        task="mvc",
        display_name="MVC Baseline",
        description="Maximum voluntary contraction",
        instruction="Squeeze your fist as hard as you can and hold for 5 seconds.",
    ),

    # Phase 1: Single-DOF isolation (15 tasks)
    "thumbCmcIso": TaskCue(
        task="thumbCmcIso",
        display_name="Thumb CMC Flexion",
        description="Isolate thumb carpometacarpal joint",
        instruction="Move ONLY your thumb CMC joint (base of thumb) slowly flex and extend.",
    ),
    "thumbMcpIso": TaskCue(
        task="thumbMcpIso",
        display_name="Thumb MCP Flexion",
        description="Isolate thumb metacarpophalangeal joint",
        instruction="Move ONLY your thumb MCP joint (knuckle) slowly flex and extend.",
    ),
    "thumbIpIso": TaskCue(
        task="thumbIpIso",
        display_name="Thumb IP Flexion",
        description="Isolate thumb interphalangeal joint",
        instruction="Move ONLY your thumb IP joint (tip) slowly flex and extend.",
    ),
    "indexMcpIso": TaskCue(
        task="indexMcpIso",
        display_name="Index MCP Flexion",
        description="Isolate index metacarpophalangeal joint",
        instruction="Move ONLY your index finger knuckle, keep other fingers still.",
    ),
    "indexPipIso": TaskCue(
        task="indexPipIso",
        display_name="Index PIP Flexion",
        description="Isolate index proximal interphalangeal joint",
        instruction="Move ONLY your index finger middle joint, keep knuckle straight.",
    ),
    "indexDipIso": TaskCue(
        task="indexDipIso",
        display_name="Index DIP Flexion",
        description="Isolate index distal interphalangeal joint",
        instruction="Move ONLY your index fingertip joint, keep other joints still.",
    ),
    "middleMcpIso": TaskCue(
        task="middleMcpIso",
        display_name="Middle MCP Flexion",
        description="Isolate middle metacarpophalangeal joint",
        instruction="Move ONLY your middle finger knuckle, keep other fingers still.",
    ),
    "middlePipIso": TaskCue(
        task="middlePipIso",
        display_name="Middle PIP Flexion",
        description="Isolate middle proximal interphalangeal joint",
        instruction="Move ONLY your middle finger middle joint.",
    ),
    "middleDipIso": TaskCue(
        task="middleDipIso",
        display_name="Middle DIP Flexion",
        description="Isolate middle distal interphalangeal joint",
        instruction="Move ONLY your middle fingertip joint.",
    ),
    "ringMcpIso": TaskCue(
        task="ringMcpIso",
        display_name="Ring MCP Flexion",
        description="Isolate ring metacarpophalangeal joint",
        instruction="Move ONLY your ring finger knuckle, keep other fingers still.",
    ),
    "ringPipIso": TaskCue(
        task="ringPipIso",
        display_name="Ring PIP Flexion",
        description="Isolate ring proximal interphalangeal joint",
        instruction="Move ONLY your ring finger middle joint.",
    ),
    "ringDipIso": TaskCue(
        task="ringDipIso",
        display_name="Ring DIP Flexion",
        description="Isolate ring distal interphalangeal joint",
        instruction="Move ONLY your ring fingertip joint.",
    ),
    "pinkyMcpIso": TaskCue(
        task="pinkyMcpIso",
        display_name="Pinky MCP Flexion",
        description="Isolate pinky metacarpophalangeal joint",
        instruction="Move ONLY your pinky knuckle, keep other fingers still.",
    ),
    "pinkyPipIso": TaskCue(
        task="pinkyPipIso",
        display_name="Pinky PIP Flexion",
        description="Isolate pinky proximal interphalangeal joint",
        instruction="Move ONLY your pinky middle joint.",
    ),
    "pinkyDipIso": TaskCue(
        task="pinkyDipIso",
        display_name="Pinky DIP Flexion",
        description="Isolate pinky distal interphalangeal joint",
        instruction="Move ONLY your pinky fingertip joint.",
    ),

    # Phase 2: Multi-DOF combinations (9 tasks)
    "powerGrip": TaskCue(
        task="powerGrip",
        display_name="Power Grip",
        description="Full-hand cylindrical grip",
        instruction="Make a strong fist, wrapping all fingers around an imaginary cylinder.",
    ),
    "tripodGrip": TaskCue(
        task="tripodGrip",
        display_name="Tripod Grip",
        description="Three-finger precision grip",
        instruction="Pinch thumb, index, and middle fingertips together as if holding a pen.",
    ),
    "tipPinch": TaskCue(
        task="tipPinch",
        display_name="Tip Pinch",
        description="Thumb-index fingertip pinch",
        instruction="Pinch your thumb and index fingertips together lightly.",
    ),
    "lateralGrip": TaskCue(
        task="lateralGrip",
        display_name="Lateral Grip",
        description="Thumb-to-side-of-index grip",
        instruction="Press your thumb against the side of your index finger (key grip).",
    ),
    "sphericalGrip": TaskCue(
        task="sphericalGrip",
        display_name="Spherical Grip",
        description="Full-hand spherical grip",
        instruction="Cup your hand as if holding a tennis ball.",
    ),
    "extensionGrip": TaskCue(
        task="extensionGrip",
        display_name="Extension Grip",
        description="Finger extension spread",
        instruction="Extend all fingers wide and spread them apart.",
    ),
    "handOpen": TaskCue(
        task="handOpen",
        display_name="Hand Open",
        description="Full hand extension",
        instruction="Open your hand fully, extending all fingers straight.",
    ),
    "fingerSpread": TaskCue(
        task="fingerSpread",
        display_name="Finger Spread",
        description="Abduction spread",
        instruction="Spread all fingers as wide apart as possible, keeping them extended.",
    ),
    "counting": TaskCue(
        task="counting",
        display_name="Finger Counting",
        description="Sequential finger counting (1-5)",
        instruction="Count from 1 to 5: extend one finger at a time, then close.",
    ),

    # Phase 3: Freeform
    "freeform": TaskCue(
        task="freeform",
        display_name="Freeform Movement",
        description="Natural unconstrained hand movement",
        instruction="Move your hand naturally and freely for 60 seconds. Try varied gestures.",
    ),
}


def get_cue(task: str) -> TaskCue:
    """Get the cue for a task. Returns a default cue if task not found."""
    if task in CUES:
        return CUES[task]
    return TaskCue(
        task=task,
        display_name=task,
        description="Unknown task",
        instruction=f"Perform the {task} movement.",
    )


def format_cue_for_display(cue: TaskCue, run_num: int, rep: int, total_reps: int) -> str:
    """Format a cue for terminal display during the PREP phase."""
    lines = [
        f"  Task: {cue.display_name}",
        f"  Run:  {run_num:02d}  (Rep {rep}/{total_reps})",
        f"  Desc: {cue.description}",
        f"  >>>  {cue.instruction}",
    ]
    return "\n".join(lines)


def format_countdown(seconds_left: float) -> str:
    """Format a countdown timer for display."""
    if seconds_left <= 0:
        return "GO!"
    return f"Starting in {seconds_left:.0f}s..."
