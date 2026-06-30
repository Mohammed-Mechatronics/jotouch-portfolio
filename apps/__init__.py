"""JoTouch collection application — BIDS writer + 3-phase regression protocol.

This package implements the collection app that produces BIDS-format data:
  - physio.csv  (FSR signals)
  - camera.csv  (MediaPipe hand landmarks)
  - targets.csv (derived 15 joint angles)

The protocol has 3 phases:
  Phase 1: Single-DOF isolation (15 tasks x 3 reps)
  Phase 2: Multi-DOF combinations (9 tasks x 3 reps)
  Phase 3: Freeform (3 x 60s runs)

Plus a mandatory MVC baseline (run-00) and pre-collection tests.
"""
