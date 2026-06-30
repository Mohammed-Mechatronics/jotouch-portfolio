"""JoTouch core — shared infrastructure for all apps and scripts.

Single source of truth for:
  - paths.py    : data root resolution
  - naming.py   : BIDS filename parsing and generation
  - schema.py   : CSV column contracts (physio, camera, targets)
  - metadata.py : BIDS metadata file loaders (participants, sessions, channels, etc.)
  - loader.py   : BIDS data loader (joins physio + camera + targets by timestamp)
  - events.py   : Classification label derivation from task names + angle thresholds
  - features.py : Windowed feature extraction from BIDS physio data
"""
