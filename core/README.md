# Core

Shared infrastructure for all JoTouch apps and scripts.

No app imports from another app — they all import from here.

## Modules

| Module | Purpose |
|--------|---------|
| `paths.py` | Resolve data root (`data/sample/`, `data/raw/`, etc.) |
| `naming.py` | Parse and generate BIDS filenames (`sub-P01_ses-S01_task-..._run-01_physio.csv`) |
| `schema.py` | Column contracts for `_physio.csv`, `_camera.csv`, `_targets.csv` |
| `metadata.py` | Load `participants.tsv`, `sessions.tsv`, `physio.json`, `channels.tsv`, `led_sync.json`, `precollect.json` |
| `loader.py` | Load BIDS sessions and runs (joins 3 modalities by timestamp) |
| `events.py` | Derive classification labels from task names + joint angle thresholds |
| `features.py` | Windowed feature extraction from physio data |

## Import rules

```
scripts/ → core/ + apps/
apps/    → core/
core/    → stdlib + third-party only
```

No app imports from another app. No script imports from another script.
