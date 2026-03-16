# Eval Tracking

Eval logs live under `eval_logs/` and are separate from `run_logs/`.

## Layout

- `eval_logs/DINO_WM/`
- `eval_logs/WAN_WM/`

Each directory has:
- `timeline.md` (chronological index)
- one file per evaluation: `<date>_<task>_job-<jobid>.md`

For local/manual evals, use `job-local`.

## Required Sections Per Eval

- `## Provenance`
  - checkpoint
  - source run log
  - config snapshot
  - dataset
- `## Job`
  - ids/timestamps/runtime/node
- `## Metrics`
- `## Qualitative`
- `## Verdict`
- `## Next`
