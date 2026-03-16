# Checkpoint Lineage

This file tracks which run logs produced the currently used WM checkpoints.

## DINO_WM

- checkpoint_dir: `/scratch/u6cr/pravsels.u6cr/latent_safety/dino3_wm_checkpoints`
- best checkpoint: `best_wm.pth`
  - checkpoint metadata: `iter=5000`, `best_eval=0.07760075369151309`, `seed=42`
  - likely source run chain:
    - `run_logs/DINO_WM/2026-02-20_dino_wm_train_job-2416104.md`
    - `run_logs/DINO_WM/2026-02-21_dino_wm_train_job-2431167.md`
    - `run_logs/DINO_WM/2026-02-22_dino_wm_train_job-2438370.md`
    - `run_logs/DINO_WM/2026-02-23_dino_wm_train_job-2449867.md`
- latest checkpoint: `latest_wm.pth`
  - checkpoint metadata: `iter=6000`, `best_eval=0.07760075369151309`, `seed=42`
  - source run log: `run_logs/DINO_WM/2026-02-23_dino_wm_train_job-2449867.md`

## WAN_WM

- checkpoint_dir: `/scratch/u6cr/pravsels.u6cr/latent_safety/wan_wm_checkpoints`
- best checkpoint: `best_wm.pth`
  - checkpoint metadata: `iter=17000`, `best_eval=0.3016213455121033`, `seed=42`
  - likely source run chain:
    - `run_logs/WAN_WM/2026-02-20_wan_wm_train_job-2416301.md`
    - `run_logs/WAN_WM/2026-02-21_wan_wm_train_job-2431168.md`
    - `run_logs/WAN_WM/2026-02-22_wan_wm_train_job-2438372.md`
    - `run_logs/WAN_WM/2026-02-23_wan_wm_train_job-2449868.md`
    - `run_logs/WAN_WM/2026-02-25_wan_wm_train_job-2482028.md`
    - `run_logs/WAN_WM/2026-03-07_wan_wm_train_job-2668095.md`
- latest checkpoint: `latest_wm.pth`
  - checkpoint metadata: `iter=18000`, `best_eval=0.3016213455121033`, `seed=42`
  - source run log: `run_logs/WAN_WM/2026-03-07_wan_wm_train_job-2668095.md`

## Notes

- Existing run logs record dataset/script/config and losses but do not record exact checkpoint artifact snapshots per run.
- For future runs, add explicit `checkpoint` and `config_snapshot` paths in each run log `## Results` block.
