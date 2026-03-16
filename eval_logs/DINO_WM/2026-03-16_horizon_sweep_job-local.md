# eval — dino_wm horizon sweep (fresh rerun from HPC checkpoint)

## Provenance
- checkpoint: `checkpoints/dino3_wm_checkpoints/best_wm_from_hpc.pth` (downloaded from `/scratch/u6cr/pravsels.u6cr/latent_safety/dino3_wm_checkpoints/best_wm.pth`)
- checkpoint_sha256: `6f3fa2bdc3c6c630f320f6f20729c985e08d0d7c8fdf181086ed6003881baefe`
- checkpoint_hf: `https://huggingface.co/pravsels/latent_safety_dino3_checkpoints/blob/main/checkpoints/best_wm_from_hpc_2026-03-16.pth`
- eval_artifacts_hf: `https://huggingface.co/pravsels/latent_safety_dino3_checkpoints/tree/main/evals/2026-03-16_horizon_sweep_hpc_best`
- source run log: `run_logs/DINO_WM/2026-02-23_dino_wm_train_job-2449867.md` (continuation chain: `2026-02-20_dino_wm_train_job-2416104.md` -> `2026-02-21_dino_wm_train_job-2431167.md` -> `2026-02-22_dino_wm_train_job-2438370.md` -> `2026-02-23_dino_wm_train_job-2449867.md`)
- config_snapshot: `configs/dino_wm_config.yaml` (resolved snapshot path was not recorded in original run logs)
- dataset: `villekuosmanen/bin_pick_pack_coffee_capsules_eval` (episode `0`)

## Job
- job_id: `local`
- submitted/start: `2026-03-16T17:13:56.789Z`
- start_human: `Monday, Mar 16th, 2026`
- end: `2026-03-16T17:26:14.874Z`
- end_human: `Monday, Mar 16th, 2026`
- runtime: `00:12:18` (two runs: normal + ablation)
- node: `local`

## Metrics
- mode: with future actions
  - latent_mse_k1: `0.049343`
  - latent_mse_k100: `0.080723`
  - state_mse_k1: `0.038476`
  - state_mse_k100: `0.111667`
- mode: no future actions (ablation)
  - latent_mse_k1: `0.049758`
  - latent_mse_k100: `0.082872`
  - state_mse_k1: `0.038178`
  - state_mse_k100: `0.096689`
- key trend:
  - with future actions and no-future-actions are now close in this rerun
  - previous large ablation gap was not reproduced with this checkpoint

## Qualitative
- numerical behavior changed significantly from prior notes; needs dashboard/curve inspection before drawing modeling conclusions.

## Verdict
- verdict: Checkpoint provenance is now fixed and reproducible; action-conditioning benefit is inconclusive on this rerun and should be re-validated on additional episodes.

## Next
- archive artifacts from `outputs/horizon_sweep_hpc_best_2026-03-16/`:
  - `horizon_sweep_results_with_future_actions.json`
  - `horizon_sweep_results_no_future_actions.json`
  - `latent_mse_vs_horizon_with_future_actions.png`
  - `latent_mse_vs_horizon_no_future_actions.png`
  - `state_mse_vs_horizon_with_future_actions.png`
  - `state_mse_vs_horizon_no_future_actions.png`
- run the same procedure for WAN checkpoint uploaded from HPC source.
- run equivalent WAN sweep and log it under `eval_logs/WAN_WM/`.
