# eval — wan_wm horizon sweep (fresh rerun from HPC checkpoint)

## Provenance
- checkpoint: `checkpoints/wan_wm_checkpoints/best_wm_from_hpc_2026-03-16.pth` (downloaded from `/scratch/u6cr/pravsels.u6cr/latent_safety/wan_wm_checkpoints/best_wm.pth`)
- checkpoint_sha256: `2c45224456bed6350ac44a244b110d230abfc3c302864f528c779d21fb11b18a`
- checkpoint_hf: `https://huggingface.co/pravsels/latent_safety_wan_checkpoints/blob/main/checkpoints/best_wm_from_hpc_2026-03-16.pth`
- eval_artifacts_hf: `https://huggingface.co/pravsels/latent_safety_wan_checkpoints/tree/main/evals/2026-03-16_horizon_sweep_hpc_best`
- source run log chain:
  - `run_logs/WAN_WM/2026-02-20_wan_wm_train_job-2416301.md`
  - `run_logs/WAN_WM/2026-02-21_wan_wm_train_job-2431168.md`
  - `run_logs/WAN_WM/2026-02-22_wan_wm_train_job-2438372.md`
  - `run_logs/WAN_WM/2026-02-23_wan_wm_train_job-2449868.md`
  - `run_logs/WAN_WM/2026-02-25_wan_wm_train_job-2482028.md`
  - `run_logs/WAN_WM/2026-03-07_wan_wm_train_job-2668095.md`
- config_snapshot: `configs/wan_wm_config.yaml` (resolved snapshot path was not recorded in original run logs)
- dataset: `villekuosmanen/bin_pick_pack_coffee_capsules_eval` (episode `0`)

## Job
- job_id: `local`
- submitted/start: `2026-03-16T17:51:31.665Z`
- end: `2026-03-16T18:45:11.565Z`
- runtime: `00:53:40` (two runs: normal + ablation)
- node: `local` (Docker `latent_safety_amd64:latest`, `--gpus all`)

## Metrics
- mode: with future actions
  - latent_mse_k1: `0.113479`
  - latent_mse_k100: `1.083168`
  - state_mse_k1: `0.090661`
  - state_mse_k100: `0.326560`
- mode: no future actions (ablation)
  - latent_mse_k1: `0.190970`
  - latent_mse_k100: `0.782959`
  - state_mse_k1: `0.035225`
  - state_mse_k100: `0.406972`
- key trend:
  - ablation is worse at short horizon in latent space (`K=1`)
  - with-future-actions becomes worse than ablation by long horizon in latent space (`K=100`)
  - state-MSE behavior differs from latent-MSE behavior and should be checked across more episodes

## Qualitative
- WAN horizon curves are highly non-monotonic across modes; single-episode conclusions are fragile.
- The previous OOM-kill issue was resolved by chunked video loading in `scripts/wan_wm_horizon_sweep.py` (`--video-query-batch`).

## Verdict
- verdict: Reproducible WAN checkpoint provenance and GPU-local eval pipeline are now established; action-conditioning effect is mixed and requires multi-episode validation before interpretation.

## Artifacts
- local_dir: `outputs/wan_horizon_sweep_hpc_best_2026-03-16`
- local_files:
  - `horizon_sweep_results_wan_with_future_actions.json`
  - `horizon_sweep_results_wan_no_future_actions.json`
  - `latent_mse_vs_horizon_wan_with_future_actions.png`
  - `state_mse_vs_horizon_wan_with_future_actions.png`
  - `latent_mse_vs_horizon_wan_no_future_actions.png`
  - `state_mse_vs_horizon_wan_no_future_actions.png`
- hf_dir: `https://huggingface.co/pravsels/latent_safety_wan_checkpoints/tree/main/evals/2026-03-16_horizon_sweep_hpc_best`
- hf_files:
  - `https://huggingface.co/pravsels/latent_safety_wan_checkpoints/blob/main/evals/2026-03-16_horizon_sweep_hpc_best/horizon_sweep_results_wan_with_future_actions.json`
  - `https://huggingface.co/pravsels/latent_safety_wan_checkpoints/blob/main/evals/2026-03-16_horizon_sweep_hpc_best/horizon_sweep_results_wan_no_future_actions.json`
  - `https://huggingface.co/pravsels/latent_safety_wan_checkpoints/blob/main/evals/2026-03-16_horizon_sweep_hpc_best/latent_mse_vs_horizon_wan_with_future_actions.png`
  - `https://huggingface.co/pravsels/latent_safety_wan_checkpoints/blob/main/evals/2026-03-16_horizon_sweep_hpc_best/state_mse_vs_horizon_wan_with_future_actions.png`
  - `https://huggingface.co/pravsels/latent_safety_wan_checkpoints/blob/main/evals/2026-03-16_horizon_sweep_hpc_best/latent_mse_vs_horizon_wan_no_future_actions.png`
  - `https://huggingface.co/pravsels/latent_safety_wan_checkpoints/blob/main/evals/2026-03-16_horizon_sweep_hpc_best/state_mse_vs_horizon_wan_no_future_actions.png`

## Next
- rerun WAN horizon sweep on additional episodes (`1..4`) with same checkpoint and settings.
- aggregate per-episode means/intervals for both latent and state MSE.
- compare WAN and DINO curves under the same episode set and sampling stride.
