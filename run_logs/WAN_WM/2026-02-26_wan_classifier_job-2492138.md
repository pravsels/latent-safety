# wan_classifier - initial classifier attempt

## Mode
- run_type: replication
- objective: Launch WAN classifier training from WM features.

## Config
- script: `slurm/train_wan_classifier_slurm.sh`
- config: `configs/wan_classifier_config.yaml`
- dataset: `/scratch/u6cr/pravsels.u6cr/latent_safety` (via `LATENT_SAFETY_DATA_ROOT`)
- key settings: 4 GPUs, DDP launch through `srun`.

## Job
- job_id: `2492138`
- submitted/start: `2026-02-26T21:53:48Z`
- start_human: `Thursday, Feb 26th, 2026`
- end: `2026-02-26T22:14:19Z`
- end_human: `Thursday, Feb 26th, 2026`
- elapsed: `00:20:31`
- node: `nid010704`
- state: `FAILED`

## Status
- 2026-02-26 21:53:48 UTC - started.
- 2026-02-26 22:14:19 UTC - failed.

- 2026-03-13 UTC - synced recovered offline run and verified W&B URL.
## Results
- WAN classifier run failed before completion.
- W&B run was recovered and synced on 2026-03-13.

- start_train_loss: `0.787109` (train_loss)
- end_train_loss: `0.787109` (train_loss)
- start_val_loss: `n/a` (n/a)
- end_val_loss: `n/a` (n/a)
- loss_one_liner: Train loss stayed flat from 0.787109 to 0.787109; validation loss was not logged.

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/latent_safety/wandb/wandb/offline-run-20260226_215407-2x3fm9sv`
- synced: `https://wandb.ai/pravsels/wan_wm_action_traj_classifier/runs/2x3fm9sv`
- notes: sync succeeded; dashboard interpretation pending.

## Next
- inspect `slurm-2492138.err` and relaunch with the fix.
