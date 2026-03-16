# dino_classifier - retry attempt

## Mode
- run_type: replication
- objective: Retry Dino classifier training after initial launch failure.

## Config
- script: `slurm/train_dinowm_classifier_slurm.sh`
- config: `configs/dino_classifier_config.yaml`
- dataset: `/scratch/u6cr/pravsels.u6cr/latent_safety` (via `LATENT_SAFETY_DATA_ROOT`)
- key settings: 4 GPUs, DDP launch through `srun`.

## Job
- job_id: `2485747`
- submitted/start: `2026-02-25T23:59:51Z`
- start_human: `Wednesday, Feb 25th, 2026`
- end: `2026-02-26T12:38:44Z`
- end_human: `Thursday, Feb 26th, 2026`
- elapsed: `12:38:53`
- node: `nid010288`
- state: `FAILED`

## Status
- 2026-02-25 23:59:51 UTC - started.
- 2026-02-26 12:38:44 UTC - failed after partial runtime.

- 2026-03-13 UTC - synced recovered offline run and verified W&B URL.
## Results
- longer runtime than job `2482022`, but still ended in failure.
- W&B run was recovered and synced on 2026-03-13.

- start_train_loss: `1.560254` (train_loss)
- end_train_loss: `0.545858` (train_loss)
- start_val_loss: `1.356542` (eval_loss)
- end_val_loss: `1.356542` (eval_loss)
- loss_one_liner: Train loss decreased from 1.560254 to 0.545858, and validation loss stayed flat from 1.356542 to 1.356542.

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/latent_safety/wandb/wandb/offline-run-20260226_000007-kqeh0n96`
- synced: `https://wandb.ai/pravsels/dino3_wm_action_traj_classifier/runs/kqeh0n96`
- notes: sync succeeded; dashboard interpretation pending.

## Next
- compare `slurm-2485747.err` vs `slurm-2482022.err` and patch root cause before next retry.
