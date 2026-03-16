# dino_classifier - initial classifier attempt

## Mode
- run_type: replication
- objective: Launch Dino classifier training from WM features.

## Config
- script: `slurm/train_dinowm_classifier_slurm.sh`
- config: `configs/dino_classifier_config.yaml`
- dataset: `/scratch/u6cr/pravsels.u6cr/latent_safety` (via `LATENT_SAFETY_DATA_ROOT`)
- key settings: 4 GPUs, DDP launch through `srun`.

## Job
- job_id: `2482022`
- submitted/start: `2026-02-25T20:15:07Z`
- start_human: `Wednesday, Feb 25th, 2026`
- end: `2026-02-25T20:15:53Z`
- end_human: `Wednesday, Feb 25th, 2026`
- elapsed: `00:00:46`
- node: `nid011260`
- state: `FAILED`

## Status
- 2026-02-25 20:15:07 UTC - started.
- 2026-02-25 20:15:53 UTC - failed quickly.

- 2026-03-13 UTC - checked slurm logs; no offline W&B run directory was produced for this early-failed job.
## Results
- classifier run failed before completing a full epoch.
- run failed before W&B initialized, so no offline run was generated.

- start_train_loss: `n/a`
- end_train_loss: `n/a`
- start_val_loss: `n/a`
- end_val_loss: `n/a`
- loss_one_liner: Loss metrics were not logged for this run, so no trend can be inferred.

## W&B
- local: `n/a (job failed before W&B offline run was created)`
- synced: `n/a`
- notes: no W&B run to sync for this attempt.

## Next
- inspect `slurm-2482022.err`, fix failure cause, and relaunch.
