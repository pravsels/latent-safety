# wan_classifier - successful run

## Mode
- run_type: replication
- objective: Validate stable WAN classifier training end-to-end.

## Config
- script: `slurm/train_wan_classifier_slurm.sh`
- config: `configs/wan_classifier_config.yaml`
- dataset: `/scratch/u6cr/pravsels.u6cr/latent_safety` (via `LATENT_SAFETY_DATA_ROOT`)
- key settings: 4 GPUs, DDP launch through `srun`.

## Job
- job_id: `2535941`
- submitted/start: `2026-03-02T18:57:44Z`
- start_human: `Monday, Mar 2nd, 2026`
- end: `2026-03-02T19:17:03Z`
- end_human: `Monday, Mar 2nd, 2026`
- elapsed: `00:19:19`
- node: `nid010959`
- state: `COMPLETED`

## Status
- 2026-03-02 18:57:44 UTC - started.
- 2026-03-02 19:17:03 UTC - completed successfully.

- 2026-03-13 UTC - synced recovered offline run and verified W&B URL.
## Results
- WAN classifier run completed with exit code success in Slurm accounting.
- W&B run was recovered and synced on 2026-03-13.

- start_train_loss: `0.787109` (train_loss)
- end_train_loss: `0.000000` (train_loss)
- start_val_loss: `1.476458` (eval_loss)
- end_val_loss: `0.020581` (eval_loss)
- loss_one_liner: Train loss decreased from 0.787109 to 0.000000, and validation loss decreased from 1.476458 to 0.020581.

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/latent_safety/wandb/wandb/offline-run-20260302_185806-dvul8d9u`
- synced: `https://wandb.ai/pravsels/wan_wm_action_traj_classifier/runs/dvul8d9u`
- notes: sync succeeded; dashboard interpretation pending.

## Next
- use this as the baseline successful WAN classifier configuration.
