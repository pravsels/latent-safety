# generate_wan_embeds - arx5 WAN embedding generation

## Mode
- run_type: pipeline (embedding extraction)
- objective: Generate WAN embeddings and write them into the ARX5 HDF5 dataset.

## Config
- script: `slurm/generate_wan_embeds_hdf5_slurm.sh` (inferred from job name `generate_wan_embeds`)
- config: n/a (direct CLI invocation in slurm output)
- dataset: `/scratch/u6cr/pravsels.u6cr/latent_safety/arx5_datasets_6Feb_26.h5`
- output_dataset: `/scratch/u6cr/pravsels.u6cr/latent_safety/arx5_datasets_6Feb_26_wan224.h5`
- key settings:
  - model: `ByteDance/Video-As-Prompt-Wan2.1-14B`
  - subfolder: `vae`
  - dtype: `bf16`
  - batch_size: `512`
  - front_key: `wan_front_embd`
  - wrist_key: `wan_wrist_embd`
  - resume: enabled

## Job
- job_id: `2372234`
- submitted/start: `2026-02-19T11:14:26Z`
- start_human: `Thursday, Feb 19th, 2026`
- end: `2026-02-20T08:28:53Z`
- end_human: `Friday, Feb 20th, 2026`
- elapsed: `21:14:27`
- node: `nid010222`
- state: `COMPLETED`

## Status
- 2026-02-19 11:14:26 UTC - started.
- 2026-02-20 08:28:53 UTC - completed with exit code `0:0` (via `sacct`).

## Results
- Job completed successfully.
- Slurm output captured the command invocation and start time.
- No explicit W&B run metadata was found in `slurm-2372234.out` (expected for this embedding generation job).

## W&B
- local: `n/a (pipeline job; no W&B logging expected)`
- synced: `n/a`
- notes: no dashboard review needed for this embedding-generation pipeline run.

## Next
- Verify the output HDF5 file exists and has expected embedding keys (`wan_front_embd`, `wan_wrist_embd`).
- Continue logging the next historical job in `run_logs/` (suggestion: `slurm-2416104.out`).
