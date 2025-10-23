#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=24

module purge
module load brics/apptainer-multi-node

echo "=== GPU/CPU Summary ==="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-N/A}"
if command -v nvidia-smi &>/dev/null; then nvidia-smi -L; else echo "nvidia-smi not found"; fi
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "nproc: $(nproc 2>/dev/null || echo N/A)"
echo

home_dir="/home/pravsels.u5dm"
repo="latent_safety"
repo_dir="${home_dir}/${repo}"
container="${repo_dir}/container/${repo}.sif"
entrypoint="python scripts/dreamer_offline.py"

start_time="$(date -Is --utc)"

srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
  apptainer exec --nv \
  --bind "${home_dir}:${home_dir}" \
  --pwd "${repo_dir}" \
  "${container}" \
  ${entrypoint}

end_time="$(date -Is --utc)"

echo
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"


