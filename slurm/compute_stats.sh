#!/bin/bash
#SBATCH --job-name=compute_stats
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module purge
module load brics/apptainer-multi-node

# Paths
home_dir="/home/u6cr/pravsels.u6cr"
scratch_dir="/scratch/u6cr/pravsels.u6cr"
repo_dir="${home_dir}/latent_safety"
data_dir="${scratch_dir}/latent_safety"
container="${data_dir}/container/latent_safety_arm64.sif"
HF_CACHE="${scratch_dir}/huggingface_cache"
WANDB_DIR="${data_dir}/wandb"
WANDB_CACHE_DIR="${data_dir}/wandb_cache"
WANDB_CONFIG_DIR="${data_dir}/wandb_config"

HDF5_FILE="${data_dir}/arx5_datasets_new.h5"
STATS_FILE="${data_dir}/arx5_datasets_new_stats.json"

mkdir -p "${HF_CACHE}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

srun --ntasks=1 --cpu-bind=cores \
apptainer exec \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    --bind "${HF_CACHE}:/root/.cache/huggingface" \
    --env "HF_HOME=/root/.cache/huggingface" \
    "${container}" \
    bash -c "export WANDB_DIR=${WANDB_DIR} WANDB_CACHE_DIR=${WANDB_CACHE_DIR} WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR} && \
        python scripts/compute_stats_json.py --file ${HDF5_FILE} --output ${STATS_FILE}"
