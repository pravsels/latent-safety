#!/bin/bash
#SBATCH --job-name=verify_actions_delta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

module purge
module load brics/apptainer-multi-node

# Paths
home_dir="/home/u6cr/pravsels.u6cr"
scratch_dir="/scratch/u6cr/pravsels.u6cr"
repo_dir="${home_dir}/latent_safety"
data_dir="${scratch_dir}/latent_safety"
container="${data_dir}/container/latent_safety_arm64.sif"

HDF5_FILE="${data_dir}/arx5_datasets_feb6_26.h5"

srun --ntasks=1 --cpu-bind=cores \
apptainer exec \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    "${container}" \
    python scripts/verify_actions_delta.py --file ${HDF5_FILE}
