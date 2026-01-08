#!/bin/bash
#SBATCH --job-name=generate_arx5_hdf5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --requeue

module purge
module load brics/apptainer-multi-node

# Paths
home_dir="/home/u5dm/pravsels.u5dm"
repo_dir="${home_dir}/latent_safety"
scratch_dir="/scratch/u5dm/pravsels.u5dm/latent_safety"
container="${scratch_dir}/container/latent_safety_arm64.sif"
HF_CACHE="/scratch/u5dm/pravsels.u5dm/huggingface_cache"
SCRATCH_WEIGHTS="/scratch/u5dm/pravsels.u5dm/latent_safety/weights"

# Input/Output config
DATASETS_LIST="${repo_dir}/arx5_datasets_new.json"
OUTPUT_HDF5="${scratch_dir}/arx5_datasets_new.h5"
BATCH_SIZE=256

mkdir -p "${scratch_dir}" "${HF_CACHE}"

start_time="$(date -Is --utc)"

# Run command with --resume to handle potential pre-emptions/requeues
GEN_CMD="python scripts/lerobot_to_hdf5.py \
    --datasets-list ${DATASETS_LIST} \
    --output-hdf5 ${OUTPUT_HDF5} \
    --batch-size ${BATCH_SIZE} \
    --resume"

echo "Running command: ${GEN_CMD}"

srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    --bind "${home_dir}:${home_dir}" \
    --bind "${HF_CACHE}:/root/.cache/huggingface" \
    --bind "${SCRATCH_WEIGHTS}:${repo_dir}/weights" \
    --env "HF_HOME=/root/.cache/huggingface" \
    "${container}" \
    bash -c "export PYTHONPATH=${repo_dir}:\$PYTHONPATH && ${GEN_CMD}"

end_time="$(date -Is --utc)"
echo
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"

