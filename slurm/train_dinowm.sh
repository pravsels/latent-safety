#!/bin/bash
#SBATCH --job-name=dino_wm
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
scratch_dir="/scratch/u5dm/pravsels.u5dm"
repo_dir="${home_dir}/latent_safety"
data_dir="${scratch_dir}/latent_safety"
container="${data_dir}/container/latent_safety_arm64.sif"

# Training config
HDF5_FILE="${data_dir}/arx5_datasets_new.h5"
STATS_FILE="${data_dir}/arx5_datasets_new_stats.json"
CHECKPOINT_DIR="${data_dir}/dino_wm_checkpoints"
CONFIG_FILE="configs/wm_config.yaml"
CONFIG_PATH="${repo_dir}/${CONFIG_FILE}"

mkdir -p "${CHECKPOINT_DIR}"

start_time="$(date -Is --utc)"

# Step 1: Generate dataset stats if they don't exist
STATS_CMD="if [ ! -f ${STATS_FILE} ]; then \
    python scripts/compute_stats_json.py --file ${HDF5_FILE} --output ${STATS_FILE}; \
    else echo 'Stats file already exists: ${STATS_FILE}'; fi"

# Step 2: Training command
TRAIN_CMD="python dino_wm/train_dino_wm.py \
    --config ${CONFIG_FILE} \
    --hdf5-file ${HDF5_FILE} \
    --dataset-stats ${STATS_FILE} \
    --checkpoint-dir ${CHECKPOINT_DIR} \
    --auto-resume"

echo "Running stats and training..."

srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    "${container}" \
    bash -c "export PYTHONPATH=${repo_dir}:\$PYTHONPATH && ${STATS_CMD} && ${TRAIN_CMD}"

end_time="$(date -Is --utc)"
echo
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"
