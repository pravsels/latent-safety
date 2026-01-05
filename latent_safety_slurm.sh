#!/bin/bash
#SBATCH --job-name=dino_decoder
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

# Training config (weights and checkpoints on scratch due to limited home storage)
HDF5_FILE="${data_dir}/arx5_subset_train.h5"
CHECKPOINT_DIR="${data_dir}/dino_decoder_checkpoints"
BATCH_SIZE=256

mkdir -p "${CHECKPOINT_DIR}"

start_time="$(date -Is --utc)"

# Auto-resume logic
LAST_CHECKPOINT=$(ls -t ${CHECKPOINT_DIR}/*.pth 2>/dev/null | head -1)
if [ -n "${LAST_CHECKPOINT}" ]; then
    echo "Found existing checkpoint at ${LAST_CHECKPOINT}. Resuming training..."
    TRAIN_CMD="python dino_wm/train_dino_decoder.py \
        --hdf5-file ${HDF5_FILE} \
        --batch-size ${BATCH_SIZE} \
        --quantize \
        --auto-resume"
else
    echo "No checkpoint found. Starting fresh training..."
    TRAIN_CMD="python dino_wm/train_dino_decoder.py \
        --hdf5-file ${HDF5_FILE} \
        --batch-size ${BATCH_SIZE} \
        --quantize"
fi

srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    "${container}" \
    bash -c "export PYTHONPATH=${repo_dir}:\$PYTHONPATH && ${TRAIN_CMD}"

end_time="$(date -Is --utc)"
echo
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"
