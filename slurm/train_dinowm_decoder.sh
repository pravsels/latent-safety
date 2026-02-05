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
home_dir="/home/u6cr/pravsels.u6cr"
scratch_dir="/scratch/u6cr/pravsels.u6cr"
repo_dir="${home_dir}/latent_safety"
data_dir="${scratch_dir}/latent_safety"
container="${data_dir}/container/latent_safety_arm64.sif"
HF_CACHE="${scratch_dir}/huggingface_cache"
WANDB_DIR="${data_dir}/wandb"
WANDB_CACHE_DIR="${data_dir}/wandb_cache"
WANDB_CONFIG_DIR="${data_dir}/wandb_config"

# Training config (weights and checkpoints on scratch due to limited home storage)
HDF5_FILE="${data_dir}/arx5_datasets_new.h5"
CHECKPOINT_DIR="${data_dir}/dino_decoder_checkpoints"
CONFIG_FILE="configs/dino_decoder_config.yaml"
CONFIG_PATH="${repo_dir}/${CONFIG_FILE}"

mkdir -p "${CHECKPOINT_DIR}" "${HF_CACHE}" \
  "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

start_time="$(date -Is --utc)"

# Auto-resume logic (match trainer behavior):
# - Prefer latest_decoder{_vq}.pth (resume-safe dict checkpoint used by --auto-resume)
# - Fall back to newest decoder_iter*{_vq}.pth or best_decoder{_vq}.pth
suffix=""
if [ -f "${CONFIG_PATH}" ] && grep -Eq '^[[:space:]]*quantize:[[:space:]]*true[[:space:]]*$' "${CONFIG_PATH}"; then
  suffix="_vq"
fi

LATEST_CKPT="${CHECKPOINT_DIR}/latest_decoder${suffix}.pth"
BEST_CKPT="${CHECKPOINT_DIR}/best_decoder${suffix}.pth"
TESTING_SD="${CHECKPOINT_DIR}/testing_decoder${suffix}.pth"
ITER_CKPT="$(ls -t ${CHECKPOINT_DIR}/decoder_iter*${suffix}.pth 2>/dev/null | head -1)"

# Choose resume strategy:
# - latest (default): best for preemption-safe continuation (no progress lost)
# - best: resume from best eval checkpoint (useful if you only care about best-so-far)
RESUME_FROM="${RESUME_FROM:-latest}"

RESUME_CKPT=""
if [ "${RESUME_FROM}" = "best" ]; then
  if [ -f "${BEST_CKPT}" ]; then
    RESUME_CKPT="${BEST_CKPT}"
  elif [ -f "${LATEST_CKPT}" ]; then
    RESUME_CKPT="${LATEST_CKPT}"
  elif [ -n "${ITER_CKPT}" ]; then
    RESUME_CKPT="${ITER_CKPT}"
  elif [ -f "${TESTING_SD}" ]; then
    RESUME_CKPT="${TESTING_SD}"
  fi
else
  if [ -f "${LATEST_CKPT}" ]; then
    RESUME_CKPT="${LATEST_CKPT}"
  elif [ -n "${ITER_CKPT}" ]; then
    RESUME_CKPT="${ITER_CKPT}"
  elif [ -f "${BEST_CKPT}" ]; then
    RESUME_CKPT="${BEST_CKPT}"
  elif [ -f "${TESTING_SD}" ]; then
    RESUME_CKPT="${TESTING_SD}"
  fi
fi

TRAIN_CMD="python dino_wm/train_dino_decoder.py \
    --config ${CONFIG_FILE} \
    --hdf5-file ${HDF5_FILE} \
    --checkpoint-dir ${CHECKPOINT_DIR} \
    --auto-resume"

if [ -n "${RESUME_CKPT}" ]; then
  echo "Found checkpoint ${RESUME_CKPT}. Resuming..."
  TRAIN_CMD="${TRAIN_CMD} --resume-checkpoint ${RESUME_CKPT}"
else
  echo "No checkpoint found. Starting fresh training..."
fi

srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    --bind "${HF_CACHE}:/root/.cache/huggingface" \
    --env "HF_HOME=/root/.cache/huggingface" \
    "${container}" \
    bash -c "export PYTHONPATH=${repo_dir}:\$PYTHONPATH && \
        export WANDB_DIR=${WANDB_DIR} WANDB_CACHE_DIR=${WANDB_CACHE_DIR} WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR} && \
        ${TRAIN_CMD}"

end_time="$(date -Is --utc)"
echo
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"
