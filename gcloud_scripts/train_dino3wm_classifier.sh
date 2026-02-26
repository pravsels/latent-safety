#!/usr/bin/env bash
# -e: exit on error, -u: error on unset vars, -o pipefail: catch failures in pipes
set -euo pipefail

# ---------------------------
# User-tunable settings
# ---------------------------
HOST_ROOT="${HOME}/latent_safety"
IMAGE="praveensels/latent_safety_amd64:latest"

CONFIG_PATH="configs/dino_classifier_config.yaml"

# GCloud-local paths (override config defaults)
HDF5_FILE="/workspace/latent_safety/train_data/bin_pick_pack_combined_labeled_dino3.h5"
DATASET_STATS="/workspace/latent_safety/train_data/bin_pick_pack_combined_labeled_stats.json"
DECODER_CKPT="/workspace/latent_safety/dino3_decoder_checkpoints/best_decoder.pth"
WM_CKPT="/workspace/latent_safety/dino3_wm_checkpoints/best_wm.pth"
CLASSIFIER_CKPT_DIR="/workspace/latent_safety/dino3_classifier_checkpoints"

# WandB overrides
WANDB_MODE="offline"
WANDB_PROJECT="dino3_action_traj_classifier"
WANDB_NAME="gcloud-dino3-classifier"

# ---------------------------
# Setup
# ---------------------------
LOG_DIR="${HOST_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${HOST_ROOT}/dino3_classifier_checkpoints"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/classifier_${TIMESTAMP}.log"
START_TIME="$(date -Is --utc)"

echo "===================================="
echo "Host: $(hostname)"
echo "Started (UTC): ${START_TIME}"
echo "Image: ${IMAGE}"
echo "Config: ${CONFIG_PATH}"
echo "Log file: ${LOG_FILE}"
echo "===================================="

# ---------------------------
# Sanity check: CUDA must be available inside the container
# ---------------------------
echo "Checking CUDA availability inside container..."
docker run --rm --gpus all "${IMAGE}" \
  python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'CUDA OK — {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}')"
echo ""

# ---------------------------
# Launch training in container
# ---------------------------
# Disable exit-on-error so we can capture the exit code manually via PIPESTATUS
# (the docker | tee pipeline would otherwise abort the script on non-zero exit)
set +e
docker run --rm --gpus all \
  --ipc=host \
  --network=host \
  -v "${HOST_ROOT}:/workspace/latent_safety" \
  -w /workspace/latent_safety \
  "${IMAGE}" \
  bash -lc "PYTHONFAULTHANDLER=1 python dino_wm/train_dino_classifier.py \
    --config ${CONFIG_PATH} \
    --hdf5-file ${HDF5_FILE} \
    --dataset-stats ${DATASET_STATS} \
    --decoder-checkpoint ${DECODER_CKPT} \
    --wm-checkpoint ${WM_CKPT} \
    --checkpoint-dir ${CLASSIFIER_CKPT_DIR} \
    --wandb-mode ${WANDB_MODE} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-name ${WANDB_NAME} \
    --eval-interval 100" \
  2>&1 | tee "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}  # exit code of docker, not tee
set -e  # re-enable exit-on-error for the rest of the script

END_TIME="$(date -Is --utc)"
echo ""
echo "===================================="
echo "Started (UTC):  ${START_TIME}"
echo "Finished (UTC): ${END_TIME}"
echo "Exit Code: ${EXIT_CODE}"
echo "Log file: ${LOG_FILE}"
echo "===================================="

exit "${EXIT_CODE}"
