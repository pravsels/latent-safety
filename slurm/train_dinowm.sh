#!/bin/bash
#SBATCH --job-name=dino_wm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

# Exit on any error
set -e

module purge
module load brics/apptainer-multi-node

# Paths
home_dir="/home/u5dm/pravsels.u5dm"
scratch_dir="/scratch/u5dm/pravsels.u5dm"
repo_dir="${home_dir}/latent_safety"
data_dir="${scratch_dir}/latent_safety"
container="${data_dir}/container/latent_safety_arm64.sif"
PYTHON_EXT_DIR="${data_dir}/python_packages"

# Training config
HDF5_FILE="${data_dir}/arx5_datasets_new.h5"
STATS_FILE="${data_dir}/arx5_datasets_new_stats.json"
CHECKPOINT_DIR="${data_dir}/dino_wm_checkpoints"
DECODER_CHECKPOINT="${data_dir}/dino_decoder_checkpoints/testing_decoder.pth"
CONFIG_FILE="configs/wm_config.yaml"
CONFIG_PATH="${repo_dir}/${CONFIG_FILE}"

mkdir -p "${CHECKPOINT_DIR}" "${PYTHON_EXT_DIR}"

# Ensure repo weights path points to scratch weights for relative lookups
if [ -L "${repo_dir}/weights" ] || [ ! -e "${repo_dir}/weights" ]; then
    ln -sfn "${data_dir}/weights" "${repo_dir}/weights"
elif [ -d "${repo_dir}/weights" ]; then
    # If a real dir exists, still link the file so relative path resolves.
    ln -sfn "${data_dir}/weights/dinov3_vits16plus.pth" "${repo_dir}/weights/dinov3_vits16plus.pth"
fi

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started (UTC): ${start_time}"
echo "===================================="

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
    --decoder-checkpoint ${DECODER_CHECKPOINT} \
    --auto-resume"

INSTALL_TORCHMETRICS_CMD="python -m pip install --upgrade --no-deps --target ${PYTHON_EXT_DIR} torchmetrics lightning-utilities packaging"

echo "Running stats and training..."
echo "Command: ${STATS_CMD} && ${TRAIN_CMD}"
echo ""

set +e
srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    "${container}" \
    bash -c "export PYTHONPATH=${PYTHON_EXT_DIR}:${repo_dir}:\$PYTHONPATH && \
        if ! python -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec(\"torchmetrics\") else 1)'; then \
            ${INSTALL_TORCHMETRICS_CMD}; \
        fi && ${STATS_CMD} && ${TRAIN_CMD}"
EXIT_CODE=$?
set -e

end_time="$(date -Is --utc)"
echo ""
echo "===================================="
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"
echo "Exit Code: ${EXIT_CODE}"
echo "===================================="

if [ ${EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "ERROR: Training failed with exit code ${EXIT_CODE}"
    echo "Check slurm-${SLURM_JOB_ID}.err for detailed error messages"
    exit ${EXIT_CODE}
fi

exit 0
