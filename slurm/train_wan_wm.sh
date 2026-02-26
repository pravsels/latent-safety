#!/bin/bash
#SBATCH --job-name=wan_wm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=0G
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -e

module purge
module load brics/apptainer-multi-node

# Paths
home_dir="/home/u6cr/pravsels.u6cr"
scratch_dir="/scratch/u6cr/pravsels.u6cr"
repo_dir="${home_dir}/latent_safety"
data_dir="${scratch_dir}/latent_safety"
container="${data_dir}/container/latent_safety_arm64.sif"
PYTHON_EXT_DIR="${data_dir}/python_packages"
HF_CACHE="${scratch_dir}/huggingface_cache"
WANDB_DIR="${data_dir}"
WANDB_CACHE_DIR="${data_dir}/wandb_cache"
WANDB_CONFIG_DIR="${data_dir}/wandb_config"

# Training config
HDF5_FILE="${data_dir}/arx5_datasets_6Feb_26_wan224.h5"
STATS_FILE="${data_dir}/arx5_datasets_6Feb_26_stats.json"
CONFIG_FILE="configs/wan_wm_config.yaml"
WAN_VAE_MODEL="ByteDance/Video-As-Prompt-Wan2.1-14B"
CHECKPOINT_DIR="${data_dir}/wan_wm_checkpoints"

mkdir -p "${PYTHON_EXT_DIR}" "${HF_CACHE}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${CHECKPOINT_DIR}"

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started (UTC): ${start_time}"
echo "===================================="

# Step 1: Generate dataset stats once (rank 0), then wait for file.
STATS_CMD="if [ \"\${SLURM_PROCID:-0}\" = \"0\" ]; then \
    if [ ! -f ${STATS_FILE} ]; then \
        python scripts/compute_stats_json.py --file ${HDF5_FILE} --output ${STATS_FILE}; \
    else echo 'Stats file already exists: ${STATS_FILE}'; fi; \
fi; \
while [ ! -f ${STATS_FILE} ]; do sleep 2; done"

# Step 2: WAN backbone WM training
TRAIN_CMD="python dino_wm/train_wan_wm.py \
    --config ${CONFIG_FILE} \
    --hdf5-file ${HDF5_FILE} \
    --dataset-stats ${STATS_FILE} \
    --checkpoint-dir ${CHECKPOINT_DIR} \
    --wan-vae-model ${WAN_VAE_MODEL} \
    --auto-resume"

INSTALL_TORCHMETRICS_CMD="python -m pip install --upgrade --no-deps --target ${PYTHON_EXT_DIR} torchmetrics lightning-utilities packaging"

echo "Running stats and WAN training..."
echo "Command: ${STATS_CMD} && ${TRAIN_CMD}"
echo ""

# Resolve MASTER_ADDR on the host (scontrol is not available inside the container).
MASTER_ADDR=$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)
MASTER_PORT="${MASTER_PORT:-29500}"
echo "DDP env (host-side): MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"
echo "SLURM vars: SLURM_NODELIST=${SLURM_NODELIST}, SLURM_NTASKS=${SLURM_NTASKS}"
echo ""

set +e
srun --ntasks=4 --gpus=4 --cpu-bind=cores \
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    --bind "${HF_CACHE}:/root/.cache/huggingface" \
    --env "HF_HOME=/root/.cache/huggingface" \
    "${container}" \
    bash -c "export PYTHONPATH=${PYTHON_EXT_DIR}:${repo_dir}:\$PYTHONPATH && \
        export WANDB_DIR=${WANDB_DIR} WANDB_CACHE_DIR=${WANDB_CACHE_DIR} WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR} && \
        export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID} && \
        export MASTER_ADDR=${MASTER_ADDR} && \
        export MASTER_PORT=${MASTER_PORT} && \
        echo \"[task \${RANK}] RANK=\${RANK} WORLD_SIZE=\${WORLD_SIZE} LOCAL_RANK=\${LOCAL_RANK} MASTER_ADDR=\${MASTER_ADDR} MASTER_PORT=\${MASTER_PORT}\" && \
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
    echo "ERROR: WAN training failed with exit code ${EXIT_CODE}"
    echo "Check slurm-${SLURM_JOB_ID}.err for detailed error messages"
    exit ${EXIT_CODE}
fi

exit 0
