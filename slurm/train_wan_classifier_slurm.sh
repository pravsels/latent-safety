#!/bin/bash
#SBATCH --job-name=wan_classifier
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
WANDB_DIR="${data_dir}/wandb"
WANDB_CACHE_DIR="${data_dir}/wandb_cache"
WANDB_CONFIG_DIR="${data_dir}/wandb_config"

# Config-driven classifier run
CONFIG_FILE="${repo_dir}/configs/wan_classifier_config.yaml"

mkdir -p "${repo_dir}/logs" "${PYTHON_EXT_DIR}" \
  "${HF_CACHE}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started (UTC): ${start_time}"
echo "===================================="

TRAIN_CMD="python dino_wm/train_wan_classifier.py --config ${CONFIG_FILE}"

echo "Running training command..."
echo "Command: ${TRAIN_CMD}"
echo ""

# Resolve MASTER_ADDR on host (scontrol is unavailable inside container).
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
        export LATENT_SAFETY_DATA_ROOT=${data_dir} && \
        export WANDB_DIR=${WANDB_DIR} WANDB_CACHE_DIR=${WANDB_CACHE_DIR} WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR} && \
        export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID} && \
        export MASTER_ADDR=${MASTER_ADDR} && \
        export MASTER_PORT=${MASTER_PORT} && \
        echo \"[task \${RANK}] RANK=\${RANK} WORLD_SIZE=\${WORLD_SIZE} LOCAL_RANK=\${LOCAL_RANK} MASTER_ADDR=\${MASTER_ADDR} MASTER_PORT=\${MASTER_PORT}\" && \
        ${TRAIN_CMD}"
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
    echo "ERROR: WAN classifier training failed with exit code ${EXIT_CODE}"
    echo "Check slurm-${SLURM_JOB_ID}.err for detailed error messages"
    exit ${EXIT_CODE}
fi

exit 0
