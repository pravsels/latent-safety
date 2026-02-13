#!/bin/bash
#SBATCH --job-name=generate_wan_embeds
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
repo_dir="${home_dir}/latent_safety"
scratch_dir="/scratch/u6cr/pravsels.u6cr"
data_dir="${scratch_dir}/latent_safety"
container="${data_dir}/container/latent_safety_arm64.sif"
HF_CACHE="${scratch_dir}/huggingface_cache"
SCRATCH_WEIGHTS="${data_dir}/weights"
PYTHON_EXT_DIR="${data_dir}/python_packages"
WANDB_DIR="${data_dir}/wandb"
WANDB_CACHE_DIR="${data_dir}/wandb_cache"
WANDB_CONFIG_DIR="${data_dir}/wandb_config"
HF_TOKEN_FILE="${home_dir}/.hf_token"

# Input/Output config
INPUT_HDF5="${data_dir}/arx5_datasets_6Feb_26.h5"
OUTPUT_HDF5="${data_dir}/arx5_datasets_6Feb_26_wan.h5"
WAN_MODEL="ByteDance/Video-As-Prompt-Wan2.1-14B"
BATCH_SIZE=128

mkdir -p "${data_dir}" "${HF_CACHE}" "${PYTHON_EXT_DIR}" \
  "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

if [[ -f "${HF_TOKEN_FILE}" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
else
  echo "Warning: HF token file not found at ${HF_TOKEN_FILE}. Proceeding unauthenticated."
fi

start_time="$(date -Is --utc)"

GEN_CMD="python scripts/add_wan_embeds_to_hdf5.py \
    --input-hdf5 ${INPUT_HDF5} \
    --output-hdf5 ${OUTPUT_HDF5} \
    --model ${WAN_MODEL} \
    --subfolder vae \
    --dtype bf16 \
    --batch-size ${BATCH_SIZE} \
    --front-key wan_front_embd \
    --wrist-key wan_wrist_embd \
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
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    "${container}" \
    bash -c "export PYTHONPATH=${PYTHON_EXT_DIR}:${repo_dir}:\$PYTHONPATH && \
        export WANDB_DIR=${WANDB_DIR} WANDB_CACHE_DIR=${WANDB_CACHE_DIR} WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR} && \
        export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 && \
        ${GEN_CMD}"

end_time="$(date -Is --utc)"
echo
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"
