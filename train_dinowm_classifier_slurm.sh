#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=24

module purge
module load brics/apptainer-multi-node

nvidia-smi 

echo "=== GPU/CPU Summary ==="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-N/A}"
if command -v nvidia-smi &>/dev/null; then nvidia-smi -L; else echo "nvidia-smi not found"; fi
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "nproc: $(nproc 2>/dev/null || echo N/A)"
echo

home_dir="/home/u5dm/pravsels.u5dm"
scratch_dir="/scratch/u5dm/pravsels.u5dm"
repo="latent_safety"
repo_dir="${home_dir}/${repo}"
data_dir="${scratch_dir}/${repo}"
container="${repo_dir}/container/${repo}_arm64.sif"

# Training parameters (easily configurable)
HDF5_FILE="${data_dir}/cubes_push_labeled_combined.h5"
DATASET_STATS="${data_dir}/cubes_push_labeled_dataset_stats.json"
DECODER_CHECKPOINT="${repo_dir}/dino_decoder_checkpoints/testing_decoder.pth"
WM_CHECKPOINT="${repo_dir}/dino_wm_checkpoints/best_wm.pth"
BATCH_SIZE=256
SEQUENCE_LENGTH=4
WANDB_PROJECT="latent-safety"
WANDB_NAME="cubes_push_classifier"
CHECKPOINT_DIR="${repo_dir}/dino_wm_checkpoints"

# Create logs directory if it doesn't exist
mkdir -p "${repo_dir}/logs"

start_time="$(date -Is --utc)"

srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
apptainer exec --nv \
          --pwd "${repo_dir}" \
          --bind "${scratch_dir}:${scratch_dir}" \
          "${container}" \
          bash -c "export PYTHONPATH=${repo_dir}:\$PYTHONPATH && python dino_wm/train_dino_classifier.py --hdf5-file ${HDF5_FILE} --dataset-stats ${DATASET_STATS} --decoder-checkpoint ${DECODER_CHECKPOINT} --wm-checkpoint ${WM_CHECKPOINT} --batch-size ${BATCH_SIZE} --sequence-length ${SEQUENCE_LENGTH} --wandb-project ${WANDB_PROJECT} --wandb-name ${WANDB_NAME} --checkpoint-dir ${CHECKPOINT_DIR}"

end_time="$(date -Is --utc)"

echo
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"

