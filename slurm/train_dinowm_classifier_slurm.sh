#!/bin/bash
#SBATCH --job-name=dino_classifier
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

# Training config (weights and checkpoints on scratch due to limited home storage)
HDF5_FILE="${data_dir}/cubes_push_labeled_combined_v3.h5"
DATASET_STATS="${data_dir}/cubes_push_labeled_dataset_stats.json"
DECODER_CHECKPOINT="${data_dir}/dino3_decoder_checkpoints/best_decoder.pth"
WM_CHECKPOINT="${data_dir}/dino3_wm_checkpoints/best_wm.pth"
CHECKPOINT_DIR="${data_dir}/dino3_classifier_checkpoints"
BATCH_SIZE=256
SEQUENCE_LENGTH=4
WANDB_PROJECT="latent-safety"
WANDB_NAME="cubes_push_classifier"

mkdir -p "${CHECKPOINT_DIR}" "${repo_dir}/logs"

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started (UTC): ${start_time}"
echo "===================================="

TRAIN_CMD="python dino_wm/train_dino_classifier.py \
    --hdf5-file ${HDF5_FILE} \
    --dataset-stats ${DATASET_STATS} \
    --dino-version v3 \
    --decoder-checkpoint ${DECODER_CHECKPOINT} \
    --wm-checkpoint ${WM_CHECKPOINT} \
    --batch-size ${BATCH_SIZE} \
    --sequence-length ${SEQUENCE_LENGTH} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-name ${WANDB_NAME} \
    --checkpoint-dir ${CHECKPOINT_DIR}"

echo "Running training command..."
echo "Command: ${TRAIN_CMD}"
echo ""

set +e
srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    "${container}" \
    bash -c "export PYTHONPATH=${repo_dir}:\$PYTHONPATH && ${TRAIN_CMD}"
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
