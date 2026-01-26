#!/bin/bash
#SBATCH --job-name=generate_cubes_push_v3
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
repo_dir="${home_dir}/latent_safety"
scratch_dir="/scratch/u5dm/pravsels.u5dm"
data_dir="${scratch_dir}/latent_safety"
container="${data_dir}/container/latent_safety_arm64.sif"
HF_CACHE="${scratch_dir}/huggingface_cache"
SCRATCH_WEIGHTS="${data_dir}/weights"

# Input/Output config
INPUT_HDF5="${data_dir}/cubes_push_labeled_combined.h5"
OUTPUT_HDF5="${data_dir}/cubes_push_labeled_combined_v3.h5"
BATCH_SIZE=256

mkdir -p "${data_dir}" "${HF_CACHE}"

start_time="$(date -Is --utc)"

# Run command with --resume to handle potential pre-emptions/requeues
GEN_CMD="python scripts/add_dino_embeds_to_hdf5.py \
    --input-hdf5 ${INPUT_HDF5} \
    --output-hdf5 ${OUTPUT_HDF5} \
    --batch-size ${BATCH_SIZE} \
    --dino-version v3 \
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
    "${container}" \
    bash -c "export PYTHONPATH=${repo_dir}:\$PYTHONPATH && \
        export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 && \
        ${GEN_CMD}"

end_time="$(date -Is --utc)"
echo
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"

