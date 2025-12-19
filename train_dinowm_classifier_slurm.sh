#!/bin/bash
#SBATCH --job-name=train_classifier
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

# Set paths
SCRATCH_DIR="/scratch/u5dm/pravsels.u5dm/latent_safety"
CONTAINER="/home/u5dm/pravsels.u5dm/latent_safety/container/latent_safety_arm64.sif"
WORK_DIR="/home/u5dm/pravsels.u5dm/latent_safety"

# Train classifier
echo "Training DINO classifier..."
apptainer exec --nv \
     --pwd "$WORK_DIR" \
     --bind "$SCRATCH_DIR:/scratch/u5dm/pravsels.u5dm/latent_safety" \
     "$CONTAINER" \
     python dino_wm/train_dino_classifier.py \
     --hdf5-file $SCRATCH_DIR/cubes_push_labeled_combined.h5 \
     --dataset-stats $SCRATCH_DIR/cubes_push_labeled_dataset_stats.json \
     --decoder-checkpoint $SCRATCH_DIR/dino_decoder_checkpoints/testing_decoder.pth \
     --wm-checkpoint $SCRATCH_DIR/dino_wm_checkpoints/best_wm.pth \
     --batch-size 64 \
     --sequence-length 16 \
     --wandb-project latent-safety \
     --wandb-name cubes_push_classifier \
     --checkpoint-dir $SCRATCH_DIR/dino_wm_checkpoints

echo "Training completed!"

