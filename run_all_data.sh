#!/bin/bash -l
#
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100:4
#SBATCH --output=magvit2dsmall_%j.log
#SBATCH --error=magvit2dsmallerror_%j.log


module load python
conda activate magviot2

# Load any necessary modules
torchrun --nproc_per_node=4 run_whole_dataset.py --modal nii --ckpt-path path_to_ckpt --config-path configs/magvit2_3d_model_config.yaml -i path_to_valid_volumes -o real_trainin_8_8_16_preprocessed_valid/quantized/
