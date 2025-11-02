#!/bin/bash
torchrun --nproc_per_node=2 run_whole_dataset.py --modal nii --ckpt-path /anvme/workspace/b180dc42-ctrate/clip_maisi/O2-MAGVIT2/exps/ckpts/train_3d_decoderfinetuning/iter_23800_fixed.ckpt --config-path configs/magvit2_3d_model_config_old.yaml -i /anvme/workspace/b180dc42-ctrate/data_volumes/dataset/valid/ -o testo
