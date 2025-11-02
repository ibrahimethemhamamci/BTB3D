#!/bin/bash
python inference.py \
--ckpt-path /anvme/workspace/b180dc42-ctrate/clip_maisi/O2-MAGVIT2/exps/ckpts/train_2d_small_512/iter_79000_fixed.ckpt \
--config-path configs/magvit2_2d_model_config.yaml \
--target-pixels 512 \
-i /anvme/workspace/b180dc42-ctrate/data_volumes/dataset/valid/valid_43/valid_43_a/valid_43_a_2.nii.gz \
-o tests \
--modal nii