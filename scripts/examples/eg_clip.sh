#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python -m ghmclip.training.train_CLIP  \
    --clip_model_type='TF' \
    --n_ttree_layer=4 \
    --n_itree_layer=4 \
    --n_ttree_child=3 \
    --n_itree_child=3 \
    --p_ttree_flip=0.4 \
    --p_itree_flip=0.4 \
    --flip_scale=1 \
    --K=4 \
    --batch_size=128 \
    --variable_type=10 \
    --clip_tmodel_nlayer=5\
    --clip_imodel_nlayer=5\
    --clip_tmodel_nhead=4 \
    --clip_imodel_nhead=4 \
    --clip_tmodel_deb=128 \
    --clip_imodel_deb=128 \
    --clip_layernorm=True \
    --clip_attennorm=True \
    --clip_guide=False \
    --lr_max=1e-3 \
    --lr_min=1e-6 \
    --total_iters=3000 \
    --penalty=1e-3 \
    --raw=True
