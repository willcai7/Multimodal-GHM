#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python -m ghmclip.training.train_sequential_NWP  \
    --job_name='exp_snwp' \
    --clip_feature='TF' \
    --model_type='TF' \
    --n_ttree_layer=4 \
    --n_itree_layer=4 \
    --n_ttree_child=3 \
    --n_itree_child=3 \
    --p_ttree_flip=0.02 \
    --p_itree_flip=0.02 \
    --flip_scale=1 \
    --batch_size=128 \
    --variable_type=10 \
    --d_eb=256 \
    --n_model_layer=9 \
    --n_head=4 \
    --layernorm=True \
    --normalize_attn=True \
    --lr_max=1e-3 \
    --lr_min=1e-6 \
    --guide=False \
    --total_iters=40000 \
    --penalty=0.001 \
    --raw=True
