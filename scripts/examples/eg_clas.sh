#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python -m ghmclip.training.train_CLS  \
    --model_type='TF' \
    --n_tree_layer=4 \
    --n_tree_child=3 \
    --p_tree_flip=0.4 \
    --flip_scale=1 \
    --batch_size=128 \
    --variable_type=10 \
    --d_eb=64 \
    --n_model_layer=9 \
    --n_head=4 \
    --layernorm=True \
    --normalize_attn=True \
    --lr_max=1e-3 \
    --lr_min=1e-6 \
    --guide=True \
    --total_iters=3000 \
    --penalty=0.001 \
    --raw=True