#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python -m ghmclip.training.train_sequential_DNS  \
     --model_type='TF' \
    --n_ttree_layer=4 \
    --n_itree_layer=4 \
    --n_ttree_child=3 \
    --n_itree_child=3 \
    --p_ttree_flip=0.04 \
    --p_itree_flip=0.04 \
    --flip_scale=1 \
    --sigma=1 \
    --batch_size=128 \
    --variable_type=10 \
    --d_eb=128 \
    --n_model_layer=9 \
    --n_head=4 \
    --layernorm=True \
    --normalize_attn=True \
    --lr_max=3e-4 \
    --lr_min=3e-7 \
    --guide=True \
    --total_iters=30000 \
    --penalty=0.1 \
    --raw=True

