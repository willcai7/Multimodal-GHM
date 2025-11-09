#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# Function to get an available GPU (GPU with 0% utilization)
python -m ghmclip.training.train_CDNS  \
    --model_type='TF' \
    --p_flip=0.3 \
    --n_tree_layer=4 \
    --n_tree_child=3 \
    --n_head=4 \
    --sigma=1 \
    --n_model_layer=9 \
    --layernorm=True \
    --maxnorm=False \
    --normalize_attn=True \
    --lr_max=1e-3 \
    --lr_min=1e-6 \
    --d_ff=256 \
    --guide=True \
    --contract=False \
    --total_iters=40000 \
    --penalty=0.1 \
    --raw=False

