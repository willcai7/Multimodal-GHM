cd "../"

python -m ghmclip.training.train_SimCLR  \
    --job_name='SimCLR' \
    --model_type='TF' \
    --n_tree_layer=4 \
    --n_tree_child=3 \
    --p_tree_flip=0.2 \
    --flip_scale=1 \
    --batch_size=128 \
    --variable_type=10 \
    --d_eb=256 \
    --n_model_layer=4 \
    --n_head=4 \
    --lr_max=1e-3 \
    --lr_min=1e-6 \
    --translation_invariance=True \
    --total_iters=30000 \
    --raw=False \
    --loss_norm=True \
    --temperature=1.0 \
    --raw=True