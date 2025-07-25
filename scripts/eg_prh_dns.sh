cd "../"

python -m ghmclip.training.train_PRH_DNS \
    --job_name='PRH_DNS' \
    --model_type='TF' \
    --n_tree_layer=4 \
    --n_tree_child=3 \
    --p_tree_flip=0.4 \
    --flip_scale=1 \
    --batch_size=256 \
    --variable_type=10 \
    --d_eb=256 \
    --n_model_layer=9 \
    --n_head=1 \
    --guide=True \
    --lr_max=1e-3 \
    --lr_min=1e-6 \
    --translation_invariance=True \
    --total_iters=3000 \
    --seed_tree=42 \
    --penalty=0.001 \
    --raw=True