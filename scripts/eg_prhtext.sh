cd "../"

python src/training/train_PRH_NWP.py  \
    --job_name='PRH_text' \
    --model_type='TF' \
    --n_ttree_layer=4 \
    --n_itree_layer=4 \
    --n_ttree_child=3 \
    --n_itree_child=3 \
    --p_ttree_flip=0.2 \
    --p_itree_flip=0.6 \
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
    --translation_invariance=True \
    --total_iters=30000 \
    --raw=False