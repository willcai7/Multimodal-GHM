#!/bin/bash


# Function to get an available GPU (GPU with 0% utilization)
get_available_gpu() {
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | awk '$2 == 0 {print $1}'
}

while true; do
    # Get the first available GPU (idle GPU with 0% utilization)
    available_gpu=$(get_available_gpu | head -n 1)

    # If a GPU is available, export it and run the Python script
    if [ -n "$available_gpu" ]; then
        echo "GPU $available_gpu is available, running the script..."
        
        # Set CUDA_VISIBLE_DEVICES to the available GPU
        export CUDA_VISIBLE_DEVICES=$available_gpu


cd "../"

python src/training/train_CLIP.py  \
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

        break
    else
        echo "No GPU available, waiting..."
        sleep 60  # Wait for 60 seconds before checking again
    fi
done
