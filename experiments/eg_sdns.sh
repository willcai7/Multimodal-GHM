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

python src/training/train_sequential_DNS.py  --model_type='TF' \
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
    --n_model_layer=1 \
    --n_head=4 \
    --layernorm=True \
    --normalize_attn=True \
    --lr_max=3e-4 \
    --lr_min=3e-7 \
    --guide=False \
    --total_iters=30000 \
    --penalty=0.1 \
    --raw=False

        break
    else
        echo "No GPU available, waiting..."
        sleep 60  # Wait for 60 seconds before checking again
    fi
done
