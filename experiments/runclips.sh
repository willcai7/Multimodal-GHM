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
        # lr_max_list=(1e-2 3e-3 3e-4 1e-4)
        # p_flip_list=(0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20)
        p_flip_list=(0.22 0.24 0.26 0.28 0.30 0.32 0.34 0.36 0.38 0.40)
        # for lr_max in "${lr_max_list[@]}"; do
        for p_flip in "${p_flip_list[@]}"; do

        # lr_max_decimal=$(printf "%.10f\n" $lr_max)
        # lr_min=$(echo "scale=10; $lr_max_decimal / 1000" | bc -l)

        python src/training/train_CLIP.py  \
            --clip_model_type='TF' \
            --n_ttree_layer=4 \
            --n_itree_layer=4 \
            --n_ttree_child=3 \
            --n_itree_child=3 \
            --p_ttree_flip=$p_flip \
            --p_itree_flip=$p_flip \
            --flip_scale=1 \
            --K=4 \
            --batch_size=128 \
            --variable_type=10 \
            --clip_tmodel_nlayer=1 \
            --clip_imodel_nlayer=1 \
            --clip_tmodel_nhead=4 \
            --clip_imodel_nhead=4 \
            --clip_tmodel_deb=128 \
            --clip_imodel_deb=128 \
            --clip_layernorm=True \
            --clip_attennorm=True \
            --clip_guide=False \
            --lr_max=3e-4 \
            --lr_min=3e-7 \
            --total_iters=40000 \
            --penalty=1e-3 \
            --raw=False
        done

        break
    else
        echo "No GPU available, waiting..."
        sleep 60  # Wait for 60 seconds before checking again
    fi
done
