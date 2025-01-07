
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


lr_max_list=(3e-3 1e-3 3e-4 1e-4)
# penalty_list=(3e-2 1e-2 3e-3)

for lr_max in "${lr_max_list[@]}"; do
# for penalty in "${penalty_list[@]}"; do 

lr_max_decimal=$(printf "%.10f\n" $lr_max)
lr_min=$(echo "scale=10; $lr_max_decimal / 1000" | bc -l)

python dns_guided_train.py  --model_type='TF' \
    --p_flip=0.3 \
    --n_tree_layer=4 \
    --n_tree_child=3 \
    --n_head=4 \
    --sigma=1 \
    --n_model_layer=9 \
    --layernorm=True \
    --maxnorm=False \
    --normalize_attn=True \
    --lr_max=$lr_max \
    --lr_min=$lr_min \
    --d_ff=256 \
    --guide=True \
    --contract=False \
    --total_iters=40000 \
    --penalty=1e-3 \
    --raw=False

# done 
done 
        break
    else
        echo "No GPU available, waiting..."
        sleep 60  # Wait for 60 seconds before checking again
    fi
done
