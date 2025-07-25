#!/bin/bash

# Import functions from modules
source "$(dirname "$0")/modules/gpu.sh"
source "$(dirname "$0")/modules/email.sh"
source "$(dirname "$0")/modules/spy.sh"
# Hyperparameters for GPU selection
visible_devices="0,1,2,3,4,5,6,7" # Define available GPUs
target_utilization=5   # 60% max 
target_memory=10      # 80% max memory

# Define the list of p_flip values to test
p_flip_list=(0.04 0.08 0.12 0.16 0.20 0.24 0.28 0.32 0.36 0.40 0.44 0.48 0.52 0.56 0.60)
# p_flip_list=(0.36 0.40 0.44 0.48 0.52 0.56 0.60)
# p_flip_list=(0.4 0.6 0.8)
# p_flip_list=(0.44 0.46 0.48)
# Function to run a single experiment
run_experiment() {
    local p_flip=$1
    
    # Get an available GPU
    gpu_index=$(find_gpu "$visible_devices" "$target_utilization" "$target_memory")
    
    # Set CUDA_VISIBLE_DEVICES to the selected GPU
    export CUDA_VISIBLE_DEVICES=$gpu_index
   
    # Generate unique ID for this experiment
    exp_uid=$(date +"%Y%m%d_%H%M%S")
    mkdir -p temp_logs/dns
    log_file="./temp_logs/dns/prh_dns_p${p_flip}_${exp_uid}.log"
    
    # Echo to stderr instead of stdout so it doesn't affect the pid capture
    echo "Running experiment with p_flip=$p_flip on GPU $gpu_index" >&2
    echo "Logging output to $log_file" >&2
    
    # Run the Python script in the background
    python -m ghmclip.training.train_PRH_DNS \
        --job_name='PRH_DNS' \
        --model_type='TF' \
        --n_tree_layer=4 \
        --n_tree_child=3 \
        --p_tree_flip=$p_flip \
        --flip_scale=1 \
        --batch_size=128 \
        --variable_type=10 \
        --d_eb=256 \
        --n_model_layer=1 \
        --n_head=1 \
        --guide=False \
        --lr_max=1e-3 \
        --lr_min=1e-6 \
        --total_iters=40000 \
        --penalty=1e-3 \
        --seed_tree=42 \
        --job='dns' \
        --raw=False > "$log_file" 2>&1 &
    
    # Get the PID of the background process
    local pid=$!
    
    # Return both GPU index and PID
    echo "$gpu_index $pid"
}

# Move to the parent directory
cd "../"

# Create logs directory if it doesn't exist
mkdir -p logs

# Array to store PIDs of all running experiments
declare -a pids
declare -a gpu_indices

# Launch each experiment
for p_flip in "${p_flip_list[@]}"; do
    # Launch the experiment and capture its GPU index and PID
    result=$(run_experiment $p_flip)
    read gpu_index pid <<< "$result"
    pids+=($pid)
    gpu_indices+=($gpu_index)
    
    # Print the captured GPU index and PID
    echo "Started experiment with p_flip=$p_flip on GPU $gpu_index (PID: $pid)"
    
    # Wait before starting the next experiment to avoid overwhelming the system
    sleep 30
done

# # Call the function with the pids array
wait_for_processes "${pids[@]}" "${gpu_indices[@]}"

# # Send email notification when all experiments are completed
send_email "CLIP Experiments Completed" "All CLIP experiments have been completed successfully.

Experiment details:
-------------------
p_flip values tested: ${p_flip_list[*]}
Model type: TF
Tree layers: 4
Tree children: 3
Batch size: 128
Total iterations: 300"

echo "All experiments completed and email notification sent."
