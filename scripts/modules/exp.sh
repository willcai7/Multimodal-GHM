#!/bin/bash
run_experiment() {
    local model_name="$1"
    local epochs="$2"
    local batch_size="$3"
    local learning_rate="$4"
    local momentum="$5"
    local weight_decay="$6"
    local norm="$7"
    local scheduler_flag="$8"
    local gamma="$9"
    local wandb_name="${11}"
    local exp_uid="${10}"
    local cuda_visible_devices="${12}"

    # Make sure outputs/temp directory exists
    mkdir -p "outputs/temp"
    
    # Generate unique ID for this run
    local uid="$(date +%y%m%d-%H%M%S)"
    
    # Create output directory name
    local output_dir="./outputs"
    mkdir -p "$output_dir"

    local gpu_index=$(find_gpu "$cuda_visible_devices")
    
    echo "$gpu_index"
    
    # Run the experiment and capture both stdout and stderr
    (
        export CUDA_VISIBLE_DEVICES=$gpu_index
        echo "Running on GPU $gpu_index with model $model_name and lr $learning_rate"
        python -m src.trainings.train_resnet \
         --model_name="$model_name" \
         --epochs=$epochs \
         --batch_size=$batch_size \
         --learning_rate=$learning_rate \
         --momentum=$momentum \
         --weight_decay=$weight_decay \
         --norm=$norm \
         --scheduler_flag=$scheduler_flag \
         --gamma=$gamma \
         --wandb_name="$wandb_name" \
         --exp_uid="$exp_uid" \
    ) > "./training_${model_name}_$(date +%y%m%d-%H%M%S).log" 2>&1 &
    
    # Get the PID *after* launching the background process
    local pid=$!
    
    # Update the log file name with the correct PID
    mv "./training_${model_name}_$(date +%y%m%d-%H%M%S).log" "./training_${model_name}_$(date +%y%m%d-%H%M%S)_pid_${pid}.log"
    
    # Return the PID of the background process
    echo $pid
}