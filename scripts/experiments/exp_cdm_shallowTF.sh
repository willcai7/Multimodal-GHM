export CUDA_VISIBLE_DEVICES=2 

# Create logs directory if it doesn't exist
mkdir -p logs/temp

p_flip_list=(0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3 0.32 0.34 0.36 0.38 0.4)

# Run all experiments in parallel
for p_flip in "${p_flip_list[@]}"; do
    # Create unique log filename based on p_flip value
    log_file="logs/temp/CDM_shallowTF_p${p_flip}.log"
    
    # Run experiment in background and redirect output to log file
    (
        python -m ghmclip.training.train_sequential_DNS \
            --clip_feature='TF' \
            --job_name='CDM' \
            --model_type='TF' \
            --n_ttree_layer=4 \
            --n_itree_layer=4 \
            --n_ttree_child=3 \
            --n_itree_child=3 \
            --p_ttree_flip=$p_flip \
            --p_itree_flip=$p_flip \
            --flip_scale=1 \
            --sigma=1 \
            --batch_size=128 \
            --variable_type=10 \
            --d_eb=128 \
            --n_model_layer=1 \
            --n_head=4 \
            --layernorm=True \
            --normalize_attn=True \
            --lr_max=1e-3 \
            --lr_min=1e-6 \
            --guide=False \
            --total_iters=30000 \
            --penalty=0.1 \
            --raw=False
    ) > "$log_file" 2>&1 &
    
    echo "Started experiment with p_flip=$p_flip, logging to $log_file"
done

# Wait for all background jobs to complete
echo "All experiments started. Waiting for completion..."
wait
echo "All experiments completed!"

# Optional: Show a summary of log files
echo "Log files created:"
ls -la logs/temp/CDM_shallowTF_p*.log 