export CUDA_VISIBLE_DEVICES=2

# Create logs directory if it doesn't exist
mkdir -p logs/temp

p_flip_list=(0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3 0.32 0.34 0.36 0.38 0.4)

# Run all experiments in parallel
for p_flip in "${p_flip_list[@]}"; do
    # Create unique log filename based on p_flip value
    log_file="logs/temp/CLIP_standardTF_p${p_flip}.log"
    
    # Run experiment in background and redirect output to log file
    (
        python -m ghmclip.training.train_CLIP  \
            --job_name='CLIP' \
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
            --clip_tmodel_nlayer=5 \
            --clip_imodel_nlayer=5 \
            --clip_tmodel_nhead=4 \
            --clip_imodel_nhead=4 \
            --clip_tmodel_deb=128 \
            --clip_imodel_deb=128 \
            --clip_layernorm=True \
            --clip_attennorm=True \
            --clip_guide=False \
            --lr_max=3e-4 \
            --lr_min=3e-7 \
            --total_iters=3000 \
            --penalty=1e-3 \
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
ls -la logs/temp/CLIP_standardTF_p*.log