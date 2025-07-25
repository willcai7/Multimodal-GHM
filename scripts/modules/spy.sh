#!/bin/bash
# Function to monitor processes with GPU information in a tmux session
wait_for_processes() {
    local -a process_ids=()
    local -a gpu_indices=()
    
    # Parse arguments - first half are PIDs, second half are GPU indices
    local num_args=$#
    local half_args=$((num_args / 2))
    
    for ((i=1; i<=half_args; i++)); do
        process_ids+=("${!i}")
    done
    
    for ((i=half_args+1; i<=num_args; i++)); do
        gpu_indices+=("${!i}")
    done
    
    echo "Waiting for all experiments to complete..."
    local all_completed=false
    local -A start_times=()
    local -A status=()
    local -A gpu_map=()
    
    # Record start times for all processes and map PIDs to GPUs
    for i in "${!process_ids[@]}"; do
        local pid="${process_ids[$i]}"
        local gpu="${gpu_indices[$i]}"
        start_times[$pid]=$(date +%s)
        status[$pid]="RUNNING"
        gpu_map[$pid]=$gpu
    done
    
    # Function to display process tracking
    display_tracking() {
        local current_time=$(date +%s)
        
        # Clear screen for clean display
        clear
        
        echo "======= PROCESS TRACKING ======="
        printf "%-10s | %-10s | %-10s | %-15s\n" "PID" "GPU" "STATUS" "RUNTIME"
        printf "%s\n" "-------------------------------------------"
        
        # Print each process status
        for pid in "${process_ids[@]}"; do
            local runtime=$((current_time - start_times[$pid]))
            local hours=$((runtime / 3600))
            local minutes=$(( (runtime % 3600) / 60 ))
            local seconds=$((runtime % 60))
            local runtime_str=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)
            
            printf "%-10s | %-10s | %-10s | %-15s\n" "$pid" "${gpu_map[$pid]}" "${status[$pid]}" "$runtime_str"
        done
        
        printf "\nLast updated: %s (Updates every 3s)\n" "$(date '+%Y-%m-%d %H:%M:%S')"
    }
    
    # Check processes efficiently
    check_processes() {
        all_completed=true
        for pid in "${process_ids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                status[$pid]="RUNNING"
                all_completed=false
            elif [ "${status[$pid]}" = "RUNNING" ]; then
                status[$pid]="COMPLETED"
            fi
        done
    }
    
    # Initial display
    display_tracking
    
    # Main monitoring loop
    while [ "$all_completed" = false ]; do
        # Check processes
        check_processes
        
        # Update display
        display_tracking
        
        if [ "$all_completed" = false ]; then
            sleep 3
        fi
    done
    
    echo "All processes have completed!"
    return 0
}

monitor_processes() {
    local -a pids=("$@")  # Accept PIDs as arguments
    local session_name="process_monitor"
    echo "Monitoring processes with PIDs: ${pids[@]}"
    
    if [ ${#pids[@]} -eq 0 ]; then
        echo "Error: No PIDs provided to monitor"
        return 1
    fi
    
    # Check if tmux is installed
    if ! command -v tmux >/dev/null 2>&1; then
        echo "Error: tmux is not installed. Please install it first."
        return 1
    fi
    
    # Create a temporary script to run in the tmux session
    local tmp_script=$(mktemp)
    chmod +x "$tmp_script"
    
    # Write the monitoring script
    cat > "$tmp_script" << 'EOFMARKER'
#!/bin/bash
# Arguments passed to this script will be the PIDs to monitor
pids=("$@")

# Function to get GPU index for a process
get_gpu_for_pid() {
    local pid=$1
    local gpu_info=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null)
    
    if [ -z "$gpu_info" ]; then
        echo "N/A"
        return
    fi
    
    local gpu_idx="N/A"
    while IFS="," read -r app_pid gpu_uuid; do
        app_pid=$(echo "$app_pid" | xargs)
        if [ "$app_pid" == "$pid" ]; then
            # Convert GPU UUID to index
            gpu_idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$gpu_uuid" | cut -d',' -f1 | xargs)
            break
        fi
    done <<< "$gpu_info"
    
    echo "$gpu_idx"
}

# Function to get GPU memory usage for a process (in MB)
get_gpu_memory_for_pid() {
    local pid=$1
    local mem_info=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
    
    if [ -z "$mem_info" ]; then
        echo "N/A"
        return
    fi
    
    local gpu_mem="N/A"
    while IFS="," read -r app_pid used_mem; do
        app_pid=$(echo "$app_pid" | xargs)
        if [ "$app_pid" == "$pid" ]; then
            gpu_mem=$(echo "$used_mem" | xargs)
            break
        fi
    done <<< "$mem_info"
    
    echo "$gpu_mem"
}

# Main monitoring loop
all_done=false
while [ "$all_done" = false ]; do
    all_done=true
    running_count=0
    
    # Clear previous output
    clear
    
    echo "============================================"
    echo "GPU STATUS SUMMARY ($(date))"
    echo "============================================"
    
    # Display GPU status - try gpustat first, fall back to nvidia-smi
    if command -v gpustat >/dev/null 2>&1; then
        gpustat --color
    else
        nvidia-smi
    fi
    
    echo "============================================"
    echo "PROCESS STATUS SUMMARY ($(date))"
    echo "Monitoring ${#pids[@]} processes"
    echo "============================================"
    
    # Format header
    printf "%-10s %-15s %-10s %-15s %-10s\n" "PID" "Runtime" "GPU" "GPU Memory" "Status"
    echo "------------------------------------------------------------"
    
    for pid in "${pids[@]}"; do
        if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
            printf "%-10s %-15s %-10s %-15s %-10s\n" "$pid" "N/A" "N/A" "N/A" "Invalid PID"
            continue
        fi
        
        if kill -0 "$pid" 2>/dev/null; then
            all_done=false
            running_count=$((running_count + 1))
            
            # Get process runtime
            if start_time=$(ps -o lstart= -p "$pid" 2>/dev/null | xargs -0 date +%s -d 2>/dev/null); then
                current_time=$(date +%s)
                runtime_seconds=$((current_time - start_time))
                runtime=$(printf '%02d:%02d:%02d' $((runtime_seconds/3600)) $((runtime_seconds%3600/60)) $((runtime_seconds%60)))
            else
                runtime="Unknown"
            fi
            
            # Get GPU info for this process
            gpu_idx=$(get_gpu_for_pid "$pid")
            gpu_mem=$(get_gpu_memory_for_pid "$pid")
            
            printf "%-10s %-15s %-10s %-15s %-10s\n" "$pid" "$runtime" "$gpu_idx" "$gpu_mem" "Running"
        else
            printf "%-10s %-15s %-10s %-15s %-10s\n" "$pid" "N/A" "N/A" "N/A" "Stopped"
        fi
    done
    
    echo "============================================"
    echo "Total running: $running_count / ${#pids[@]}"
    echo "Press Ctrl+B then D to detach from this monitor (it will keep running)"
    sleep 10
done

echo "All processes completed!"
EOFMARKER
    
    # Check if tmux session already exists
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "Attaching to existing monitor session..."
        tmux attach-session -t "$session_name"
    else
        echo "Creating new monitor session with ${#pids[@]} processes..."
        # Create a new tmux session and run the monitoring script
        tmux new-session -d -s "$session_name" "$tmp_script" "${pids[@]}"
        echo "Monitor started in tmux session: $session_name"
        echo "To view the monitor: tmux attach-session -t $session_name"
        echo "To detach from monitor (keeping it running): Ctrl+B then D"
    fi
    
    # Clean up the temporary script when tmux session ends
    (tmux wait-for -S "$session_name"; rm "$tmp_script") &
    
    # Ask user if they want to attach to the session
    read -p "Do you want to attach to the monitor now? (y/n): " attach
    if [[ "$attach" =~ ^[Yy]$ ]]; then
        tmux attach-session -t "$session_name"
    fi
    
    return 0
}   