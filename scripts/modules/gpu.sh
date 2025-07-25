#!/bin/bash
find_gpu() {
    local visible_devices="${1:-0,1,2,3}"
    local target_utilization="${2:-0}"
    local target_memory_percent="${3:-10}"  # Default: 10% memory usage
    local gpu_index=-1
    while [ $gpu_index -eq -1 ]; do
        # Create a temporary file
        local tmp_file=$(mktemp)
        # Get both used memory and total memory
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits > "$tmp_file"
        
        while IFS=',' read -r index utilization memory_used memory_total; do
            # Remove leading/trailing whitespace
            index=$(echo "$index" | xargs)
            utilization=$(echo "$utilization" | xargs)
            memory_used=$(echo "$memory_used" | xargs)
            memory_total=$(echo "$memory_total" | xargs)
            
            # Calculate memory usage percentage
            memory_percent=$(echo "scale=2; $memory_used * 100 / $memory_total" | bc -l)
            
            # Use grep instead of bash pattern matching
            if echo ",$visible_devices," | grep -q ",$index," && \
               (( $(echo "$utilization < $target_utilization" | bc -l) )) && \
               (( $(echo "$memory_percent < $target_memory_percent" | bc -l) )); then
                gpu_index=$index
                break
            fi
        done < "$tmp_file"
        
        rm "$tmp_file"
        
        if [ $gpu_index -eq -1 ]; then
            sleep 30
        fi
    done
    echo $gpu_index
}

# Output of this function:
# The find_gpu function returns the index of the first available GPU that meets these criteria:
# 1. The GPU index is in the list of visible devices (default: 0,1,2,3)
# 2. The GPU utilization is less than the target utilization (default: 0)
# 3. The GPU memory usage percentage is less than the target percentage (default: 10%)
# 
# If no GPU meets these criteria, the function will wait 30 seconds and check again,
# continuing until it finds a suitable GPU. The function then outputs the index
# of the selected GPU as a single number.