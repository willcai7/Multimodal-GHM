#!/bin/bash
save_script_and_src() {
    # Get timestamp once and use it consistently
    local exp_uid=$1
    
    # Define directories with clear naming
    local output_dir="./outputs"
    local backup_dir="${output_dir}/backup"
    local backup_file="${backup_dir}/code_backup_${exp_uid}.tar.gz"
    
    # Create directories with error checking
    mkdir -p "$output_dir" || { echo "ERROR: Failed to create output directory"; return 1; }
    mkdir -p "$backup_dir" || { echo "ERROR: Failed to create backup directory"; return 1; }
    
    # Verify directories exist
    [ -d "$output_dir" ] || { echo "ERROR: Output directory doesn't exist after creation"; return 1; }
    [ -d "$backup_dir" ] || { echo "ERROR: Backup directory doesn't exist after creation"; return 1; }
    
    # Create tar archive
    tar -czf "$backup_file" ./src/ ./exp.sh || { echo "ERROR: tar command failed"; return 1; }
    
    # Verify tar file was created
    if [ -f "$backup_file" ]; then
        echo "SUCCESS: Backup created at: $backup_file"
    else
        echo "ERROR: Backup file not created: $backup_file"
        return 1
    fi
    
    return 0
}
