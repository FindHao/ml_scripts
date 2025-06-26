#!/bin/bash

# Script to list all GPU processes with detailed process information

# Check if nvidia-smi is available
if ! command -v nvidia-smi &>/dev/null; then
    echo "nvidia-smi command not found. Are NVIDIA drivers installed?"
    exit 1
fi

echo "Listing all GPU processes..."
echo "============================================="

# Get all process IDs using GPUs
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)

if [ -z "$GPU_PIDS" ]; then
    echo "No GPU processes found."
    exit 0
fi

# Print header
printf "%-8s %-12s %-15s %-20s %-10s %-s\n" "PID" "USER" "COMMAND" "START_TIME" "CPU%" "FULL_COMMAND"
printf "%-8s %-12s %-15s %-20s %-10s %-s\n" "----" "----" "-------" "----------" "----" "------------"

# Process each GPU PID
for PID in $GPU_PIDS; do
    # Get process information
    PROCESS_USER=$(ps -o user= -p $PID 2>/dev/null)
    PROCESS_COMMAND=$(ps -o comm= -p $PID 2>/dev/null)
    PROCESS_START=$(ps -o lstart= -p $PID 2>/dev/null)
    PROCESS_CPU=$(ps -o pcpu= -p $PID 2>/dev/null)
    PROCESS_FULL_CMD=$(ps -o cmd= -p $PID 2>/dev/null)
    
    # Check if process still exists
    if [ -z "$PROCESS_USER" ]; then
        printf "%-8s %-12s %-15s %-20s %-10s %-s\n" "$PID" "N/A" "N/A" "N/A" "N/A" "Process not found"
    else
        # Truncate long commands for display
        TRUNCATED_CMD=$(echo "$PROCESS_FULL_CMD" | cut -c1-50)
        if [ ${#PROCESS_FULL_CMD} -gt 50 ]; then
            TRUNCATED_CMD="${TRUNCATED_CMD}..."
        fi
        
        printf "%-8s %-12s %-15s %-20s %-10s %-s\n" "$PID" "$PROCESS_USER" "$PROCESS_COMMAND" "$PROCESS_START" "${PROCESS_CPU}%" "$TRUNCATED_CMD"
    fi
done

echo ""
echo "============================================="
echo "GPU Memory Usage:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

echo ""
echo "============================================="
echo "Detailed GPU Process Information:"
echo "============================================="

# Get GPU UUID to index mapping
GPU_MAPPING=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)

# Get compute apps with GPU UUID
COMPUTE_APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits)

if [ -n "$COMPUTE_APPS" ]; then
    # Print header with better alignment
    printf "%-6s %-8s %-70s %-12s\n" "GPU" "PID" "Process Name" "Memory Used"
    printf "%-6s %-8s %-70s %-12s\n" "===" "========" "======================================================================" "============"
    
    while IFS=',' read -r gpu_uuid pid process_name memory_used; do
        # Find GPU index for this UUID
        gpu_index=$(echo "$GPU_MAPPING" | grep "$gpu_uuid" | cut -d',' -f1)
        
        # Clean up whitespace from fields
        gpu_index=$(echo "$gpu_index" | xargs)
        pid=$(echo "$pid" | xargs)
        process_name=$(echo "$process_name" | xargs)
        memory_used=$(echo "$memory_used" | xargs)
        
        # Truncate process name if too long (increased limit to 69 characters)
        if [ ${#process_name} -gt 69 ]; then
            process_name="${process_name:0:66}..."
        fi
        
        printf "%-6s %-8s %-70s %-12s\n" "$gpu_index" "$pid" "$process_name" "$memory_used"
    done <<< "$COMPUTE_APPS"
    
    echo "============================================="
else
    echo "No compute applications found."
    echo "============================================="
fi 