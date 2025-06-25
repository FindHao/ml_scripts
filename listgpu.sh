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
echo "Detailed GPU Process Information:"
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv 