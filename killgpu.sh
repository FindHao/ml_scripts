#!/bin/bash

# Script to kill all GPU processes listed by nvidia-smi

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi command not found. Are NVIDIA drivers installed?"
    exit 1
fi

# Get all process IDs using GPUs
echo "Fetching GPU processes..."
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)

if [ -z "$GPU_PIDS" ]; then
    echo "No GPU processes found."
    exit 0
fi

# Count processes
PROCESS_COUNT=$(echo "$GPU_PIDS" | wc -l)
echo "Found $PROCESS_COUNT GPU processes. Preparing to terminate them..."

# Kill each process
for PID in $GPU_PIDS; do
    PROCESS_NAME=$(ps -p $PID -o comm= 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "Killing process $PID ($PROCESS_NAME)..."
        # let's not use -9 to avoid killing the process forcefully
        kill $PID
        if [ $? -eq 0 ]; then
            echo "Process $PID terminated successfully."
        else
            echo "Failed to terminate process $PID. You might need root privileges."
        fi
    else
        echo "Process $PID no longer exists."
    fi
done

echo "All GPU processes have been terminated."
echo "Sleeping for 2 seconds to verify..."
sleep 2
# Verify all processes are gone
REMAINING=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if [ -z "$REMAINING" ]; then
    echo "Verification complete: No GPU processes remaining."
else
    REMAINING_COUNT=$(echo "$REMAINING" | wc -l)
    echo "Warning: $REMAINING_COUNT GPU processes still running."
    echo "You might need root privileges to terminate these processes:"
    echo "$REMAINING"
fi
