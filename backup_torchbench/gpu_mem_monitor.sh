#!/bin/bash

while true; do
  # Get available memory for the first GPU
  gpu_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1{print $1}')

  # Check if available memory is greater than 70GB
  if [[ $gpu_mem -gt 70000 ]]; then

    # Call the test.py script using Python
    bash training_scripts/single_gpu/run_350m.sh

    # Check the return code of the script
    if [[ $? -eq 0 ]]; then
      # If the script returns successfully, send a "successfully" notification
      notify "Task completed successfully."
    else
      # If the script fails, send a "fail" notification
      notify "Task failed."
    fi

    exit 0
  fi

  # Sleep for 60 seconds before the next check
  sleep 60
done
