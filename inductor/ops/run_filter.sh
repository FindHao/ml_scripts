#!/bin/bash

source ~/.notify.sh

# Check if at least input directory is provided
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <input_dir> [output_dir]"
  echo "Example: $0 /tmp/tritonbench/20241122_165859"
  echo "         $0 /tmp/tritonbench/20241122_165859 /tmp/tritonbench/results/202411250935"
  exit 1
fi

# Get input path from command line argument
INPUT_DIR="$1"

# Set output path: either from argument or default with timestamp
if [ "$#" -eq 2 ]; then
  OUTPUT_DIR="$2"
else
  # Generate timestamp in YYYYMMDD_HHMMSS format
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  OUTPUT_DIR="/tmp/tritonbench/results/${TIMESTAMP}"
fi

# Ensure output directory exists
mkdir -p $OUTPUT_DIR

# Run merge operations script
./run_merge_ops.sh $INPUT_DIR $OUTPUT_DIR

# Run summary results script
python summarize_ops_results.py -i $INPUT_DIR -o $OUTPUT_DIR

# Send notification when complete
notify "Inductor ops analysis complete" "Results saved to $OUTPUT_DIR"
