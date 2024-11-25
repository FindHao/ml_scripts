#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Check if input directory is provided
if [ $# -eq 0 ]; then
    INPUT_BASE="/tmp/tritonbench" # Default value
else
    INPUT_BASE="$1" # Use first command line argument
fi

# Get current date and time in YYYYMMDDHHMM format
DATE_STR=$(date +"%Y%m%d_%H%M")

# Check if output directory is provided as second argument
if [ $# -ge 2 ]; then
    OUTPUT_BASE="$2"
else
    OUTPUT_BASE="/tmp/tritonbench/results/${DATE_STR}"
fi

if [ ! -d "$OUTPUT_BASE" ]; then
    mkdir -p "$OUTPUT_BASE"
fi

# Check if merge_ops_results.py exists
if [ ! -f "$SCRIPT_DIR/merge_ops_results.py" ]; then
    echo "Error: merge_ops_results.py not found in $SCRIPT_DIR"
    exit 1
fi

# Loop through all directories in INPUT_BASE
for dir in "$INPUT_BASE"/*/; do
    # Get directory name
    dir_name=$(basename "$dir")

    # Skip the results directory
    if [ "$dir_name" != "results" ]; then
        echo "Processing directory: $dir_name"

        # Create output Excel file path
        output_file="$OUTPUT_BASE/${dir_name}.xlsx"

        # Run the Python script using SCRIPT_DIR
        python "$SCRIPT_DIR/merge_ops_results.py" --input "$dir" --output "$output_file"
    fi
done
