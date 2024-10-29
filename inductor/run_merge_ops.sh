#!/bin/bash

# Check if input directory is provided
if [ $# -eq 0 ]; then
    INPUT_BASE="/tmp/tritonbench"  # Default value
else
    INPUT_BASE="$1"  # Use first command line argument
fi

OUTPUT_BASE="/tmp/tritonbench/results"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_BASE"

# Loop through all directories in /tmp/tritonbench
for dir in "$INPUT_BASE"/*/ ; do
    # Get directory name
    dir_name=$(basename "$dir")
    
    # Skip the results directory
    if [ "$dir_name" != "results" ]; then
        echo "Processing directory: $dir_name"
        
        # Create output Excel file path
        output_file="$OUTPUT_BASE/${dir_name}.xlsx"
        
        # Run the Python script
        python ~/ml_scripts/inductor/merge_ops_results.py --input "$dir" --output "$output_file"
    fi
done