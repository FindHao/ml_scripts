#!/bin/bash

# Script to generate CSV file with directory sizes for each op
# Usage: TARGET_DIR=/path/to/logs ./get_file_size.sh

# Accept directory path from environment variable
TARGET_DIR=${TARGET_DIR:-"."}

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist" >&2
    exit 1
fi

# Output CSV header
echo "op_name,raw_logs_size_bytes,parsed_logs_size_bytes,parsed_to_raw_ratio"

# Function to calculate directory size excluding specific files
calculate_dir_size() {
    local dir="$1"
    local exclude_file="$2"

    if [ -d "$dir" ]; then
        if [ -n "$exclude_file" ]; then
            # Calculate size excluding the specified file
            find "$dir" -type f ! -name "$exclude_file" -exec stat -c%s {} + 2>/dev/null | awk '{sum+=$1} END {print sum+0}'
        else
            # Calculate total size of all files
            find "$dir" -type f -exec stat -c%s {} + 2>/dev/null | awk '{sum+=$1} END {print sum+0}'
        fi
    else
        echo "0"
    fi
}

# Function to calculate ratio with proper handling of division by zero
calculate_ratio() {
    local numerator="$1"
    local denominator="$2"

    if [ "$denominator" -gt 0 ]; then
        # Use awk for floating point arithmetic
        echo "$numerator $denominator" | awk '{printf "%.4f", $1/$2}'
    else
        echo "0.0000"
    fi
}

# Iterate through each subdirectory in the target directory
for op_dir in "$TARGET_DIR"/*; do
    if [ -d "$op_dir" ]; then
        op_name=$(basename "$op_dir")
        raw_logs_dir="$op_dir/raw_logs"
        parsed_logs_dir="$op_dir/parsed_logs"

        # Calculate raw_logs directory size
        raw_size=$(calculate_dir_size "$raw_logs_dir")

        # Calculate parsed_logs directory size (excluding log_file_list.json)
        parsed_size=$(calculate_dir_size "$parsed_logs_dir" "log_file_list.json")

        # Calculate ratio (parsed_size / raw_size)
        ratio=$(calculate_ratio "$parsed_size" "$raw_size")

        # Output CSV row
        echo "$op_name,$raw_size,$parsed_size,$ratio"
    fi
done
