#!/bin/bash

directions=("bwd" "fwd" "fwd_bwd")

precisions=("bf16" "fp32")

# Add start time before the loops
start_time=$(date +%s)

if [ ! -f "run.py" ]; then
    echo "Error: run.py not found in current directory"
    echo "Please run this script in the root directory of tritonbench"
    exit 1
fi

DATE_STR=$(date +%Y%m%d_%H%M%S)
output_dir="/tmp/tritonbench/${DATE_STR}"
mkdir -p $output_dir

for direction in "${directions[@]}"; do
    for precision in "${precisions[@]}"; do
        echo "Running with direction: $direction, precision: $precision"
        echo "Running: python run.py --op-collection liger --mode $direction --precision $precision --metrics latency,gpu_peak_mem,speedup,mem_footprint_compression_ratio,accuracy  --dump-csv"
        python run.py \
            --op-collection liger \
            --mode $direction \
            --precision $precision \
            --metrics latency,gpu_peak_mem,speedup,mem_footprint_compression_ratio,accuracy \
            --dump-csv --isolate
        mkdir -p $output_dir/${direction}_${precision}
        mv /tmp/tritonbench/*.csv $output_dir/${direction}_${precision}/
    done
done

# Calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

# Format duration string
duration_str=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)
echo "Total execution time: $duration_str"
source ~/.notify.sh
notify "TritonBench on dev done (Duration: $duration_str)"
