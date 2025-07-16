#!/bin/bash

directions=("bwd" "fwd" "fwd_bwd")

precisions=("bf16" "fp32")

# Add start time before the loops
start_time=$(date +%s)

for direction in "${directions[@]}"; do
  for precision in "${precisions[@]}"; do
    echo "Running with direction: $direction, precision: $precision"
    python run_benchmark.py triton \
      --op-collection liger \
      --mode $direction \
      --precision $precision \
      --metrics latency,gpu_peak_mem,speedup,mem_footprint \
      --dump-csv
    mkdir -p /tmp/tritonbench/${direction}_${precision}
    mv /tmp/tritonbench/*.csv /tmp/tritonbench/${direction}_${precision}/
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

notify "TritonBench on dev done (Duration: $duration_str)"
