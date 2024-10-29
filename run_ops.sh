#!/bin/bash

directions=("--bwd" "--fwd" "--fwd_bwd")

precisions=("bf16" "fp32")

for direction in "${directions[@]}"
do
    for precision in "${precisions[@]}"
    do
        echo "Running with direction: $direction, precision: $precision"
        python run_benchmark.py triton \
            --op-collection liger \
            $direction \
            --precision $precision \
            --metrics latency,gpu_peak_mem,speedup,mem_footprint \
            --dump-csv
        mkdir -p /tmp/tritonbench/$direction_$precision
        mv /tmp/tritonbench/*.csv /tmp/tritonbench/$direction_$precision/
    done
done
notify TritonBench on dev done