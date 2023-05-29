#!/bin/bash

SHELL_FOLDER=$(
    cd "$(dirname "$0")"
    pwd
)
# may not need 
export USE_ROCM=0
export USE_NCCL=1
# check run_base exist
if [ ! -f "${SHELL_FOLDER}/../run_base.sh" ]; then
    echo "run_base.sh does not exist."
    exit 1
fi
source ${SHELL_FOLDER}/../run_base.sh

conda activate $env1
source $cuda_env1

cold_start_latency=${cold_start_latency:-0}
if [ $cold_start_latency -eq 1 ]; then
    cold_start_latency_placeholder="--cold-start-latency"
else
    cold_start_latency_placeholder=""
fi
enable_debug=${enable_debug:-0}
if [ $enable_debug -eq 1 ]; then
    debug_placeholder="env TORCH_COMPILE_DEBUG=1"
else
    debug_placeholder=""
fi


# get date
date_suffix=$(date +%Y%m%d_%H%M%S)
cd $pytorch_path

if [ "$mode" == "train" ]; then
    mode="--training"
elif [ "$mode" == "eval" ]; then
    mode="--inference"
fi

${debug_placeholder} python benchmarks/dynamo/huggingface.py --performance ${cold_start_latency_placeholder} $mode  --amp --backend inductor --disable-cudagraphs --device cuda >>$output  2>&1

notify 