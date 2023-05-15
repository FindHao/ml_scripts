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

# get date
date_suffix=$(date +%Y%m%d_%H%M%S)
cd $pytorch_path

if [ "$mode" == "train" ]; then
    mode="--training"
fi
if [ "$mode" == "eval" ]; then
    mode="--inference"
fi

python benchmarks/dynamo/huggingface.py --performance --cold-start-latency $mode  --amp --backend inductor --disable-cudagraphs --device cuda >>$output  2>&1

notify 