#!/bin/bash

SHELL_FOLDER=$(
    cd "$(dirname "$0")"
    pwd
)
source ${SHELL_FOLDER}/run_base.sh

cd $tb_path

if [[ -z ${tb_tflops} ]]; then
    tflops=""
else
    tflops="--metrics flops --metrics-gpu-backend dcgm"
    echo "enable dcgm tflops"
fi

export NVIDIA_TF32_OVERRIDE=0

func() {
    for ((i = 1; i <= $max_iter; i++)); do
        # attention: fp32 is default
        python run.py -d cuda ${tflops} -t $mode $model --precision fp32 >>$output 2>&1
    done
}

echo "cuda_env1: $cuda_env1" >>$output
echo "cuda_env2: $cuda_env2" >>$output
echo $(date) >>$output

# for model in timm_nfnet
for model in $all_models; do
    source ${cuda_env1}
    conda activate $env1
    echo "@Yueming Hao origin $model" >>$output
    func

    source ${cuda_env2}
    conda activate $env2
    echo "@Yueming Hao optimize $model" >>$output
    func
done

echo $(date) >>$output

notify
