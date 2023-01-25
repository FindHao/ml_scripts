#!/bin/bash

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
source ${SHELL_FOLDER}/run_base.sh
# Yueming Hao: @TODO for now, we only support the pure FP32 TFLOPS computation.
export NVIDIA_TF32_OVERRIDE=0

cd $tb_path

func(){
    for (( i = 1 ; i <= $max_iter; i++ ))
    do
        # python run.py -d cuda -m jit -t train $model --precision fp32 --torchdynamo nvfuser  >> $output 2>&1
        python run.py -d cuda -t $mode --metrics flops --metrics-gpu-backend dcgm $model  --precision fp32  >> $output 2>&1
        # error return
        if [ $? -ne 0 ]; then
            break
        fi
    done
}


conda activate $env1

for model in $all_models
do 
    echo "@Yueming Hao origin $model" >>$output
    func
done

echo `date` >> $output
notify
