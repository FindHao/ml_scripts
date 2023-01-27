#!/bin/bash

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
source ${SHELL_FOLDER}/run_base.sh
# Yueming Hao: @TODO for now, we only support the pure FP32 TFLOPS computation.
export NVIDIA_TF32_OVERRIDE=0

cd $tb_path

func(){
    python run.py  -d cuda -t $mode --profile --profile-detailed  --profile-folder ${logs_path}/$model $model  --precision fp32 >> $output 2>&1
    # error return
    if [ $? -ne 0 ]; then
        return
    fi
    python run.py -d cuda -t $mode --metrics flops --metrics-gpu-backend dcgm --export-metrics $model  --precision fp32  >> $output 2>&1
    # error return
    if [ $? -ne 0 ]; then
        return
    fi
    mv ${model}*metrics.csv ${logs_path}/
}


conda activate $env1

for model in $all_models
do 
    echo "@Yueming Hao origin $model" >>$output
    func
done


echo `date` >> $output
notify
