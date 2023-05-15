#!/bin/bash

SHELL_FOLDER=$(
    cd "$(dirname "$0")"
    pwd
)
source ${SHELL_FOLDER}/run_base.sh

cd $tb_path


func() {
    for ((i = 1; i <= $max_iter; i++)); do
        python run.py -d cuda ${tflops} -t $mode $model >>$output 2>&1
        if [ $? -ne 0 ]; then
            break
        fi
    done
}



echo $(date) >>$output
for model in $all_models; do
# for model in resnet50 hf_Bart hf_Bart; do
    conda activate $env1
    echo "@Yueming Hao origin $model" >>$output
    func
done

echo $(date) >>$output

notify
