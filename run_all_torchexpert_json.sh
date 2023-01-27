#!/bin/bash


SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
source ${SHELL_FOLDER}/run_base.sh

cd $tb_path

jsons_path=${jsons_path:-/home/yhao/d/p8/logs/logs_run_all_tflops_profile_train}
conda activate $env1
echo `date` >> $output
for model in ${all_models}
do 
    # will only analyze the latest one
    json_path=${jsons_path}/${model}/
    # check if json_path exist
    if [ ! -d $json_path ]; then
        echo "json_path ${json_path} not exist"
        continue
    fi
    echo "@Yueming Hao origin $model" >>$output
    python ${torchexpert} --json_path ${json_path} --model_name $model >> $output 2>&1
done

echo `date` >> $output

