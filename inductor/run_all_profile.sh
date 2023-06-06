#!/bin/bash
source /storage/users/yhao24/.notify.sh
var_date=$(date +'%Y%m%d%H%M')
work_path=${work_path:-/home/users/yhao24/b/p9/logs/merge_trace_train}
log_path=${log_path:-/home/users/yhao24/b/p9/logs}
output=${log_path}/run_all_output_profile_${var_date}.log

conda_dir=${conda_dir:-/home/users/yhao24/b/miniconda3}
env1=${env1:-pt_may25_compiled}
source ${conda_dir}/bin/activate
if [ $? -ne 0 ]; then
    echo "can not activate conda"
    exit 1
fi
echo $output
echo "work_path: $work_path" >> $output 2>&1
echo "log_path: $log_path" >> $output 2>&1
echo "conda_dir: $conda_dir" >> $output 2>&1
echo "env1: $env1" >> $output 2>&1

conda activate $env1


cd $work_path

for model in `ls -d */`
do
    cd ${work_path}/${model}
    echo "run $model" >> $output 2>&1
    all_folders=$(ls -d *__*/)
    for folder in $all_folders
    do
        # check if output_code.py exist
        if [ ! -f "${folder}/output_code.py" ]; then
            echo "output_code.py does not exist in $folder" >> $output 2>&1
        else
            echo "run output_code.py in $folder" >> output_profile_${var_date}.log 2>&1
            python ${folder}/output_code.py --profile >> output_profile_${var_date}.log 2>&1
            folder_name=$(echo ${folder%/} | tr '/' '_')
            mv /tmp/compiled_module_profile.json ${folder_name}_profile.json
        fi
    done
done

notify "run all inductor output_code.py done"