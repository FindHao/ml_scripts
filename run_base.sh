#!/bin/bash
# Environment variables:
#   work_path: where benchmark folder locates
#   tb_mode: train, eval
#   tb_path: optional. if you want to specifiy where torchbench locates
#   torchexpert_path: optional. if you want to specify where torchexpert locates
#   tb_conda_dir: where conda locates
#   tb_env1: the conda env you would like to test

var_date=$(date +'%Y%m%d%H%M')
mode=${mode:-train}
#work_path
work_path=${work_path:-/home/yhao/d}
if [ ! -d $work_path ]; then
    echo "work_path not exist"
    exit 1
fi
# torchbench path
tb_path=${tb_path:-${work_path}/benchmark}
torchexpert_path=${torchexpert_path:-${work_path}/TorchExpert}
torchexpert=${torchexpert_path}/torchexpert.py
cur_filename=$(basename $0)
prefix_filename=${cur_filename%.*}
logs_path=${work_path}/logs/logs_${prefix_filename}
if [ ! -d $logs_path ]; then
    mkdir -p $logs_path
fi
output=${work_path}/logs/${prefix_filename}_${mode}_${var_date}.log
conda_dir=${conda_dir:-/home/yhao/d/conda}
env1=${env1:-pt_oct26}
env2=${env2:-pt_sep14_allopt}
enable_jit=${enable_jit:-0}
cuda_env1=${cuda_env1:-/home/yhao/setenvs/set11.6-cudnn8.3.3.sh}
cuda_env2=${cuda_env2:-/home/yhao/setenvs/set11.6-cudnn8.5.0.sh}


echo $output
source ${conda_dir}/bin/activate
echo "" > $output
echo `date` >> $output
echo "torchexpert: $torchexpert" >> $output
echo "work_path: $work_path" >> $output
echo "output_csv_file: $output" >> $output
echo "mode: $mode" >> $output
echo "conda_dir: $conda_dir" >> $output
echo "conda envs:" >> $output
echo "env1" $env1 >> $output
echo "env2" $env2 >> $output
if [ $enable_jit -eq 1 ]; then
    echo "enable_jit: True" >> $output
fi

notify()
{
    hostname=`cat /proc/sys/kernel/hostname`
    curl -s \
    --form-string "token=${PUSHOVER_API}" \
    --form-string "user=${PUSHOVER_USER_KEY}" \
    --form-string "message=${prefix_filename} on ${hostname} done! " \
    https://api.pushover.net/1/messages.json
}