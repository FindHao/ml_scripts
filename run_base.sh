#!/bin/bash
# Environment variables:
#   work_path: where benchmark folder locates
#   tb_mode: train, eval
#   tb_path: optional. if you want to specifiy where torchbench locates
#   torchexpert_path: optional. if you want to specify where torchexpert locates
#   tb_conda_dir: where conda locates
#   tb_env1: the conda env you would like to test

var_date=$(date +'%Y%m%d%H%M')
mode=${tb_mode:-train}
#work_path
work_path=${work_path:-/home/yhao/d}
# torchbench path
tb_path=${tb_path:-${work_path}/benchmark}
torchexpert_path=${torchexpert_path:-${work_path}/TorchExpert}
torchexpert=${torchexpert_path}/torchexpert.py
cur_filename=$(basename $0)
prefix_filename=${cur_filename%.*}
logs_path=${work_path}/logs_${prefix_filename}
if [ ! -d $logs_path ]; then
    mkdir -p $logs_path
fi
output=${work_path}/${prefix_filename}_${mode}_${var_date}.log
echo $output
conda_dir=${tb_conda_dir:-/home/yhao/d/conda}
source ${conda_dir}/bin/activate
echo "" > $output
echo `date` >> $output
echo "torchexpert: $torchexpert" >> $output
echo "work_path: $work_path" >> $output
echo "output_csv_file: $output" >> $output
echo "mode: $mode" >> $output
echo "conda_dir: $conda_dir" >> $output

echo "conda envs:" >> $output
env1=${tb_env1:-pt_sep14}
echo "env1" $env1 >> $output
env2=${tb_env2:-pt_sep14_allopt}
echo "env2" $env2 >> $output


notify()
{
    hostname=`cat /proc/sys/kernel/hostname`
    curl -s \
    --form-string "token=${PUSHOVER_API}" \
    --form-string "user=${PUSHOVER_USER_KEY}" \
    --form-string "message=${prefix_filename} on ${hostname} done! " \
    https://api.pushover.net/1/messages.json
}