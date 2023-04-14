#!/bin/bash
# This script is used to print out all guard check logs with TorchInductor
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
source ${SHELL_FOLDER}/run_base.sh

cd $tb_path

max_iter=1
func_torchinductor(){
    for (( i = 1 ; i <= $max_iter; i++ ))
    do
        python run.py -d cuda  -t $mode $model --torchdynamo inductor   >> $output 2>&1
    done
}
echo `date` >> $output
for model in $all_models
do 
conda activate $env1
echo "@Yueming Hao origin $model" >>$output
func_torchinductor
done
echo `date` >> $output
notify