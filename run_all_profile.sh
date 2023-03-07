#!/bin/bash

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
source ${SHELL_FOLDER}/run_base.sh

profile_suffix=logs_profile_${mode}
if [ $tb_env1 ] && [ $tb_env1 != "pt_sep14" ]; then
    profile_suffix=logs_profile_${mode}_$env1_$(date +'%Y%m%d%H%M')
fi

enable_inductor=${enable_inductor:-0}

cd $tb_path
func(){
    for (( i = 1 ; i <= $max_iter; i++ ))
    do
        # python run.py -d cuda -m jit -t train $model --precision fp32 --torchdynamo nvfuser  >> $output 2>&1
        python run.py -d cuda -t $mode --profile --profile-detailed --profile-devices cpu,cuda --profile-folder ${work_path}/${profile_suffix}/$model  $model   >> $output 2>&1
        if [ $? -ne 0 ]; then
            break
        fi
    done
}

func_jit(){
    for (( i = 1 ; i <= $max_iter; i++ ))
    do
        python run.py -d cuda -m jit -t $mode --profile --profile-detailed --profile-devices cpu,cuda --profile-folder ${work_path}/${profile_suffix}/$model  $model  --precision fp32  >> $output 2>&1
        if [ $? -ne 0 ]; then
            break
        fi
    done
}

func_inductor(){
    for (( i = 1 ; i <= $max_iter; i++ ))
    do
        python run.py -d cuda -t $mode --profile --profile-detailed --profile-devices cpu,cuda --profile-folder ${work_path}/${profile_suffix}/$model  $model --torchdynamo inductor  >> $output 2>&1
        if [ $? -ne 0 ]; then
            break
        fi
    done
}


conda activate $env1


run_func=func
if [ $enable_jit -eq 1 ]; then
    profile_suffix=${profile_suffix}_jit
    run_func=func_jit
fi

if [ $enable_inductor -eq 1 ]; then
    profile_suffix=${profile_suffix}_inductor
    run_func=func_inductor
fi


echo `date` >> $output
for model in phlippe_densenet resnext50_32x4d hf_Reformer timm_vovnet
# for model in $all_models;
do 
    echo "@Yueming Hao origin $model" >>$output
    ${run_func}
done


echo `date` >> $output

notify