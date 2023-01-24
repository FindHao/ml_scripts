#!/bin/bash

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
source ${SHELL_FOLDER}/run_base.sh

cd $tb_path

max_iter=20
if [[ -n ${tb_tflops} ]] ;
then
    tflops="--flops dcgm"
    echo "enable dcgm tflops"
else
    tflops=""
fi

func(){
    for (( i = 1 ; i <= $max_iter; i++ ))
    do
        python run.py -d cuda ${tflops} -t $mode $model --precision fp32  >> $output 2>&1
        if [ $? -ne 0 ]; then
            break
        fi
    done
}

func2(){
    for (( i = 0 ; i <= $max_iter; i++ )) ; do
        python run.py -d cuda ${tflops}  -t $mode $model --precision fp32  >> $output 2>&1
        if [ $? -ne 0 ]; then
            break
        fi
    done
    
}

echo `date` >> $output
for model in $all_models
do 
conda activate $env1
echo "@Yueming Hao origin $model" >>$output
func
conda activate $env2
echo "@Yueming Hao optimize $model" >>$output
func

done

echo `date` >> $output

notify