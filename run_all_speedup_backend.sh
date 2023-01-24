#!/bin/bash

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
source ${SHELL_FOLDER}/run_base.sh

cd $tb_path


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
        python run.py -d cuda ${tflops} -t $mode $model   >> $output 2>&1
    done
}

func_torchscript(){
    for (( i = 1 ; i <= $max_iter; i++ ))
    do
        python run.py -d cuda ${tflops} -t $mode $model --backend torchscript  >> $output 2>&1
        if [ $? -ne 0 ]; then
            break
        fi
    done
}

func_fx2trt(){
    for (( i = 1 ; i <= $max_iter; i++ ))
    do
        python run.py -d cuda ${tflops} -t $mode $model --fx2trt  >> $output 2>&1
        if [ $? -ne 0 ]; then
            break
        fi
    done
}

func_torch_trt(){
    for (( i = 1 ; i <= $max_iter; i++ ))
    do
        python run.py -d cuda ${tflops} -t $mode $model --torch_trt >> $output 2>&1
        if [ $? -ne 0 ]; then
            break
        fi
    done
}

func_torchdynamo(){
    for (( i = 1 ; i <= $max_iter; i++ ))
    do
        python run.py -d cuda ${tflops} -t $mode $model --torchdynamo inductor >> $output 2>&1
        if [ $? -ne 0 ]; then
            break
        fi
    done
}

echo `date` >> $output
# for model in $all_models
for model in phlippe_densenet resnext50_32x4d phlippe_resnet mobilenet_v3_large resnet18 timm_nfnet shufflenet_v2_x1_0 timm_resnest mobilenet_v2 mnasnet1_0 timm_vision_transformer resnet152 hf_Albert vgg16 alexnet hf_T5 hf_GPT2 resnet50 Super_SloMo timm_efficientnet fastNLP_Bert hf_Bart hf_DistilBert hf_Bert_large timm_regnet hf_Bert hf_Reformer timm_vovnet
do 
conda activate $env1
echo "@Yueming Hao origin $model" >>$output
func
echo "@Yueming Hao optimize0 $model" >>$output
func_torchdynamo

# func_torchscript
# echo "@Yueming Hao optimize1 $model" >>$output
# func_fx2trt
# echo "@Yueming Hao optimize2 $model" >>$output
# func_torch_trt

done

echo `date` >> $output

notify