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
        python run.py -d cuda ${tflops} -t $mode --metrics cpu_peak_mem,gpu_peak_mem --metrics-gpu-backend ${metrics_gpu_backend} $model   >> $output 2>&1
    done
}

func_torchscript(){
    for (( i = 1 ; i <= $max_iter; i++ ))
    do
        python run.py -d cuda ${tflops} -t $mode $model  --metrics cpu_peak_mem,gpu_peak_mem --metrics-gpu-backend ${metrics_gpu_backend} --backend torchscript  >> $output 2>&1
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

func_torchinductor(){
    for (( i = 1 ; i <= $max_iter; i++ ))
    do
        # python run.py -d cuda ${tflops} -t $mode --metrics cpu_peak_mem,gpu_peak_mem --metrics-gpu-backend dcgm $model   --torchdynamo inductor >> $output 2>&1
        python run.py -d cuda ${tflops} -t $mode --metrics cpu_peak_mem,gpu_peak_mem  --metrics-gpu-backend ${metrics_gpu_backend} $model   --torchdynamo inductor >> $output 2>&1
        if [ $? -ne 0 ]; then
            success_run=0
            break
        fi
    done
}

echo `date` >> $output
for model in $all_models
# for model in hf_Bart hf_Reformer doctr_det_predictor hf_Bert timm_efficientnet timm_resnest dlrm pyhpc_equation_of_state hf_T5_base attention_is_all_you_need_pytorch fambench_xlmr DALLE2_pytorch doctr_reco_predictor hf_T5_large pyhpc_turbulent_kinetic_energy timm_regnet hf_T5 hf_Longformer timm_efficientdet pyhpc_isoneutral_mixing maml timm_vovnet hf_GPT2_large hf_GPT2 hf_Bert_large hf_DistilBert timm_vision_transformer_large hf_Albert demucs timm_nfnet hf_BigBird detectron2_fcos_r_50_fpn timm_vision_transformer drq
do 
conda activate $env1
echo "@Yueming Hao optimize1 $model" >>$output
success_run=1
func_torchinductor
echo "@Yueming Hao origin $model" >>$output
# check if success_run is 1
if [[ $success_run -eq 1 ]]; then
    func
fi

# echo "@Yueming Hao optimize0 $model" >>$output
# func_torchdynamo

# func_fx2trt
# echo "@Yueming Hao optimize2 $model" >>$output
# func_torch_trt

done

echo `date` >> $output

notify