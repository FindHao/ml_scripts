#!/bin/bash

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
source ${SHELL_FOLDER}/run_base.sh
conda activate $env1

profile_suffix=logs_profile_${mode}
if [ $tb_env1 ] && [ $tb_env1 != "pt_sep14" ]; then
    profile_suffix=logs_profile_${mode}_$env1
fi

if [ $enable_jit -eq 1 ]; then
    echo "enabled jit" >> $output
    profile_suffix=${profile_suffix}_jit
fi
# you can define a profilesuffix by yourself.
profile_suffix=${tb_profile_suffix:-$profile_suffix}

if [ -d ${work_path}/${profile_suffix} ]; then
    echo "${work_path}/${profile_suffix} exists" >> $output
    echo "${work_path}/${profile_suffix} exists"
else
    echo "${work_path}/${profile_suffix} does not exist" >> $output
    echo "${work_path}/${profile_suffix} does not exist"
    exit 1
fi

output_csv_file=${work_path}/${profile_suffix}/profile_${mode}.csv

echo "Model, memcpy, active, busy, total, memcpy ratio, active ratio, busy ratio, average occupancy" > $output_csv_file

for model in detectron2_maskrcnn_r_101_fpn mnasnet1_0 shufflenet_v2_x1_0 BERT_pytorch detectron2_maskrcnn_r_50_c4 mobilenet_v2 soft_actor_critic Background_Matting detectron2_maskrcnn_r_50_fpn mobilenet_v2_quantized_qat speech_transformer LearningToPaint dlrm mobilenet_v3_large squeezenet1_1 Super_SloMo drq moco tacotron2 alexnet fambench_dlrm nvidia_deeprecommender timm_efficientdet attention_is_all_you_need_pytorch fambench_xlmr opacus_cifar10 timm_efficientnet dcgan fastNLP_Bert pplbench_beanmachine timm_nfnet demucs hf_Albert pyhpc_equation_of_state timm_regnet densenet121 hf_Bart pyhpc_isoneutral_mixing timm_resnest detectron2_fasterrcnn_r_101_c4 hf_Bert pyhpc_turbulent_kinetic_energy timm_vision_transformer detectron2_fasterrcnn_r_101_dc5 hf_BigBird  timm_vovnet detectron2_fasterrcnn_r_101_fpn hf_DistilBert pytorch_stargan tts_angular detectron2_fasterrcnn_r_50_c4 hf_GPT2 pytorch_struct vgg16 detectron2_fasterrcnn_r_50_dc5 hf_Longformer pytorch_unet vision_maskrcnn detectron2_fasterrcnn_r_50_fpn hf_Reformer resnet18 yolov3 detectron2_fcos_r_50_fpn hf_T5 resnet50 detectron2_maskrcnn maml resnet50_quantized_qat detectron2_maskrcnn_r_101_c4 maml_omniglot resnext50_32x4d pytorch_CycleGAN_and_pix2pix
do 
    json_path=${work_path}/${profile_suffix}/$model
    if [ -d $json_path ]; then
        echo "@Yueming Hao origin $model" >>$output
        for json_file in $json_path/*.json
        do
            python3 $torchexpert --json_path $json_file --model_name $model --output_csv_file $output_csv_file  >> $output 2>&1
        done
    fi

done

python3 ${SHELL_FOLDER}/filter_ratios.py -i ${output_csv_file} -o ${work_path}/${profile_suffix}/profile_${mode}_filter.csv


echo `date` >> $output