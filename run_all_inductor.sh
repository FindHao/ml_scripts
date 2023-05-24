#!/bin/bash
# This script is used to print out all guard check logs with TorchInductor
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
source ${SHELL_FOLDER}/run_base.sh

cd $tb_path

profile_suffix=logs_profile_${mode}
if [ $env1 ] && [ $env1 != "pt_sep14" ]; then
    profile_suffix=logs_profile_${mode}_$env1_$(date +'%Y%m%d%H%M')
fi

enable_profile=${enable_profile:-0}
if [ $enable_profile -eq 1 ]; then
    profile_placeholder="--profile --profile-detailed --profile-devices cpu,cuda --profile-folder ${work_path}/${profile_suffix}/$model"
else
    profile_placeholder=""
fi

enable_amp=${enable_amp:-0}
if [ $enable_amp -eq 1 ]; then
    amp_placeholder="--amp"
else
    amp_placeholder=""
fi

max_iter=1
func_torchinductor(){
    for (( i = 1 ; i <= $max_iter; i++ ))
    do
        python run.py -d cuda $profile_placeholder $amp_placeholder -t $mode --metrics none $model --torchdynamo inductor   >> $output 2>&1
    done
}

source $cuda_env1

echo `date` >> $output
conda activate $env1
for model in $all_models
# for model in Super_SloMo tacotron2 timm_efficientdet timm_efficientnet timm_nfnet timm_regnet timm_resnest timm_vision_transformer timm_vision_transformer_large timm_vovnet torchrec_dlrm tts_angular vgg16 vision_maskrcnn yolov3 
do 
echo "@Yueming Hao origin $model" >>$output
func_torchinductor
done
echo `date` >> $output
notify