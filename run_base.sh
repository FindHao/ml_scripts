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
echo "tb_path: $tb_path" >> $output
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

all_models="detectron2_fasterrcnn_r_101_dc5 drq hf_GPT2_large mobilenet_v3_large resnet152 timm_nfnet BERT_pytorch detectron2_fasterrcnn_r_101_fpn fambench_xlmr hf_Longformer moco resnet18 timm_regnet Background_Matting detectron2_fasterrcnn_r_50_c4 fastNLP_Bert hf_Reformer nvidia_deeprecommender resnet50 timm_resnest DALLE2_pytorch detectron2_fasterrcnn_r_50_dc5 functorch_dp_cifar10 hf_T5 opacus_cifar10 resnet50_quantized_qat timm_vision_transformer LearningToPaint detectron2_fasterrcnn_r_50_fpn functorch_maml_omniglot hf_T5_base phlippe_resnet resnext50_32x4d timm_vision_transformer_large Super_SloMo detectron2_fcos_r_50_fpn hf_Albert hf_T5_large pyhpc_equation_of_state shufflenet_v2_x1_0 timm_vovnet alexnet detectron2_maskrcnn hf_Bart lennard_jones pyhpc_isoneutral_mixing soft_actor_critic tts_angular attention_is_all_you_need_pytorch detectron2_maskrcnn_r_101_c4 hf_Bert maml pyhpc_turbulent_kinetic_energy speech_transformer vgg16 dcgan detectron2_maskrcnn_r_101_fpn hf_Bert_large maml_omniglot pytorch_CycleGAN_and_pix2pix squeezenet1_1 vision_maskrcnn demucs detectron2_maskrcnn_r_50_c4 hf_BigBird mnasnet1_0 pytorch_stargan tacotron2 yolov3 densenet121 detectron2_maskrcnn_r_50_fpn hf_DistilBert mobilenet_v2 pytorch_struct timm_efficientdet detectron2_fasterrcnn_r_101_c4 dlrm hf_GPT2 mobilenet_v2_quantized_qat pytorch_unet timm_efficientnet"