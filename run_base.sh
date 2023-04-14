#!/bin/bash
# This is a base script for all other scripts. It contains some common configs and checks.

var_date=$(date +'%Y%m%d%H%M')


# ====config begin======
# mode: train or eval
mode=${mode:-train}
# the main work path
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
output=${work_path}/logs/${prefix_filename}_${mode}_${var_date}.log
conda_dir=${conda_dir:-/home/yhao/d/conda}
# env1 is the default environment
env1=${env1:-pt_jan02}
# env2 is used to speedup comparion scripts like run_all_speedup*
env2=${env2:-pt_jan02_all}
enable_jit=${enable_jit:-0}
cuda_env1=${cuda_env1:-/home/yhao/setenvs/set11.6-cudnn8.3.3.sh}
# cuda_env2 is used to speedup comparion scripts like run_all_speedup_cuda
cuda_env2=${cuda_env2:-/home/yhao/setenvs/set11.6-cudnn8.5.0.sh}
max_iter=${max_iter:-10}
metrics_gpu_backend=${metrics_gpu_backend:-default}
# ====config end======



echo $output
source ${conda_dir}/bin/activate
if [ $? -ne 0 ]; then
    echo "can not activate conda"
    exit 1
fi

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
echo "metrics_gpu_backend: $metrics_gpu_backend" >> $output

# use pushover to notify the end of the script, need extra environment variables PUSHOVER_API PUSHOVER_USER_KEY
notify()
{
    hostname=`cat /proc/sys/kernel/hostname`
    curl -s \
    --form-string "token=${PUSHOVER_API}" \
    --form-string "user=${PUSHOVER_USER_KEY}" \
    --form-string "message=${prefix_filename} on ${hostname} done! Output file is ${output}. " \
    https://api.pushover.net/1/messages.json
}


# list all folder names under torchbenchmark/models
all_models=`ls -d ${tb_path}/torchbenchmark/models/*/ | xargs -n 1 basename`
all_models=`echo $all_models | tr '\n' ' '`


# all_models="detectron2_fasterrcnn_r_101_dc5 drq hf_GPT2_large mobilenet_v3_large resnet152 timm_nfnet BERT_pytorch detectron2_fasterrcnn_r_101_fpn fambench_xlmr hf_Longformer moco resnet18 timm_regnet Background_Matting detectron2_fasterrcnn_r_50_c4 fastNLP_Bert hf_Reformer nvidia_deeprecommender resnet50 timm_resnest DALLE2_pytorch detectron2_fasterrcnn_r_50_dc5 functorch_dp_cifar10 hf_T5 opacus_cifar10 resnet50_quantized_qat timm_vision_transformer LearningToPaint detectron2_fasterrcnn_r_50_fpn functorch_maml_omniglot hf_T5_base phlippe_resnet resnext50_32x4d timm_vision_transformer_large Super_SloMo detectron2_fcos_r_50_fpn hf_Albert hf_T5_large pyhpc_equation_of_state shufflenet_v2_x1_0 timm_vovnet alexnet detectron2_maskrcnn hf_Bart lennard_jones pyhpc_isoneutral_mixing soft_actor_critic tts_angular attention_is_all_you_need_pytorch detectron2_maskrcnn_r_101_c4 hf_Bert maml pyhpc_turbulent_kinetic_energy speech_transformer vgg16 dcgan detectron2_maskrcnn_r_101_fpn hf_Bert_large maml_omniglot pytorch_CycleGAN_and_pix2pix squeezenet1_1 vision_maskrcnn demucs detectron2_maskrcnn_r_50_c4 hf_BigBird mnasnet1_0 pytorch_stargan tacotron2 yolov3 densenet121 detectron2_maskrcnn_r_50_fpn hf_DistilBert mobilenet_v2 pytorch_struct timm_efficientdet detectron2_fasterrcnn_r_101_c4 dlrm hf_GPT2 mobilenet_v2_quantized_qat pytorch_unet timm_efficientnet"