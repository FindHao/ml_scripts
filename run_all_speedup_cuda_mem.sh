#!/bin/bash

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
source ${SHELL_FOLDER}/run_base.sh

cd $tb_path

max_iter=20
if [[ -z ${tb_tflops} ]] ;
then
    tflops=""
else
    tflops="--metrics flops --metrics-gpu-backend dcgm"
    echo "enable dcgm tflops"
fi

func(){
    for (( i = 1 ; i <= $max_iter; i++ ))
    do
        python run.py -d cuda ${tflops} --metrics cpu_peak_mem,gpu_peak_mem --metrics-gpu-backend dcgm  -t $mode $model --precision fp32  >> $output 2>&1
    done
}


echo `date` >> $output



for model in  detectron2_maskrcnn_r_101_fpn hf_T5_large resnet50 BERT_pytorch detectron2_maskrcnn_r_50_c4 lennard_jones resnet50_quantized_qat Background_Matting detectron2_maskrcnn_r_50_fpn maml resnext50_32x4d DALLE2_pytorch dlrm maml_omniglot shufflenet_v2_x1_0 LearningToPaint drq mnasnet1_0 soft_actor_critic Super_SloMo fambench_xlmr mobilenet_v2 speech_transformer alexnet fastNLP_Bert mobilenet_v2_quantized_qat squeezenet1_1 attention_is_all_you_need_pytorch functorch_dp_cifar10 mobilenet_v3_large tacotron2 dcgan functorch_maml_omniglot moco timm_efficientdet demucs hf_Albert nvidia_deeprecommender timm_efficientnet densenet121 hf_Bart opacus_cifar10 timm_nfnet detectron2_fasterrcnn_r_101_c4 hf_Bert pyhpc_equation_of_state timm_regnet detectron2_fasterrcnn_r_101_dc5 hf_BigBird pyhpc_isoneutral_mixing timm_resnest detectron2_fasterrcnn_r_101_fpn hf_DistilBert pyhpc_turbulent_kinetic_energy timm_vision_transformer detectron2_fasterrcnn_r_50_c4 hf_GPT2 pytorch_CycleGAN_and_pix2pix timm_vision_transformer_large detectron2_fasterrcnn_r_50_dc5 hf_GPT2_large pytorch_stargan timm_vovnet detectron2_fasterrcnn_r_50_fpn hf_Longformer pytorch_struct tts_angular detectron2_fcos_r_50_fpn hf_Reformer pytorch_unet vgg16 detectron2_maskrcnn hf_T5 resnet152 vision_maskrcnn detectron2_maskrcnn_r_101_c4 hf_T5_base resnet18 yolov3  
# for model in timm_nfnet
do 
source ${cuda_env1}
conda activate $env1
echo "@Yueming Hao origin $model" >>$output
func

source ${cuda_env2}
conda activate $env2
echo "@Yueming Hao optimize $model" >>$output
func
done

echo `date` >> $output

notify