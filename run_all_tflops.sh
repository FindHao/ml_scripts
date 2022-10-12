#!/bin/bash

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
source ${SHELL_FOLDER}/run_base.sh

cd $tb_path

max_iter=10
func(){
    for (( i = 1 ; i <= $max_iter; i++ ))
    do
        # python run.py -d cuda -m jit -t train $model --precision fp32 --torchdynamo nvfuser  >> $output 2>&1
        python run.py -d cuda -t $mode --flops dcgm  $model  --precision fp32  >> $output 2>&1
        # error return
        if [ $? -ne 0 ]; then
            break
        fi
    done
}


conda activate $env1
# for model in detectron2_maskrcnn_r_101_fpn mnasnet1_0 shufflenet_v2_x1_0 BERT_pytorch detectron2_maskrcnn_r_50_c4 mobilenet_v2 soft_actor_critic Background_Matting detectron2_maskrcnn_r_50_fpn mobilenet_v2_quantized_qat speech_transformer LearningToPaint dlrm mobilenet_v3_large squeezenet1_1 Super_SloMo drq moco tacotron2 alexnet fambench_dlrm nvidia_deeprecommender timm_efficientdet attention_is_all_you_need_pytorch fambench_xlmr opacus_cifar10 timm_efficientnet dcgan fastNLP_Bert pplbench_beanmachine timm_nfnet demucs hf_Albert pyhpc_equation_of_state timm_regnet densenet121 hf_Bart pyhpc_isoneutral_mixing timm_resnest detectron2_fasterrcnn_r_101_c4 hf_Bert pyhpc_turbulent_kinetic_energy timm_vision_transformer detectron2_fasterrcnn_r_101_dc5 hf_BigBird  timm_vovnet detectron2_fasterrcnn_r_101_fpn hf_DistilBert pytorch_stargan tts_angular detectron2_fasterrcnn_r_50_c4 hf_GPT2 pytorch_struct vgg16 detectron2_fasterrcnn_r_50_dc5 hf_Longformer pytorch_unet vision_maskrcnn detectron2_fasterrcnn_r_50_fpn hf_Reformer resnet18 yolov3 detectron2_fcos_r_50_fpn hf_T5 resnet50 detectron2_maskrcnn maml resnet50_quantized_qat detectron2_maskrcnn_r_101_c4 maml_omniglot resnext50_32x4d pytorch_CycleGAN_and_pix2pix
for model in hf_BigBird mobilenet_v3_large
# for model in detectron2_fcos_r_50_fpn fambench_dlrm fastNLP_Bert maml pplbench_beanmachine pyhpc_equation_of_state pyhpc_isoneutral_mixing pyhpc_turbulent_kinetic_energy pytorch_CycleGAN_and_pix2pix dlrm timm_efficientdet demucs
do 
    echo "@Yueming Hao origin $model" >>$output
    func
done

echo `date` >> $output
notify
