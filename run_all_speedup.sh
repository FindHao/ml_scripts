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
    done
}

func2(){
    for (( i = 0 ; i <= $max_iter; i++ )) ; do
        python run.py -d cuda ${tflops} -m jit -t $mode $model --precision fp32  >> $output 2>&1
    done
}

echo `date` >> $output
for model in detectron2_maskrcnn_r_101_fpn mnasnet1_0 shufflenet_v2_x1_0 BERT_pytorch detectron2_maskrcnn_r_50_c4 mobilenet_v2 soft_actor_critic Background_Matting detectron2_maskrcnn_r_50_fpn mobilenet_v2_quantized_qat speech_transformer LearningToPaint dlrm mobilenet_v3_large squeezenet1_1 Super_SloMo drq moco tacotron2 alexnet fambench_dlrm nvidia_deeprecommender timm_efficientdet attention_is_all_you_need_pytorch fambench_xlmr opacus_cifar10 timm_efficientnet dcgan fastNLP_Bert pplbench_beanmachine timm_nfnet demucs hf_Albert pyhpc_equation_of_state timm_regnet densenet121 hf_Bart pyhpc_isoneutral_mixing timm_resnest detectron2_fasterrcnn_r_101_c4 hf_Bert pyhpc_turbulent_kinetic_energy timm_vision_transformer detectron2_fasterrcnn_r_101_dc5 hf_BigBird  timm_vovnet detectron2_fasterrcnn_r_101_fpn hf_DistilBert pytorch_stargan tts_angular detectron2_fasterrcnn_r_50_c4 hf_GPT2 pytorch_struct vgg16 detectron2_fasterrcnn_r_50_dc5 hf_Longformer pytorch_unet vision_maskrcnn detectron2_fasterrcnn_r_50_fpn hf_Reformer resnet18 yolov3 detectron2_fcos_r_50_fpn hf_T5 resnet50 detectron2_maskrcnn maml resnet50_quantized_qat detectron2_maskrcnn_r_101_c4 maml_omniglot resnext50_32x4d
# sgd, adam, zerograd
# for model in timm_vision_transformer timm_efficientnet yolov3 timm_resnest moco timm_vovnet timm_regnet vision_maskrcnn fambench_xlmr fambench_dlrm detectron2_maskrcnn timm_nfnet nvidia_deeprecommender dlrm mobilenet_v3_large speech_transformer soft_actor_critic mnasnet1_0 opacus_cifar10 dcgan shufflenet_v2_x1_0 resnext50_32x4d resnet18 attention_is_all_you_need_pytorch mobilenet_v2 hf_DistilBert BERT_pytorch mobilenet_v2_quantized_qat densenet121 hf_Bart resnet50 resnet50_quantized_qat hf_Bert hf_BigBird pytorch_CycleGAN_and_pix2pix hf_Reformer hf_T5 squeezenet1_1 alexnet hf_Albert hf_GPT2 hf_Longformer maml Background_Matting Super_SloMo drq pytorch_stargan pytorch_struct vgg16 maml_omniglot pytorch_unet tts_angular fastNLP_Bert timm_efficientdet
# sgd, adam
# for model in timm_vision_transformer timm_efficientnet yolov3 timm_resnest moco timm_vovnet timm_regnet vision_maskrcnn fambench_xlmr fambench_dlrm detectron2_maskrcnn timm_nfnet nvidia_deeprecommender dlrm mobilenet_v3_large speech_transformer soft_actor_critic mnasnet1_0 opacus_cifar10 dcgan shufflenet_v2_x1_0 resnext50_32x4d resnet18 attention_is_all_you_need_pytorch mobilenet_v2 hf_DistilBert BERT_pytorch mobilenet_v2_quantized_qat densenet121 hf_Bart resnet50 resnet50_quantized_qat hf_Bert hf_BigBird pytorch_CycleGAN_and_pix2pix hf_Reformer hf_T5 squeezenet1_1 alexnet hf_Albert hf_GPT2 hf_Longformer maml Background_Matting Super_SloMo drq pytorch_stargan pytorch_struct vgg16 maml_omniglot
# sgd
# for model in nvidia_deeprecommender vision_maskrcnn timm_efficientnet dlrm timm_vision_transformer moco timm_vovnet fambench_xlmr timm_regnet fambench_dlrm detectron2_maskrcnn timm_nfnet timm_resnest yolov3 
# zero_grad
# for model in tts_angular alexnet detectron2_maskrcnn soft_actor_critic resnet50_quantized_qat fastNLP_Bert drq maml timm_regnet dcgan timm_vision_transformer squeezenet1_1 vgg16 timm_vovnet pytorch_CycleGAN_and_pix2pix opacus_cifar10 resnext50_32x4d yolov3 resnet18 mobilenet_v3_large fambench_xlmr vision_maskrcnn timm_nfnet pytorch_struct mnasnet1_0 fambench_dlrm BERT_pytorch pytorch_unet dlrm nvidia_deeprecommender pytorch_stargan attention_is_all_you_need_pytorch timm_resnest timm_efficientnet speech_transformer resnet50 Background_Matting mobilenet_v2 shufflenet_v2_x1_0 moco mobilenet_v2_quantized_qat Super_SloMo maml_omniglot densenet121
# for model in timm_vovnet opacus_cifar10 resnext50_32x4d yolov3 resnet18 mobilenet_v3_large fambench_xlmr vision_maskrcnn timm_nfnet pytorch_struct mnasnet1_0 fambench_dlrm BERT_pytorch pytorch_unet dlrm nvidia_deeprecommender pytorch_stargan attention_is_all_you_need_pytorch timm_resnest timm_efficientnet speech_transformer resnet50 Background_Matting mobilenet_v2 shufflenet_v2_x1_0 moco mobilenet_v2_quantized_qat Super_SloMo maml_omniglot densenet121
# negative impact
# for model  in detectron2_maskrcnn nvidia_deeprecommender Background_Matting soft_actor_critic mobilenet_v2_quantized_qat pytorch_CycleGAN_and_pix2pix
# for model in pytorch_CycleGAN_and_pix2pix
# len_and_dim_norm
# for model in hf_Reformer
# remaining test aug4
# for model in pyhpc_isoneutral_mixing timm_vision_transformer detectron2_fasterrcnn_r_50_fpn timm_vovnet detectron2_fasterrcnn_r_50_c4 hf_GPT2 hf_Longformer hf_Reformer hf_Bert tts_angular timm_resnest hf_BigBird resnet18 densenet121 timm_regnet detectron2_maskrcnn resnext50_32x4d pyhpc_equation_of_state pytorch_stargan detectron2_fcos_r_50_fpn pytorch_unet detectron2_fasterrcnn_r_101_c4 yolov3 detectron2_fasterrcnn_r_101_dc5 pyhpc_turbulent_kinetic_energy pytorch_struct detectron2_fasterrcnn_r_50_dc5 resnet50 maml hf_Bart demucs hf_Albert maml_omniglot hf_DistilBert vgg16 detectron2_maskrcnn_r_101_c4 hf_T5 vision_maskrcnn resnet50_quantized_qat detectron2_fasterrcnn_r_101_fpn
# table 3
# for model in mobilenet_v3_large  timm_vision_transformer dcgan shufflenet_v2_x1_0 soft_actor_critic opacus_cifar10 timm_efficientnet attention_is_all_you_need_pytorch yolov3 timm_efficientdet mobilenet_v2 mobilenet_v2_quantized_qat densenet121 resnet50_quantized_qat tts_angular hf_Bart hf_DistilBert resnet50 timm_resnest BERT_pytorch moco timm_vovnet hf_Reformer hf_T5 drq detectron2_fasterrcnn_r_101_fpn
# for model in hf_BigBird
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