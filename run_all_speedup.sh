#!/bin/bash




# model=detectron2_maskrcnn_r_50_c4
output=/tmp/run.log
echo "" > $output
cd /home/yhao/d/benchmark
func(){
    for i in {1..20} ; do
        python run.py -d cuda -t train  $model --precision fp32  >> $output 2>&1
    done
}
# commented models: too long
# pytorch_CycleGAN_and_pix2pix

# for model in detectron2_maskrcnn_r_101_fpn mnasnet1_0 shufflenet_v2_x1_0 BERT_pytorch detectron2_maskrcnn_r_50_c4 mobilenet_v2 soft_actor_critic Background_Matting detectron2_maskrcnn_r_50_fpn mobilenet_v2_quantized_qat speech_transformer LearningToPaint dlrm mobilenet_v3_large squeezenet1_1 Super_SloMo drq moco tacotron2 alexnet fambench_dlrm nvidia_deeprecommender timm_efficientdet attention_is_all_you_need_pytorch fambench_xlmr opacus_cifar10 timm_efficientnet dcgan fastNLP_Bert pplbench_beanmachine timm_nfnet demucs hf_Albert pyhpc_equation_of_state timm_regnet densenet121 hf_Bart pyhpc_isoneutral_mixing timm_resnest detectron2_fasterrcnn_r_101_c4 hf_Bert pyhpc_turbulent_kinetic_energy timm_vision_transformer detectron2_fasterrcnn_r_101_dc5 hf_BigBird  timm_vovnet detectron2_fasterrcnn_r_101_fpn hf_DistilBert pytorch_stargan tts_angular detectron2_fasterrcnn_r_50_c4 hf_GPT2 pytorch_struct vgg16 detectron2_fasterrcnn_r_50_dc5 hf_Longformer pytorch_unet vision_maskrcnn detectron2_fasterrcnn_r_50_fpn hf_Reformer resnet18 yolov3 detectron2_fcos_r_50_fpn hf_T5 resnet50 detectron2_maskrcnn maml resnet50_quantized_qat detectron2_maskrcnn_r_101_c4 maml_omniglot resnext50_32x4d
# sgd
# for model in nvidia_deeprecommender vision_maskrcnn timm_efficientnet dlrm timm_vision_transformer moco timm_vovnet fambench_xlmr timm_regnet fambench_dlrm detectron2_maskrcnn timm_nfnet timm_resnest yolov3 
# zero_grad
for model in tts_angular alexnet detectron2_maskrcnn soft_actor_critic resnet50_quantized_qat fastNLP_Bert drq maml timm_regnet dcgan timm_vision_transformer squeezenet1_1 vgg16 timm_vovnet pytorch_CycleGAN_and_pix2pix opacus_cifar10 resnext50_32x4d yolov3 resnet18 mobilenet_v3_large fambench_xlmr vision_maskrcnn timm_nfnet pytorch_struct mnasnet1_0 fambench_dlrm BERT_pytorch pytorch_unet dlrm nvidia_deeprecommender pytorch_stargan attention_is_all_you_need_pytorch timm_resnest timm_efficientnet speech_transformer resnet50 Background_Matting mobilenet_v2 shufflenet_v2_x1_0 moco mobilenet_v2_quantized_qat Super_SloMo maml_omniglot densenet121
do 

source /home/yhao/d/conda/bin/activate
conda activate pt
echo "@Yueming Hao origin $model" >>$output
func
conda activate opt_zerograd
echo "@Yueming Hao optimize $model" >>$output
func

done

