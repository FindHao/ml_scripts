#!/bin/bash

cd /home/yhao/d/benchmark

output=/home/yhao/d/testing/runlog_100ms_train_all.txt

func(){
    echo "" > $output

    # for model in fambench_dlrm fambench_xlmr detectron2_maskrcnn vision_maskrcnn timm_efficientnet timm_vision_transformer hf_Bert hf_GPT2 hf_T5 
    for model in detectron2_maskrcnn_r_101_fpn mnasnet1_0 shufflenet_v2_x1_0 BERT_pytorch detectron2_maskrcnn_r_50_c4 mobilenet_v2 soft_actor_critic Background_Matting detectron2_maskrcnn_r_50_fpn mobilenet_v2_quantized_qat speech_transformer LearningToPaint dlrm mobilenet_v3_large squeezenet1_1 Super_SloMo drq moco tacotron2 alexnet fambench_dlrm nvidia_deeprecommender timm_efficientdet attention_is_all_you_need_pytorch fambench_xlmr opacus_cifar10 timm_efficientnet dcgan fastNLP_Bert pplbench_beanmachine timm_nfnet demucs hf_Albert pyhpc_equation_of_state timm_regnet densenet121 hf_Bart pyhpc_isoneutral_mixing timm_resnest detectron2_fasterrcnn_r_101_c4 hf_Bert pyhpc_turbulent_kinetic_energy timm_vision_transformer detectron2_fasterrcnn_r_101_dc5 hf_BigBird pytorch_CycleGAN_and_pix2pix timm_vovnet detectron2_fasterrcnn_r_101_fpn hf_DistilBert pytorch_stargan tts_angular detectron2_fasterrcnn_r_50_c4 hf_GPT2 pytorch_struct vgg16 detectron2_fasterrcnn_r_50_dc5 hf_Longformer pytorch_unet vision_maskrcnn detectron2_fasterrcnn_r_50_fpn hf_Reformer resnet18 yolov3 detectron2_fcos_r_50_fpn hf_T5 resnet50 detectron2_maskrcnn maml resnet50_quantized_qat detectron2_maskrcnn_r_101_c4 maml_omniglot resnext50_32x4d
    do 

    echo "@Yueming Hao: start model tests"  >> $output
    # echo "@Yueming Hao: Run $model" >> $output
    # python run.py -d cuda $model  >> $output 2>&1
    echo "@Yueming Hao: Run $model with dcgm flops" >> $output 2>&1
    python run.py -d cuda -t train  --flops dcgm $model --precision fp32 >> $output 2>&1
    echo "@Yueming Hao: end model tests" >> $output

    done

}

func 

# output=runlog_100ms_all.txt

# func