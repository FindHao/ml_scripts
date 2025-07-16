#!/bin/bash

cd /home/yhao/d/benchmark/

output=/home/yhao/d/testing/runlog_for_profiling_jul12.txt

work_dir=/home/yhao/d/benchmark

func() {
  echo "" >$output

  # for model in fambench_dlrm fambench_xlmr detectron2_maskrcnn vision_maskrcnn timm_efficientnet timm_vision_transformer hf_Bert hf_GPT2 hf_T5
  # for model in fambench_dlrm fambench_xlmr detectron2_maskrcnn vision_maskrcnn timm_efficientnet timm_vision_transformer hf_Bert hf_GPT2 hf_T5  resnet50 timm_resnest resnext50_32x4d hf_BigBird hf_Bart soft_actor_critic alexnet timm_vovnet mobilenet_v3_large vgg16 shufflenet_v2_x1_0 pytorch_unet dlrm mnasnet1_0 resnet50_quantized_qat tts_angular hf_Reformer nvidia_deeprecommender mobilenet_v2_quantized_qat mobilenet_v2 LearningToPaint hf_Longformer opacus_cifar10 resnet18 timm_regnet dcgan maml BERT_pytorch Super_SloMo pytorch_struct pplbench_beanmachine drq pyhpc_isoneutral_mixing hf_Albert attention_is_all_you_need_pytorch moco Background_Matting pyhpc_turbulent_kinetic_energy maml_omniglot pyhpc_equation_of_state timm_nfnet demucs densenet121 pytorch_CycleGAN_and_pix2pix tacotron2 squeezenet1_1 fastNLP_Bert pytorch_stargan hf_DistilBert speech_transformer yolov3 timm_efficientdet
  for model in dlrm nvidia_deeprecommender; do

    echo "@Yueming Hao: start model tests" >>$output
    echo "@Yueming Hao: Run $model" >>$output
    mkdir ./logs/$model
    python run.py -d cuda --profile --profile-detailed --profile-devices cpu,cuda --profile-folder ./logs/$model/ -t train $model --precision fp32 >>$output 2>&1
    echo "@Yueming Hao: end model tests" >>$output

  done

}

func
