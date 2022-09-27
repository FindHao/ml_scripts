#!/bin/bash




var_date=$(date +'%Y%m%d%H%M')
output=/home/yhao/d/tmp/runall_ncu_$var_date.log
echo "" > $output
cd /home/yhao/d/benchmark
work_dir=/home/yhao/d/tmp/logs_ncu

max_iter=1
func(){
    ncu -f --target-processes all --sampling-buffer-size=536870912 --clock-control none  --csv --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed,sm__cycles_elapsed.avg.per_second,dram__bytes.sum.per_second,sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained,dram__bytes.sum.peak_sustained,dram__cycles_elapsed.avg.per_second --page raw --log-file $work_dir/$model -o $work_dir/$model  python3 run.py -d cuda -t train $model --precision fp32 
    if [ $? -ne 0 ]; then
        echo "Error in $model" >> $output
    fi
}


source /home/yhao/d/conda/bin/activate
conda activate pt_sep14

echo `date` >> $output


# for model in detectron2_maskrcnn_r_101_fpn mnasnet1_0 shufflenet_v2_x1_0 BERT_pytorch detectron2_maskrcnn_r_50_c4 mobilenet_v2 soft_actor_critic Background_Matting detectron2_maskrcnn_r_50_fpn mobilenet_v2_quantized_qat speech_transformer LearningToPaint dlrm mobilenet_v3_large squeezenet1_1 Super_SloMo drq moco tacotron2 alexnet fambench_dlrm nvidia_deeprecommender timm_efficientdet attention_is_all_you_need_pytorch fambench_xlmr opacus_cifar10 timm_efficientnet dcgan fastNLP_Bert pplbench_beanmachine timm_nfnet demucs hf_Albert pyhpc_equation_of_state timm_regnet densenet121 hf_Bart pyhpc_isoneutral_mixing timm_resnest detectron2_fasterrcnn_r_101_c4 hf_Bert pyhpc_turbulent_kinetic_energy timm_vision_transformer detectron2_fasterrcnn_r_101_dc5 hf_BigBird  timm_vovnet detectron2_fasterrcnn_r_101_fpn hf_DistilBert pytorch_stargan tts_angular detectron2_fasterrcnn_r_50_c4 hf_GPT2 pytorch_struct vgg16 detectron2_fasterrcnn_r_50_dc5 hf_Longformer pytorch_unet vision_maskrcnn detectron2_fasterrcnn_r_50_fpn hf_Reformer resnet18 yolov3 detectron2_fcos_r_50_fpn hf_T5 resnet50 detectron2_maskrcnn maml resnet50_quantized_qat detectron2_maskrcnn_r_101_c4 maml_omniglot resnext50_32x4d
for model in mobilenet_v3_large timm_efficientnet_b3 Resnet50 hf_Longformer hf_BigBird Attention_is_all_you_need_pytorch Vgg16 Mobilenet_v2 Timm_efficientdet timm_regnet
do 

    echo "@Yueming Hao origin $model" >>$output
    echo `date` >> $output
    func
    echo `date` >> $output
done

echo `date` >> $output

