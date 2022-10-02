#!/bin/bash
# Environment variables:
#   work_path: where benchmark folder locates
#   tb_mode: train, eval
#   tb_path: optional. if you want to specifiy where torchbench locates
#   torchexpert_path: optional. if you want to specify where torchexpert locates
#   tb_conda_dir: where conda locates




var_date=$(date +'%Y%m%d%H%M')
mode=${tb_mode:-train}
#work_path
work_path=${work_path:-/home/yhao/d}
# torchbench path
tb_path=${tb_path:-${work_path}/benchmark}
torchexpert_path=${torchexpert_path:-${work_path}/TorchExpert}
torchexpert=${torchexpert_path}/torchexpert.py
cur_filename=$(basename $0)
prefix_filename=${cur_filename%.*}
logs_path=${work_path}/logs_${prefix_filename}
if [ ! -d $logs_path ]; then
    mkdir -p $logs_path
fi
output=${work_path}/${prefix_filename}_${mode}_${var_date}.log
echo $output
conda_dir=${tb_conda_dir:-/home/yhao/d/conda}
source ${conda_dir}/bin/activate
conda activate pt_sep14
echo "" > $output
echo `date` >> $output
echo "torchexpert: $torchexpert" >> $output
echo "work_path: $work_path" >> $output
echo "output_csv_file: $output" >> $output
echo "mode: $mode" >> $output
echo "conda_dir: $conda_dir" >> $output

cd $tb_path

max_iter=1
func(){
    ncu -f --target-processes all --sampling-buffer-size=536870912 --clock-control none  --csv --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed,sm__cycles_elapsed.avg.per_second,dram__bytes.sum.per_second,sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained,dram__bytes.sum.peak_sustained,dram__cycles_elapsed.avg.per_second --page raw --log-file $logs_path/${model}.log -o $logs_path/$model  python3 run.py -d cuda -t train $model --precision fp32 
    echo $model $mode 
    if [ $? -ne 0 ]; then
        echo "Error in $model" >> $output
    fi
}


# for model in detectron2_maskrcnn_r_101_fpn mnasnet1_0 shufflenet_v2_x1_0 BERT_pytorch detectron2_maskrcnn_r_50_c4 mobilenet_v2 soft_actor_critic Background_Matting detectron2_maskrcnn_r_50_fpn mobilenet_v2_quantized_qat speech_transformer LearningToPaint dlrm mobilenet_v3_large squeezenet1_1 Super_SloMo drq moco tacotron2 alexnet fambench_dlrm nvidia_deeprecommender timm_efficientdet attention_is_all_you_need_pytorch fambench_xlmr opacus_cifar10 timm_efficientnet dcgan fastNLP_Bert pplbench_beanmachine timm_nfnet demucs hf_Albert pyhpc_equation_of_state timm_regnet densenet121 hf_Bart pyhpc_isoneutral_mixing timm_resnest detectron2_fasterrcnn_r_101_c4 hf_Bert pyhpc_turbulent_kinetic_energy timm_vision_transformer detectron2_fasterrcnn_r_101_dc5 hf_BigBird  timm_vovnet detectron2_fasterrcnn_r_101_fpn hf_DistilBert pytorch_stargan tts_angular detectron2_fasterrcnn_r_50_c4 hf_GPT2 pytorch_struct vgg16 detectron2_fasterrcnn_r_50_dc5 hf_Longformer pytorch_unet vision_maskrcnn detectron2_fasterrcnn_r_50_fpn hf_Reformer resnet18 yolov3 detectron2_fcos_r_50_fpn hf_T5 resnet50 detectron2_maskrcnn maml resnet50_quantized_qat detectron2_maskrcnn_r_101_c4 maml_omniglot resnext50_32x4d
for model in Vgg16 Mobilenet_v2 Timm_efficientdet timm_regnet
do 

    echo "@Yueming Hao origin $model" >>$output
    echo `date` >> $output
    func
    echo `date` >> $output
done

echo `date` >> $output

