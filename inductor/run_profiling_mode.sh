#!/bin/bash
source ~/.notify.sh
# conda_dir="/scratch/yhao24/miniconda3"
# source ${conda_dir}/bin/activate
# if [ $? -ne 0 ]; then
#     echo "can not activate conda"
#     exit 1
# fi
# conda activate pt_may23_compiled

# export TEMP=/mnt/beegfs/users/yhao24/sys_tmp
# export TMP=/mnt/beegfs/users/yhao24/sys_tmp
# export TMPDIR=/mnt/beegfs/users/yhao24/sys_tmp

# models="hf_Bert BERT_pytorch timm_vision_transformer hf_DistilBert pytorch_struct hf_Bert_large pytorch_CycleGAN_and_pix2pix dlrm densenet121 speech_transformer squeezenet1_1 LearningToPaint  shufflenet_v2_x1_0 fastNLP_Bert hf_Bart lennard_jones resnet152 attention_is_all_you_need_pytorch resnext50_32x4d phlippe_resnet functorch_dp_cifar10 mobilenet_v3_large alexnet mobilenet_v2 hf_GPT2  phlippe_densenet functorch_maml_omniglot timm_resnest timm_efficientnet soft_actor_critic mnasnet1_0 maml_omniglot drq dcgan timm_vovnet hf_Albert timm_regnet hf_Reformer yolov3 resnet18"
models="mobilenetv3_large_100 ghostnet_100 fbnetc_100 fbnetv3_b res2net101_26w_4s levit_128 lcnet_050 twins_pcpvt_base regnety_002 res2net50_14w_8s ese_vovnet19b_dw inception_v3 mnasnet_100 gluon_inception_v3 cspdarknet53 mobilevit_s sebotnet33ts_256 gernet_l tf_efficientnet_b0 tinynet_a spnasnet_100 dpn107 rexnet_100 mobilenetv2_100 repvgg_a2 adv_inception_v3 visformer_small"
# models="resnet50 "
var_date=$(date +%Y%m%d_%H%M%S)

profile_mode_base_path=${profile_mode_base_path:-"/home/users/yhao24/b/tmp/profile_mode"}
pytorch_path=${pytorch_path:-"/scratch/yhao24/p9_inductor/pytorch"}
torchexpert_path=${torchexpert_path:-"/scratch/yhao24/p9_inductor/TorchExpert"}

profile_path="${profile_mode_base_path}/profiles"
stream_path="${profile_mode_base_path}/streams"
updated_stream_path="${profile_mode_base_path}/updated_streams"
model_statistic_path="${profile_mode_base_path}/model_statistic_${var_date}.csv"
mode="training"
function set_precision() {
  if [ "$mode" = "inference" ]; then
    precision="${precision}"
  elif [ "$mode" = "training" ]; then
    precision="amp"
  else
    echo "mode not supported"
  fi
}
log_path="${profile_mode_base_path}/logs"
output_file="${log_path}/output_${var_date}.log"
# collection="torchbench"
collection="timm_models"

# check if profile_mode_base_path exist
if [ ! -d "$profile_mode_base_path" ]; then
  echo "Error: $profile_mode_base_path does not exist"
  exit 1
fi

function create_dir() {
  if [ ! -d "$1" ]; then
    mkdir -p $1
  fi
}
create_dir $profile_path
create_dir $stream_path
create_dir $updated_stream_path
create_dir $log_path

set_precision
start_time=$(date +%s)

echo "log file: ${output_file}"
echo "Profile path: ${profile_path}"
echo "Profile path: ${profile_path}" >>$output_file

work_path=$(dirname $(realpath $0))
for model in $models; do
  export model=$model
  echo "Running $model" >>$output_file
  echo "Running $model"
  model_log_file="${log_path}/${model}_${var_date}.log"
  model_stream_path="${stream_path}/${model}_${var_date}.json"
  cd $pytorch_path
  TORCHINDUCTOR_MULTIPLE_STREAMS_PROFILING=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=1 TORCHINDUCTOR_STREAM_FILE_PATH=${model_stream_path} TORCHINDUCTOR_STREAM_LOG_PATH=${stream_path} python benchmarks/dynamo/${collection}.py --performance --${precision} -dcuda --${mode} --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/${var_date}_multiple --disable-cudagraphs --only ${model} >/dev/null 2>&1
  # check the return value of the last command
  if [ $? -eq 0 ]; then
    echo "Generate profile and stream graph successfully" >>$output_file
  else
    echo "Error: Generate profile and stream graph failed" >>$output_file
    continue
  fi
  # original speedup
  TORCHINDUCTOR_MULTIPLE_STREAMS=0 python benchmarks/dynamo/${collection}.py --performance --${precision} -dcuda --${mode} --inductor --disable-cudagraphs --only ${model} >/dev/null 2>&1
  # check execution success
  if [ $? -eq 0 ]; then
    echo "Run original model successfully" >>$output_file
  else
    echo "Error: Run original model failed" >>$output_file
    continue
  fi
  echo "original speedup" >>$model_log_file
  for i in {1..3}; do
    TORCHINDUCTOR_MULTIPLE_STREAMS=0 python benchmarks/dynamo/${collection}.py --performance --${precision} -dcuda --${mode} --inductor --disable-cudagraphs --only ${model} >>${model_log_file} 2>&1
  done
  echo "=================" >>$model_log_file
  # multiple streams speedup
  TORCHINDUCTOR_MULTIPLE_STREAMS=1 python benchmarks/dynamo/${collection}.py --performance --${precision} -dcuda --${mode} --inductor --disable-cudagraphs --only ${model} >/dev/null
  if [ $? -eq 0 ]; then
    echo "Run multiple streams model successfully" >>$output_file
  else
    echo "Error: Run multiple streams model failed" >>$output_file
    continue
  fi
  echo "multiple streams speedup" >>$model_log_file
  for i in {1..3}; do
    TORCHINDUCTOR_MULTIPLE_STREAMS=1 python benchmarks/dynamo/${collection}.py --performance --${precision} -dcuda --${mode} --inductor --disable-cudagraphs --only ${model} >>${model_log_file} 2>&1
  done
  echo "=================" >>$model_log_file

  cd $work_path
  # search for the generated profile file
  profile_file=$(find ${profile_path} -name "*${model}*")
  if [ -z "$profile_file" ]; then
    echo "Error: Cannot find profile file for $model" >>$output_file
    continue
  fi
  # if there are multiple profile files, use the latest one
  profile_file=$(echo $profile_file | awk '{print $NF}')
  echo "profile file is $profile_file" >>$output_file
  # search for the generated stream graph file
  stream_file=$(find ${stream_path} -name "*${model}*")
  if [ -z "$stream_file" ]; then
    echo "Error: Cannot find stream graph file for $model" >>$output_file
    continue
  fi
  # if there are multiple stream graph files, use the latest one
  stream_file=$(echo $stream_file | awk '{print $NF}')
  echo "stream file is $stream_file" >>$output_file

  updated_stream_file=${updated_stream_path}/${model}_updated.json
  python ${torchexpert_path}/torchexpert.py -i ${profile_file} -s ${stream_file} --export_graph ${updated_stream_file} >>$output_file 2>&1
  if [ $? -eq 0 ]; then
    echo "Generate updated graph successfully" >>$output_file
  else
    echo "Error: Generate updated graph failed" >>$output_file
    continue
  fi
  cd $pytorch_path
  # updated speedup
  TORCHINDUCTOR_MULTIPLE_STREAMS=1 TORCHINDUCTOR_LOAD_EXISTING_STREAM_ASSIGNMENT=${updated_stream_file} python benchmarks/dynamo/${collection}.py --performance --${precision} -dcuda --${mode} --inductor --disable-cudagraphs --only ${model} >/dev/null
  if [ $? -eq 0 ]; then
    echo "Run updated model successfully" >>$output_file
  else
    echo "Error: Run updated model failed" >>$output_file
    continue
  fi
  echo "updated stream speedup" >>$model_log_file
  for i in {1..3}; do
    TORCHINDUCTOR_MULTIPLE_STREAMS=1 TORCHINDUCTOR_LOAD_EXISTING_STREAM_ASSIGNMENT=${updated_stream_file} python benchmarks/dynamo/${collection}.py --performance --${precision} -dcuda --${mode} --inductor --disable-cudagraphs --only ${model} >>${model_log_file} 2>&1
  done
  echo "${model}, success" >>$model_statistic_path
done

end_time=$(date +%s)
duration=$((end_time - start_time))
# change time_diff to hours and minutes
duration=$(date -d@$duration -u +%H:%M:%S)
echo "duration is $duration" >>$output_file
notify "Profiling finished" "duration is $duration" "success"
