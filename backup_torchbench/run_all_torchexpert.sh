#!/bin/bash
SHELL_FOLDER=$(
  cd "$(dirname "$0")"
  pwd
)
source ${SHELL_FOLDER}/run_base.sh
cd ${tb_path}

# export NVIDIA_TF32_OVERRIDE=0

func() {
  for ((i = 1; i <= $max_iter; i++)); do
    # python run.py -d cuda -m jit -t train $model --precision fp32 --torchdynamo nvfuser  >> $output 2>&1
    python run.py -d cuda -t $mode --profile --profile-detailed --profile-devices cpu,cuda --profile-folder /tmp/logs_profile_$mode/$model --torchexpert_output ${work_path}/logs/torch_expert_result_${mode}_${var_date}.log $model >>$output 2>&1
    if [ $? -ne 0 ]; then
      break
    fi
  done
}

conda activate $env1
echo $(date) >>$output

# for model in $all_models
# for model in hf_t5_large
for model in $all_models; do
  echo "@Yueming Hao origin $model" >>$output
  func
done

echo $(date) >>$output

notify
