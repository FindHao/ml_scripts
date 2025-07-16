#!/bin/bash

SHELL_FOLDER=$(
  cd "$(dirname "$0")"
  pwd
)
source ${SHELL_FOLDER}/run_base.sh

profile_suffix=logs_profile_${mode}_$(date +'%Y%m%d%H%M')

max_iter=1
cd $tb_path
func() {
  for ((i = 1; i <= $max_iter; i++)); do
    # python run.py -d cuda -m jit -t train $model --precision fp32 --torchdynamo nvfuser  >> $output 2>&1
    python run.py -d cuda -t $mode ${jit_placeholder} ${amp_placeholder} --profile --profile-detailed --profile-devices cpu,cuda --profile-folder ${work_path}/${profile_suffix}/$model $model ${inductor_placeholder} >>$output 2>&1
    if [ $? -ne 0 ]; then
      break
    fi
  done
}

conda activate $env1

if [ $enable_jit -eq 1 ]; then
  profile_suffix=${profile_suffix}_jit
fi

if [ $enable_inductor -eq 1 ]; then
  profile_suffix=${profile_suffix}_inductor
fi
echo "profile suffix: $profile_suffix" >>$output
echo $(date) >>$output
for model in timm_efficientdet; do # for model in $all_models
  echo "@Yueming Hao origin $model" >>$output
  func
done

echo $(date) >>$output

notify
