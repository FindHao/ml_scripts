#!/bin/bash

SHELL_FOLDER=$(
  cd "$(dirname "$0")"
  pwd
)
source ${SHELL_FOLDER}/run_base.sh
cd ${SHELL_FOLDER}

jsons_path=${jsons_path:-/home/yhao/d/p8/logs/logs_run_all_tflops_profile_train}
# get all model names from jsons_path rather than TorchBench
enable_inplace_models=${enable_inplace_models:-1}
if [ $enable_inplace_models -eq 1 ]; then
  all_models=$(ls ${jsons_path})
fi
conda activate $env1
echo $(date) >>$output
for model in ${all_models}; do
  # will only analyze the latest one
  json_path=${jsons_path}/${model}
  # check if json_path exist
  if [ ! -d $json_path ]; then
    # check if this is a file path
    if [ ! -f $json_path ]; then
      echo "either a folder or file named ${json_path} not exist"
      continue
    fi
  fi
  echo "@Yueming Hao origin $model" >>$output
  python ${torchexpert} -i ${json_path} --model_name $model -o ${log_path}/${prefix_filename}_${var_date}.csv --log_file ${log_path}/${prefix_filename}_${mode}_anslysis_${var_date}.csv >>$output 2>&1
done

echo $(date) >>$output
