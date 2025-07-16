#!/bin/bash

SHELL_FOLDER=$(
  cd "$(dirname "$0")"
  pwd
)
source ${SHELL_FOLDER}/run_base.sh

cd $tb_path

func() {
  for ((i = 1; i <= $max_iter; i++)); do
    # python run.py -d cuda -m jit -t train $model --precision fp32 --torchdynamo nvfuser  >> $output 2>&1
    python run.py -d cuda -t $mode --metrics cpu_peak_mem,gpu_peak_mem --metrics-gpu-backend dcgm $model >>$output 2>&1
    # error return
    if [ $? -ne 0 ]; then
      break
    fi
  done
}

conda activate $env1

for model in $all_models; do
  echo "@Yueming Hao origin $model" >>$output
  func
done

echo $(date) >>$output
notify
