#!/bin/bash

SHELL_FOLDER=$(
  cd "$(dirname "$0")"
  pwd
)
source ${SHELL_FOLDER}/run_base.sh

cd $tb_path

if [[ -n ${tb_tflops} ]]; then
  tflops="--flops dcgm"
  echo "enable dcgm tflops"
else
  tflops=""
fi

func() {
  for ((i = 1; i <= $max_iter; i++)); do
    python run.py -d cuda ${tflops} -t $mode $model >>$output 2>&1
    if [ $? -ne 0 ]; then
      break
    fi
  done
}

check_conda_env_exist() {
  if [[ $(conda env list | grep -c "${conda_env}") -eq 0 ]]; then
    echo "Conda environment ${conda_env} does not exist."
    exit 1
  fi
}

check_conda_env_exist $env1
check_conda_env_exist $env2

echo $(date) >>$output
for model in $all_models; do
  conda activate $env1
  echo "@Yueming Hao origin $model" >>$output
  func
  conda activate $env2
  echo "@Yueming Hao optimize $model" >>$output
  func

done

echo $(date) >>$output

notify
