#!/bin/bash

SHELL_FOLDER=$(
  cd "$(dirname "$0")"
  pwd
)
# may not need
export USE_ROCM=0
export USE_NCCL=1
# check run_base exist
if [ ! -f "${SHELL_FOLDER}/../run_base.sh" ]; then
  echo "run_base.sh does not exist."
  exit 1
fi
source ${SHELL_FOLDER}/../run_base.sh

conda activate $env1
source $cuda_env1

cold_start_latency=${cold_start_latency:-0}
if [ $cold_start_latency -eq 1 ]; then
  cold_start_latency_placeholder="--cold-start-latency"
else
  cold_start_latency_placeholder=""
fi
enable_debug=${enable_debug:-0}
if [ $enable_debug -eq 1 ]; then
  debug_placeholder="env TORCH_COMPILE_DEBUG=1"
else
  debug_placeholder=""
fi

enable_cuda_graphs=${enable_cuda_graphs:-0}
if [ $enable_cuda_graphs -eq 1 ]; then
  cuda_graphs_placeholder=""
else
  cuda_graphs_placeholder="--disable-cudagraphs"
fi

enable_dynamic_shapes=${enable_dynamic_shapes:-0}
if [ $enable_dynamic_shapes -eq 1 ]; then
  dynamic_shapes_placeholder="--dynamic-shapes --dynamic-batch-only"
else
  dynamic_shapes_placeholder=""
fi

enable_profile=${enable_profile:-0}
if [ $enable_profile -eq 1 ]; then
  # check if profile folder is set
  if [ -z "$profile_folder" ]; then
    echo "profile_folder is not set. use default path ~/b/p9/logs_profile_$(date +'%Y%m%d%H%M')"
    BASE_PROFILE_FOLDER=~/b/p9/logs_profile_${mode}_$(date +'%Y%m%d%H%M')
  else
    echo "PROFILE_FOLDER is set to $profile_folder"
    BASE_PROFILE_FOLDER=$profile_folder
  fi
  profile_placeholder="--export-profiler-trace"
else
  profile_placeholder=""
fi

# get date
date_suffix=$(date +%Y%m%d_%H%M%S)
cd $pytorch_path

if [ "$mode" == "train" ]; then
  mode="--training"
elif [ "$mode" == "eval" ]; then
  mode="--inference"
fi

start_time=$(date +%s)

export PROFILE_FOLDER=${BASE_PROFILE_FOLDER}/huggingface/
echo "${debug_placeholder} python benchmarks/dynamo/huggingface.py --performance ${cold_start_latency_placeholder} $mode  --amp --backend inductor ${dynamic_shapes_placeholder} ${cuda_graphs_placeholder}  ${profile_placeholder} --device cuda" >>$output 2>&1
${debug_placeholder} python benchmarks/dynamo/huggingface.py --performance ${cold_start_latency_placeholder} $mode --amp --backend inductor ${dynamic_shapes_placeholder} ${cuda_graphs_placeholder} ${profile_placeholder} --device cuda >>$output 2>&1
mv ${pytorch_path}/speedup_inductor.csv ${pytorch_path}/speedup_inductor_huggingface_${var_date}.csv
mv ${pytorch_path}/speedup_inductor_compilation_metrics.csv ${pytorch_path}/speedup_inductor_compilation_metrics_huggingface_${var_date}.csv
echo "${pytorch_path}/speedup_inductor_huggingface_${var_date}.csv" >>$output 2>&1
echo "${pytorch_path}/speedup_inductor_compilation_metrics_huggingface_${var_date}.csv" >>$output 2>&1

export PROFILE_FOLDER=${BASE_PROFILE_FOLDER}/torchbench/
echo "${debug_placeholder} python benchmarks/dynamo/torchbench.py --performance ${cold_start_latency_placeholder} $mode  --amp --backend inductor ${dynamic_shapes_placeholder} ${cuda_graphs_placeholder}  ${profile_placeholder} --device cuda" >>$output 2>&1
${debug_placeholder} python benchmarks/dynamo/torchbench.py --performance ${cold_start_latency_placeholder} $mode --amp --backend inductor ${dynamic_shapes_placeholder} ${cuda_graphs_placeholder} ${profile_placeholder} --device cuda >>$output 2>&1
mv ${pytorch_path}/speedup_inductor.csv ${pytorch_path}/speedup_inductor_torchbench_${var_date}.csv
mv ${pytorch_path}/speedup_inductor_compilation_metrics.csv ${pytorch_path}/speedup_inductor_compilation_metrics_torchbench_${var_date}.csv
echo "${pytorch_path}/speedup_inductor_torchbench_${var_date}.csv" >>$output 2>&1
echo "${pytorch_path}/speedup_inductor_compilation_metrics_torchbench_${var_date}.csv" >>$output 2>&1

export PROFILE_FOLDER=${BASE_PROFILE_FOLDER}/timm_models/
echo "${debug_placeholder} python benchmarks/dynamo/timm_models.py --performance ${cold_start_latency_placeholder} $mode  --amp --backend inductor ${dynamic_shapes_placeholder} ${cuda_graphs_placeholder}  ${profile_placeholder} --device cuda" >>$output 2>&1
${debug_placeholder} python benchmarks/dynamo/timm_models.py --performance ${cold_start_latency_placeholder} $mode --amp --backend inductor ${dynamic_shapes_placeholder} ${cuda_graphs_placeholder} ${profile_placeholder} --device cuda >>$output 2>&1
mv ${pytorch_path}/speedup_inductor.csv ${pytorch_path}/speedup_inductor_timm_models_${var_date}.csv
mv ${pytorch_path}/speedup_inductor_compilation_metrics.csv ${pytorch_path}/speedup_inductor_compilation_metrics_timm_models_${var_date}.csv
echo "${pytorch_path}/speedup_inductor_timm_models_${var_date}.csv" >>$output 2>&1
echo "${pytorch_path}/speedup_inductor_compilation_metrics_timm_models_${var_date}.csv" >>$output 2>&1

end_time=$(date +%s)
duration=$((end_time - start_time))
notify
