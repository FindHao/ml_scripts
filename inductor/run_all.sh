#!/bin/bash

# mode=inference test=acc cpp_wrapper=0 log_path=/home/users/yhao24/b/p9/inductor_logs ./run_all.sh;
# mode=inference test=perf cpp_wrapper=0 log_path=/home/users/yhao24/b/p9/inductor_logs ./run_all.sh;
# mode=inference test=perf cpp_wrapper=0 single_stream=1 log_path=/home/users/yhao24/b/p9/inductor_logs ./run_all.sh;



source ~/.notify.sh
# =================== Configurations ====================
work_path=${work_path:-"/home/yhao/p9_clean"}
pt_path=${pt_path:-${work_path}/pytorch}
test=${test:-"perf"}
mode=${mode:-"inference"}
var_date=$(date +%Y%m%d_%H%M%S)
single_stream=${single_stream:-0}
log_path=${log_path:-${work_path}/logs}
output_file=${log_path}/run_${mode}_${test}_${var_date}.log
conda_dir=${conda_dir:-/home/yhao/miniconda3}
cpp_wrapper=${cpp_wrapper:-0}
# debug_flags="TORCH_COMPILE_DEBUG=1 " etc.
debug_flags=${debug_flags:-""}
# env1 is the default environment
env1=${env1:-pt_compiled_clean_for_ms}
STREAMSCHEDULER_REORDER=${STREAMSCHEDULER_REORDER:-0}
TORCHINDUCTOR_BYPASS_TINY=${TORCHINDUCTOR_BYPASS_TINY:-0}
# =================== end Configurations ====================

source ${conda_dir}/bin/activate
if [ $? -ne 0 ]; then
    echo "can not activate conda"
    exit 1
fi
if [ ! -d ${log_path} ]; then
    mkdir ${log_path}
fi
echo $output_file
check_conda_env_exist() {
    if [[ $(conda env list | grep -c "${conda_env}") -eq 0 ]]; then
        echo "Conda environment ${conda_env} does not exist."
        exit 1
    fi
}
check_conda_env_exist $env1
conda activate $env1

cd $pt_path

# check $test, it can only be perf or acc
if [ $test != "perf" ] && [ $test != "acc" ]; then
    echo "test can only be perf or acc"
    exit 1
fi
if [ $test == "perf" ]; then
    test_name="performance"
    test_acc_or_perf="--performance"
else
    test_name="accuracy"
    test_acc_or_perf="--accuracy"
fi
if [ $mode == "inference" ]; then
    precision_place_holder="--bfloat16"
    mode_place_holder="--inference"
else
    precision_place_holder="--amp"
    mode_place_holder="--training"
fi
PREFIX=""
# if debug_flags is not empty, then add it to prefix
if [ ! -z "$debug_flags" ]; then
    PREFIX="${debug_flags}"
fi
if [ $single_stream -eq 1 ]; then
    PREFIX="${PREFIX} TORCHINDUCTOR_MULTIPLE_STREAMS=0"
else
    PREFIX="${PREFIX} TORCHINDUCTOR_MULTIPLE_STREAMS=1"
fi
PREFIX="env ${PREFIX} "

if [ $cpp_wrapper -eq 1 ]; then
    cpp_wrapper_place_holder="--cpp-wrapper"
else
    cpp_wrapper_place_holder=""
fi

if [ $STREAMSCHEDULER_REORDER -eq 1 ]; then
    export STREAMSCHEDULER_REORDER=1
else
    export STREAMSCHEDULER_REORDER=0
fi

if [ $TORCHINDUCTOR_BYPASS_TINY -eq 1 ]; then
    export TORCHINDUCTOR_BYPASS_TINY=1
else
    export TORCHINDUCTOR_BYPASS_TINY=0
fi

echo "work_path is $work_path" >>$output_file
echo "test_name is $test_name" >>$output_file
echo "mode is $mode" >>$output_file
echo "var_date is $var_date" >>$output_file
echo "single_stream is $single_stream" >>$output_file
echo "log_path is $log_path" >>$output_file
echo "output_file is $output_file" >>$output_file
echo "conda_dir is $conda_dir" >>$output_file
echo "env1 is $env1" >>$output_file
echo "cpp_wrapper is $cpp_wrapper" >>$output_file
echo "STREAMSCHEDULER_REORDER is $STREAMSCHEDULER_REORDER" >>$output_file
start_time=$(date +%s)

for collection in torchbench timm_models huggingface; do
    # if single_stream
    if [ $single_stream -eq 1 ]; then
        stream_file_affix="_single_stream"
    else
        stream_file_affix=""
    fi
    output_csv_file=${log_path}/${var_date}_${collection}_${mode}_${test_name}${stream_file_affix}.csv
    echo "output_csv_file is $output_csv_file" >>$output_file
    ${PREFIX} python benchmarks/dynamo/${collection}.py ${test_acc_or_perf} ${cpp_wrapper_place_holder} ${precision_place_holder} -dcuda ${mode_place_holder} --inductor --disable-cudagraphs --output ${output_csv_file} >>$output_file 2>&1
done

end_time=$(date +%s)
duration=$((end_time - start_time))
# change time_diff to hours and minutes
duration=$(date -d@$duration -u +%H:%M:%S)
echo "duration is $duration" >>$output_file
notify "${test_name} ${mode} ${cpp_wrapper_place_holder} finished, it takes $duration. The log is saved to ${output_file}. log_path is ${log_path}"
