#!/bin/bash
source ~/.notify.sh
# =================== Configurations ====================
work_path=${work_path:-"/home/users/yhao24/b/p9/pytorch"}
test=${test:-"perf"}
mode=${mode:-"inference"}
var_date=$(date +%Y%m%d_%H%M%S)
single_stream=${single_stream:-0}
log_path=${log_path:-"/mnt/beegfs/users/yhao24/p9/inductor_logs"}
output_file=${log_path}/run_${mode}_${test}_${var_date}.log
conda_dir=${conda_dir:-/mnt/beegfs/users/yhao24/miniconda3}
# env1 is the default environment
env1=${env1:-pt_may25_compiled}
# =================== end Configurations ====================
echo $output_file
source ${conda_dir}/bin/activate
if [ $? -ne 0 ]; then
    echo "can not activate conda"
    exit 1
fi
check_conda_env_exist() {
    if [[ $(conda env list | grep -c "${conda_env}") -eq 0 ]]; then
        echo "Conda environment ${conda_env} does not exist."
        exit 1
    fi
}
check_conda_env_exist $env1
conda activate $env1

cd $work_path

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
if [ $single_stream -eq 1 ]; then
    stream_place_holder="TORCHINDUCTOR_MULTIPLE_STREAMS=0"
else
    stream_place_holder=""
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

start_time=$(date +%s)

for collection in torchbench timm_models huggingface; do
    output_csv_file=${log_path}/${collection}_${mode}_${test_name}${var_date}.csv
    echo "output_csv_file is $output_csv_file" >>$output_file
    ${stream_place_holder} python benchmarks/dynamo/${collection}.py ${test_acc_or_perf} ${precision_place_holder} -dcuda ${mode_place_holder} --inductor --disable-cudagraphs --output ${output_csv_file} >>$output_file 2>&1
done

end_time=$(date +%s)
duration=$((end_time - start_time))
# change time_diff to hours and minutes
duration=$(date -d@$duration -u +%H:%M:%S)
echo "duration is $duration" >>$output_file
notify "${test_name} ${mode} finished, it takes $duration. The log is saved to ${output_file}. log_path is ${log_path}"
