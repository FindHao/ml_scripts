#!/bin/bash

# 设置conda和Python环境
CONDA_PATH="~/miniconda3"
PYTHON_ENV="~/miniconda3/envs/ptd/bin/python"
CONDA_ENV_NAME="ptd"

# 设置默认GPU设备
export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-1}
echo "GPU device set to: ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"

echo "Initializing conda environment..."

# 严格检查并初始化conda
if [[ ! -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: Conda not found at $HOME/miniconda3/etc/profile.d/conda.sh"
    echo "Please ensure conda is properly installed."
    exit 1
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
echo "✓ Conda initialized successfully"

# 严格检查conda命令是否可用
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: Conda command not available after initialization"
    exit 1
fi

# 严格激活conda环境
echo "Activating conda environment: $CONDA_ENV_NAME"
if ! conda activate "$CONDA_ENV_NAME" 2>/dev/null; then
    echo "ERROR: Failed to activate conda environment '$CONDA_ENV_NAME'"
    echo "Please ensure the environment exists: conda env list"
    exit 1
fi
echo "✓ Activated conda environment: $CONDA_ENV_NAME"

# 严格验证Python环境
python_cmd="${PYTHON_ENV/\~/$HOME}"
if [[ ! -f "$python_cmd" ]]; then
    echo "ERROR: Python executable not found at $python_cmd"
    echo "Please check your conda environment installation."
    exit 1
fi

echo "✓ Using Python: $python_cmd"

# 验证Python可以正常执行
if ! "$python_cmd" --version >/dev/null 2>&1; then
    echo "ERROR: Python executable is not working properly"
    exit 1
fi

python_version=$("$python_cmd" --version 2>&1)
echo "✓ Python version: $python_version"

# 读取tritonparse环境变量，默认为false
tritonparse=${TRITONPARSE:-false}
export TRITON_TRACE_GZIP=${TRITON_TRACE_GZIP:-false}

# 设置tritonparse日志目录，可从环境变量读取
TRITONPARSE_LOGS_DIR=${TRITONPARSE_LOGS_DIR:-tritonparse_logs}
echo "Tritonparse logs directory: $TRITONPARSE_LOGS_DIR"

# 读取warmup相关环境变量
WARMUP_ENABLED=${WARMUP_ENABLED:-true}
WARMUP_RUNS=${WARMUP_RUNS:-2}
BENCHMARK_RUNS=${BENCHMARK_RUNS:-10}
echo "Warmup enabled: $WARMUP_ENABLED"
echo "Warmup runs: $WARMUP_RUNS"
echo "Benchmark runs: $BENCHMARK_RUNS"

# 操作符列表
ops=(
    'bf16xint16_gemm'
    'blackwell_attentions'
    'cross_entropy'
    'embedding'
    'flash_attention'
    'flex_attention'
    'fp8_attention'
    'fp8_gemm'
    'fused_linear_cross_entropy'
    'fused_linear_jsd'
    'gather_gemv'
    'gdpa'
    'geglu'
    'gemm'
    'grouped_gemm'
    'int4_gemm'
    'jagged_layer_norm'
    'jagged_mean'
    'jagged_softmax'
    'jagged_sum'
    'jsd'
    'kl_div'
    'launch_latency'
    'layer_norm'
    'low_mem_dropout'
    'mixed_gemm'
    'ragged_attention'
    'rms_norm'
    'rope'
    'softmax'
    'sum'
    'swiglu'
    'template_attention'
    'test_op'
    'vector_add'
    'vector_exp'
    'welford'
)

# 获取脚本所在目录和工作目录
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work_dir="/home/yhao/ptd/tritonbench"

# 创建时间记录文件和日志目录（在脚本目录下）
timestamp=$(date +"%Y%m%d_%H%M%S")
if [ "$tritonparse" = "true" ]; then
    time_log_file="$script_dir/benchmark_times_${timestamp}_tritonparse.csv"
    log_dir="$script_dir/benchmark_logs_${timestamp}_tritonparse"
    echo "========================================="
    echo "Running TRITONPARSE version benchmarks"
    echo "========================================="
else
    time_log_file="$script_dir/benchmark_times_${timestamp}.csv"
    log_dir="$script_dir/benchmark_logs_${timestamp}"
    echo "========================================="
    echo "Running ORIGINAL version benchmarks"
    echo "========================================="
fi

mkdir -p "$log_dir"

echo "Script directory: $script_dir"
echo "Work directory: $work_dir"
echo "Tritonparse mode: $tritonparse"
echo "Time log will be saved to: $time_log_file"
echo "Error logs will be saved to: $log_dir/"

# 检查工作目录是否存在
if [ ! -d "$work_dir" ]; then
    echo "Error: Work directory $work_dir does not exist!"
    exit 1
fi

# 检查run.py是否存在
if [ ! -f "$work_dir/run.py" ]; then
    echo "Error: run.py not found in $work_dir!"
    exit 1
fi

# 公共执行函数
# 参数: $1=操作符名称, $2=是否记录时间(true/false), $3=输出文件路径, $4=标识符(用于tritonparse日志)
run_benchmark() {
    local op="$1"
    local measure_time="$2"
    local output_file="$3"
    local identifier="$4"

    local cmd_prefix=""
    local redirect_output=""

    # 根据是否需要测量时间设置命令前缀
    if [ "$measure_time" = "true" ]; then
        if command -v /usr/bin/time >/dev/null 2>&1; then
            cmd_prefix="/usr/bin/time -f %e"
        fi
    fi

    # 设置输出重定向
    if [ -n "$output_file" ]; then
        redirect_output=">\"$output_file\" 2>&1"
    else
        redirect_output=">/dev/null 2>&1"
    fi

    # 构建完整命令
    if [ "$tritonparse" = "true" ]; then
        full_cmd="cd \"$work_dir\" && $cmd_prefix \"$python_cmd\" run.py --op \"$op\" --num-inputs 50 --tritonparse \"$TRITONPARSE_LOGS_DIR/${op}${identifier}\" $redirect_output"
    else
        full_cmd="cd \"$work_dir\" && $cmd_prefix \"$python_cmd\" run.py --op \"$op\" --num-inputs 50 $redirect_output"
    fi

    # 执行命令
    set +e  # 临时允许命令失败
    eval "$full_cmd"
    local exit_code=$?
    cd "$script_dir"  # 切换回脚本目录
    set -e  # 重新启用错误时退出

    return $exit_code
}

echo "Operator,Min_Time,Max_Time,Mean_Time,Median_Time,Std_Dev,Status" > "$time_log_file"

# 遍历每个操作符并执行命令
for op in "${ops[@]}"; do
    echo "Running benchmark for operator: $op"
    echo "========================================="

    # 为每个操作符创建日志文件
    op_log_file="$log_dir/${op}.log"

    # 执行warmup运行（如果启用）
    if [ "$WARMUP_ENABLED" = "true" ]; then
        echo "Running warmup ($WARMUP_RUNS times)..."
        for ((i=1; i<=WARMUP_RUNS; i++)); do
            echo "  Warmup run $i/$WARMUP_RUNS"
            if ! run_benchmark "$op" "false" "" "_warmup${i}"; then
                echo "  Warning: Warmup run $i failed"
            fi
        done
        echo "Warmup completed. Starting benchmark..."
    fi

    # 执行多次基准测试并收集时间数据
    echo "Running benchmark ($BENCHMARK_RUNS times)..."
    times=()
    failed_runs=0

    for ((i=1; i<=BENCHMARK_RUNS; i++)); do
        echo "  Benchmark run $i/$BENCHMARK_RUNS"

        # 临时文件保存单次运行的输出
        temp_log="$log_dir/${op}_run${i}.log"

        if run_benchmark "$op" "true" "$temp_log" "_run${i}"; then
            # 成功执行，提取时间
            real_time=$(tail -n1 "$temp_log" | grep -o '[0-9]*\.[0-9]*')
            if [ -n "$real_time" ]; then
                times+=("$real_time")
                echo "    Time: ${real_time}s"
            else
                echo "    Warning: Could not parse time from output"
                failed_runs=$((failed_runs + 1))
            fi
        else
            echo "    Failed (keeping error log: $temp_log)"
            failed_runs=$((failed_runs + 1))
        fi
    done

    # 计算统计信息
    if [ ${#times[@]} -gt 0 ]; then
        # 计算最小值、最大值、平均值
        min_time=${times[0]}
        max_time=${times[0]}
        sum=0

        for time in "${times[@]}"; do
            sum=$(echo "$sum + $time" | bc -l)
            if (( $(echo "$time < $min_time" | bc -l) )); then
                min_time=$time
            fi
            if (( $(echo "$time > $max_time" | bc -l) )); then
                max_time=$time
            fi
        done

        mean_time=$(echo "scale=6; $sum / ${#times[@]}" | bc -l)

        # 计算中位数
        sorted_times=($(printf '%s\n' "${times[@]}" | sort -n))
        array_len=${#sorted_times[@]}
        if (( array_len % 2 == 1 )); then
            median_time=${sorted_times[$((array_len/2))]}
        else
            mid1=${sorted_times[$((array_len/2-1))]}
            mid2=${sorted_times[$((array_len/2))]}
            median_time=$(echo "scale=6; ($mid1 + $mid2) / 2" | bc -l)
        fi

        # 计算标准差
        variance_sum=0
        for time in "${times[@]}"; do
            diff=$(echo "$time - $mean_time" | bc -l)
            variance_sum=$(echo "$variance_sum + ($diff * $diff)" | bc -l)
        done
        variance=$(echo "scale=6; $variance_sum / ${#times[@]}" | bc -l)
        std_dev=$(echo "scale=6; sqrt($variance)" | bc -l)

        # 格式化输出（保留6位小数）
        min_time=$(printf "%.6f" "$min_time")
        max_time=$(printf "%.6f" "$max_time")
        mean_time=$(printf "%.6f" "$mean_time")
        median_time=$(printf "%.6f" "$median_time")
        std_dev=$(printf "%.6f" "$std_dev")

        echo "Statistics:"
        echo "  Successful runs: ${#times[@]}/$BENCHMARK_RUNS"
        echo "  Min time: ${min_time}s"
        echo "  Max time: ${max_time}s"
        echo "  Mean time: ${mean_time}s"
        echo "  Median time: ${median_time}s"
        echo "  Std deviation: ${std_dev}s"

        echo "$op,$min_time,$max_time,$mean_time,$median_time,$std_dev,Success" >> "$time_log_file"
        echo "Status: SUCCESS"

        # 创建最终日志文件
        {
            echo "Benchmark completed successfully at $(date)"
            echo "Successful runs: ${#times[@]}/$BENCHMARK_RUNS"
            echo "Min time: ${min_time}s"
            echo "Max time: ${max_time}s"
            echo "Mean time: ${mean_time}s"
            echo "Median time: ${median_time}s"
            echo "Std deviation: ${std_dev}s"
            echo ""
            echo "All times: ${times[*]}"
        } > "$op_log_file"

    else
        echo "All benchmark runs failed!"
        echo "$op,N/A,N/A,N/A,N/A,N/A,Failed" >> "$time_log_file"
        echo "Status: FAILED"
        echo "All benchmark runs failed at $(date)" > "$op_log_file"
    fi

    echo "Completed benchmark for operator: $op"
    echo "========================================="
    echo ""
done

echo "All benchmarks completed!"
echo "Time summary saved to: $time_log_file"

# 显示总体时间统计
if [[ -f "$time_log_file" ]]; then
    echo ""
    echo "=== Timing Summary ==="
    column -t -s',' "$time_log_file"
fi
