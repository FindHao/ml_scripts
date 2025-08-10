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

echo "Operator,Real_Time,Status" > "$time_log_file"

# 遍历每个操作符并执行命令
for op in "${ops[@]}"; do
    echo "Running benchmark for operator: $op"
    echo "========================================="

    # 为每个操作符创建日志文件
    op_log_file="$log_dir/${op}.log"

    # 使用time命令记录执行时间，捕获错误和执行状态
    if command -v /usr/bin/time >/dev/null 2>&1; then
        # 使用/usr/bin/time提取精确的real time并捕获执行状态
        set +e  # 临时允许命令失败
        if [ "$tritonparse" = "true" ]; then
            cd "$work_dir" && /usr/bin/time -f "%e" "$python_cmd" run.py --op "$op" --num-inputs 50 --tritonparse "$TRITONPARSE_LOGS_DIR/$op" >"$op_log_file" 2>&1
        else
            cd "$work_dir" && /usr/bin/time -f "%e" "$python_cmd" run.py --op "$op" --num-inputs 50 >"$op_log_file" 2>&1
        fi
        exit_code=$?
        cd "$script_dir"  # 切换回脚本目录
        set -e  # 重新启用错误时退出

        if [ $exit_code -eq 0 ]; then
            # 成功执行，从时间输出中提取real time
            real_time=$(tail -n1 "$op_log_file" | grep -o '[0-9]*\.[0-9]*')
            if [ -n "$real_time" ]; then
                echo "Real time: ${real_time}s"
                echo "$op,$real_time,Success" >> "$time_log_file"
                # 清理成功执行的日志文件（只保留时间信息）
                echo "Benchmark completed successfully at $(date)" > "$op_log_file"
                echo "Real time: ${real_time}s" >> "$op_log_file"
            else
                echo "Real time: N/A (parsing failed)"
                echo "$op,N/A,Success" >> "$time_log_file"
            fi
            echo "Status: SUCCESS"
        else
            echo "Real time: N/A (execution failed)"
            echo "Status: FAILED (exit code: $exit_code)"
            echo "$op,N/A,Failed" >> "$time_log_file"
            # 错误日志已经写入到op_log_file中
            echo "Check error details in: $op_log_file"
        fi
    else
        # 使用内置time命令的备用方案
        set +e
        if [ "$tritonparse" = "true" ]; then
            cd "$work_dir" && { time "$python_cmd" run.py --op "$op" --num-inputs 50 --tritonparse "$TRITONPARSE_LOGS_DIR/$op" >"$op_log_file" 2>&1; } 2>&1
        else
            cd "$work_dir" && { time "$python_cmd" run.py --op "$op" --num-inputs 50 >"$op_log_file" 2>&1; } 2>&1
        fi
        exit_code=$?
        cd "$script_dir"  # 切换回脚本目录
        set -e

        if [ $exit_code -eq 0 ]; then
            echo "Status: SUCCESS (time measurement unavailable with built-in time)"
            echo "$op,N/A,Success" >> "$time_log_file"
        else
            echo "Status: FAILED (exit code: $exit_code)"
            echo "$op,N/A,Failed" >> "$time_log_file"
            echo "Check error details in: $op_log_file"
        fi
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
