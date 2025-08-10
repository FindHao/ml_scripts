#!/bin/bash

# 操作符列表
ops=(
    'addmm'
    'bf16xint16_gemm'
    'blackwell_attentions'
    'cross_entropy'
    'decoding_attention'
    'embedding'
    'flash_attention'
    'flex_attention'
    'fp8_attention'
    'fp8_fused_quant_gemm_rowwise'
    'fp8_gemm'
    'fp8_gemm_blockwise'
    'fp8_gemm_rowwise'
    'fp8_gemm_rowwise_grouped'
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
time_log_file="$script_dir/benchmark_times_${timestamp}.log"
log_dir="$script_dir/benchmark_logs_${timestamp}"
mkdir -p "$log_dir"

echo "Script directory: $script_dir"
echo "Work directory: $work_dir"
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
        cd "$work_dir" && /usr/bin/time -f "%e" python run.py --op "$op" --num-inputs 50 >"$op_log_file" 2>&1
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
        cd "$work_dir" && { time python run.py --op "$op" --num-inputs 50 >"$op_log_file" 2>&1; } 2>&1
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
