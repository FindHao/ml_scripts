
# Run all ops. All csv results are saved in a folder named with date and time in /tmp/tritonbench/. For different phase and precision, a subfolder is created, such as fwd_bf16.
run_ops.sh 

# Merge all csv results in xlsx files. The input folder is the same as the output folder of run_ops.sh. The output folder is like /tmp/tritonbench/results/20241122_171645. A new folder with a new date and time.
run_merge_ops.sh /tmp/tritonbench/20241122_161645

# summarize all csv files into different excel files. This one is independent of run_merge_ops.sh. It shows the summary of all ops for different phase and precision. Its input is the output folder of run_ops.sh. Its output can be the same output folder of run_merge_ops.sh. All xlsx files generated are named with `_summary`.
summarize_ops_results.py -i /tmp/tritonbench/20241122_161645 -o /tmp/tritonbench/results/20241122_171645
