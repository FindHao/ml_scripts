
# Run all ops.

```
cd tritonbench
/full_path/run_opsonebyone.sh
```
all csv files are saved in `output_dir="/tmp/tritonbench/${DATE_STR}_${direction}_${precision}"`.

# Merge all csv results in one xlsx files
```
python merge_ops_results.py -i /tmp/tritonbench/20241210_144416_bwd_bf16 -o bwd_fp32
```
This script merge all results in `/tmp/tritonbench/20241210_144416_bwd_bf16/bwd_fp32.xlsx`.

# Append summary sheets
```
python summarize_ops_resultsv2.py -i /tmp/tritonbench/20241210_144416_bwd_bf16 -o bwd_fp16
```
The summary will be added to `/tmp/tritonbench/20241210_144416_bwd_bf16/bwd_fp32.xlsx`.
