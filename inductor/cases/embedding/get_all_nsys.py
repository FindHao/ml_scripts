import csv
import os
import subprocess
import sys
from run_copy import benchmark_compiled_module3
from typing import List, Dict

nsys_metrics_to_reports = {
    # the sum of kernel execution time
    "nsys_gpu_kernel_sum": ["nvtx_kern_sum", "nvtx_sum"],
    # the overhead of kernel launch
    "nsys_launch_overhead": ["nvtx_kern_sum", "nvtx_sum"],
    # the names of kernels
    "nsys_kernel_names": ["nvtx_kern_sum"],
    # the durations of kernels
    "nsys_kernel_durations": ["nvtx_kern_sum"],
    # the duration of nvtx range
    "nsys_nvtx_range_duration": ["nvtx_sum"],
    # the number of kernels
    "nsys_num_of_kernels": ["nvtx_kern_sum"],
}

def read_nsys_report(
    report_path: str, required_metrics: List[str]
) -> Dict[str, List[float]]:
    assert os.path.exists(
        report_path
    ), f"The nsys report at {report_path} does not exist."
    reports_required = []
    for metric in required_metrics:
        if metric in nsys_metrics_to_reports:
            reports_required.extend(nsys_metrics_to_reports[metric])
    reports_required = list(set(reports_required))
    assert reports_required, "No nsys reports required"
    cmd = f"nsys stats --report {','.join(reports_required)} --timeunit ns --force-export=true --format csv --output . --force-overwrite=true {report_path}"
    try:
        subprocess.check_call(
            cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to run nsys command: {cmd}\nError: {e}")
        raise e
    # Get the base path and filename without extension
    base_path = os.path.dirname(report_path)
    base_name = os.path.splitext(os.path.basename(report_path))[0]

    results = {}
    csv_contents = {}

    for report in reports_required:
        csv_path = os.path.join(base_path, f"{base_name}_{report}.csv")
        if not os.path.exists(csv_path):
            raise RuntimeError(f"Expected CSV report not found at {csv_path}")

        # Read CSV using DictReader
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            csv_contents[report] = list(reader)
    kernel_duration = []
    kernel_names = []
    sum_kernel_duration = 0
    nvtx_range_duration = 0
    if "nvtx_kern_sum" in csv_contents:
        # gpu kernel execution time summary
        for row in csv_contents["nvtx_kern_sum"]:
            # use ms as the unit
            kernel_duration.append(float(row["Total Time (ns)"]) / 1_000_000)
            kernel_names.append(row["Kernel Name"])
        sum_kernel_duration = sum(kernel_duration)
    if "nvtx_sum" in csv_contents:
        # It is supposed to be only one row. The nvtx range is `:tritonbench_range`
        assert len(csv_contents["nvtx_sum"]) == 1
        nvtx_range_duration = (
            float(csv_contents["nvtx_sum"][0]["Total Time (ns)"]) / 1_000_000
        )

    # Define mapping of metrics to their values. The keys must be in nsys_bench_metrics.
    metrics_map = {
        # Because tritonbench takes the median of numerical values, we need to convert
        # the list of floats to a list of strings.
        "nsys_kernel_durations": [str(duration) for duration in kernel_duration],
        "nsys_kernel_names": kernel_names,
        "nsys_gpu_kernel_sum": sum_kernel_duration,
        "nsys_nvtx_range_duration": nvtx_range_duration,
        "nsys_launch_overhead": nvtx_range_duration - sum_kernel_duration,
        "nsys_num_of_kernels": len(kernel_names),
    }
    # Verify that metrics_map keys match nsys_metrics_to_reports keys
    assert set(metrics_map.keys()) == set(nsys_metrics_to_reports.keys()), (
        f"Mismatch between metrics_map keys and nsys_metrics_to_reports keys.\n"
        f"metrics_map keys: {set(metrics_map.keys())}\n"
        f"nsys_metrics_to_reports keys: {set(nsys_metrics_to_reports.keys())}"
    )
    # Add only requested metrics to results
    results.update(
        {
            metric: metrics_map[metric]
            for metric in required_metrics
            if metric in metrics_map
        }
    )

    return results

def run_benchmark_with_nsys(XBLOCK=128, YBLOCK=128, nwarps=4):
    # First check if the benchmark runs correctly
    print("Checking benchmark correctness...")
    is_correct = benchmark_compiled_module3(XBLOCK=XBLOCK, YBLOCK=YBLOCK, nwarps=nwarps)
    if not is_correct:
        print("Benchmark validation failed. Skipping nsys profiling.")
        return {
            "XBLOCK": XBLOCK,
            "YBLOCK": YBLOCK,
            "nwarps": nwarps,
            "execution_time_ms": "N/A",
            "is_correct": False
        }

    # Continue with nsys profiling if benchmark is correct
    report_base = f"benchmark_report_X{XBLOCK}_Y{YBLOCK}_W{nwarps}"
    report_file = f"{report_base}.nsys-rep"
    
    # Construct the command to run benchmark_compiled_module3
    benchmark_command = (
        f"import torch\n"
        f"from run_copy import benchmark_compiled_module3\n"
        f"benchmark_compiled_module3(XBLOCK={XBLOCK}, YBLOCK={YBLOCK}, nwarps={nwarps})"
    )
    
    # Construct the nsys command with the modified output filename
    nsys_command = [
        "nsys",
        "profile",
        "-c", "cudaProfilerApi",
        "-t", "nvtx,osrt,cuda,cudnn,cublas",
        "--force-overwrite=true",
        "--kill=none",
        "--duration=0",
        "--output", report_base,
        "python",
        "-c",
        benchmark_command
    ]
    
    print("Running nsys profiling command...")
    try:
        result = subprocess.run(nsys_command, check=True, capture_output=True, text=True)
        print("nsys profiling completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"nsys profiling failed with return code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
    
    # Parse the nsys report
    print("Parsing nsys report...")
    required_metrics = ["nsys_kernel_durations", "nsys_kernel_names"]
    report_path = report_file  # Updated to use the modified report filename
    
    metrics = read_nsys_report(report_path, required_metrics)
    
    # Extract the execution time of the specific kernel
    kernel_names = metrics.get("nsys_kernel_names", [])
    kernel_durations = metrics.get("nsys_kernel_durations", [])
    
    target_kernel = "triton_poi_fused_embedding_0"
    execution_time = None
    
    for name, duration in zip(kernel_names, kernel_durations):
        if name == target_kernel:
            execution_time = float(duration)  # In milliseconds
            break
    
    if execution_time is not None:
        print(f"Execution time for kernel '{target_kernel}': {execution_time} ms")
    else:
        print(f"Kernel '{target_kernel}' not found in the nsys report.")
    
    # Get the result of benchmark_compiled_module3
    is_correct = benchmark_compiled_module3(XBLOCK=XBLOCK, YBLOCK=YBLOCK, nwarps=nwarps)
    
    return {
        "XBLOCK": XBLOCK,
        "YBLOCK": YBLOCK,
        "nwarps": nwarps,
        "execution_time_ms": execution_time if execution_time is not None else "N/A",
        "is_correct": is_correct
    }

def write_results_to_csv(results, csv_file='benchmark_results.csv'):
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["XBLOCK", "YBLOCK", "nwarps", "execution_time_ms", "is_correct"])
        
        # If the file does not exist, write the header
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark with nsys and log results to CSV.")
    parser.add_argument("--max_val", type=int, default=4096, help="Maximum value for XBLOCK and YBLOCK")
    parser.add_argument("--csv", type=str, default="benchmark_results.csv", help="CSV file to store results")
    
    args = parser.parse_args()
    
    # Nested loops to explore parameter space
    MAX_VAL = args.max_val
    for XBLOCK in range(1, MAX_VAL, 32):
        for YBLOCK in range(1, MAX_VAL, 32):
            for nwarps in [4, 8]:
                print(f"\nTesting configuration: XBLOCK={XBLOCK}, YBLOCK={YBLOCK}, nwarps={nwarps}")
                # Run the benchmark test
                results = run_benchmark_with_nsys(XBLOCK=XBLOCK, YBLOCK=YBLOCK, nwarps=nwarps)
                
                # Write to CSV
                write_results_to_csv(results, csv_file=args.csv)
    
    print(f"\nAll results written to {args.csv}")
