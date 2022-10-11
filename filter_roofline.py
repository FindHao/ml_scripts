

import argparse
import os
from pathlib import Path

file_name = ''


def filter_from_csv(file_path):
    global file_name
    # sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained
    with open(file_path, 'r') as f:
        content = f.read()
    file_name = Path(file_path).stem
    print(file_name)
    if content.find('"ID","Process ID","Process Name",') < 0:
        print("Invalid csv file")
        return
    
    content = content.split('"ID","Process ID","Process Name",')[1]
    lines = content.splitlines()
    lines[0] = '"ID","Process ID","Process Name",'+lines[0]
    head_line = lines[0]
    head_line = head_line.replace('"', '').split(',')
    kernels = []
    for line in lines[2:]:
        # line = line.replace('"', '').split(',')
        line = line.split('","')
        line[0] = line[0].replace('"', '')
        line[-1] = line[-1].replace('"', '')
        kernel = {}
        for i in range(len(line)):
            kernel[head_line[i]] = line[i]
        kernels.append(kernel)
    return kernels


def compute_roofline(d):
    peak_fp32 = 2 * \
        float(d['sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained'].replace(',', ''))
    achieved_fp32 = 0.0
    achieved_fp32 += float(
        d["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed"].replace(',', ''))
    achieved_fp32 += float(
        d["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed"].replace(',', ''))
    achieved_fp32 += 2 * float(
        d["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed"].replace(',', ''))
    achieved_point_y = achieved_fp32
    achieved_point_x = achieved_fp32 * \
        float(d["sm__cycles_elapsed.avg.per_second"].replace(',', ''))

    roofline_y = float(
        d['sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained'].replace(',', '')) * 2
    roofline_oblique_k = float(
        d['dram__bytes.sum.peak_sustained'].replace(',','')) * float(d['dram__cycles_elapsed.avg.per_second'].replace(',', '')) 
    intersection_x = roofline_y / roofline_oblique_k

    eligible_peak_flops = 0
    compute_bound = True
    if achieved_point_x > intersection_x:
        eligible_peak_flops = roofline_y
    else:
        eligible_peak_flops = achieved_point_x * roofline_oblique_k
        compute_bound = False
    gpu_time_duration = float(d['gpu__time_duration.sum'].replace(',', ''))
    return (compute_bound, eligible_peak_flops, achieved_point_y, peak_fp32, gpu_time_duration)


def work(input_path):
    kernels = filter_from_csv(input_path)
    achieved_flops = []
    eligible_peak_flops = []
    peak_flops = []
    gpu_time_durations = []
    for kernel in kernels:
        compute_bound, eligible_peak_flop, achieved_flop, peak_flop, gpu_time_duration = compute_roofline(
            kernel)
        # if achieved_flop > 0.1:
        achieved_flops.append(achieved_flop)
        eligible_peak_flops.append(eligible_peak_flop)
        peak_flops.append(peak_flop)
        gpu_time_durations.append(gpu_time_duration)
    whole_duration = sum(gpu_time_durations)
    mean_achieved_flops = sum(
        [x*y for x, y in zip(achieved_flops, gpu_time_durations)]) / whole_duration
    mean_eligible_peak_flops = sum(
        [x*y for x, y in zip(eligible_peak_flops, gpu_time_durations)]) / whole_duration
    mean_peak_flops = sum(
        [x*y for x, y in zip(peak_flops, gpu_time_durations)]) / whole_duration
    print("model, Mean achieved, eligible peak, peak flops:\n{}, {}, {}, {}".format(file_name, mean_achieved_flops, mean_eligible_peak_flops, mean_peak_flops))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='/home/yhao24/ncsugdrive/data/logs_run_all_ncu/BERT_pytorch.log')
    args = parser.parse_args()
    work(args.input)