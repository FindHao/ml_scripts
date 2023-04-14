"""
This script is used to filter the metric valus, such as execution time, memory usage etc., from the log file.
"""
import argparse
import os
import re
from numpy import mean
import time

def work_multi_models(input_file, output_file):
    content = ''
    with open(input_file, 'r') as fin:
        content = fin.read()
    content_s = [_ for _ in content.split(
        "@Yueming Hao origin") if _.strip()][1:]
    results = {}
    for amodel in content_s:
        model_name = amodel.strip().split()[0].strip()
        # when run tflops and profile for each model, there will be two cpu time and gpu time.
        if amodel.find("Saved TensorBoard Profiler traces") != -1:
            amodel = re.split(r"Running .* method from", amodel)[-1]
        measurements = reg_filter(amodel)
        if not measurements:
            continue
        mean_results = {}
        for k in measurements:
            mean_results[k] = mean(measurements[k])
        results[model_name] = mean_results
    first_model = list(results.keys())[0]
    metrics_order = ['cpu', 'gpu', 'tflops', 'cpu_mem', 'gpu_mem', 'batch_size', 'guard_checks']
    table_head = 'model'
    for metric in metrics_order:
        if metric in results[first_model]:
            table_head += ', %s' % metric
    table_head += '\n'
    
    with open(output_file, 'w') as fout:
        print("writing to file %s" % output_file)
        fout.write(table_head)
        for model in results:
            fout.write("%s, " % model)
            for metric in metrics_order:
                if metric in results[model]:
                    fout.write("%.2f, " % results[model][metric])
            fout.write('\n')

def guard_checks(raw_str):
    from collections import Counter
    reg = re.compile(r"guard_types: (.*),")
    all_results = reg.findall(raw_str)
    if all_results:
        return Counter(all_results)
    return None
        

def reg_filter(raw_str):
    measurements = {}

    def check_bs(tmp_batch_str="", tmp_total_str=""):
        reg_cpu = re.compile(r"CPU %sWall Time%s:(.*) milliseconds" %
                             (tmp_total_str, tmp_batch_str))
        return reg_cpu.findall(raw_str)
    batch_str = " per batch"
    total_str = "Total "
    if not check_bs(tmp_total_str=total_str):
        if not check_bs(tmp_batch_str=batch_str):
            print("error when processing ", raw_str)
            return None
        total_str = ""
    else:
        batch_str = ""
    reg_cpu = re.compile(r"CPU %sWall Time%s:(.*) milliseconds" %
                         (total_str, batch_str))
    reg_gpu = re.compile(r"GPU Time%s:(.*) milliseconds" % batch_str)
    reg_flops = re.compile(r"FLOPS:(.*) TFLOPs per second")
    reg_cpu_mem = re.compile(r"CPU Peak Memory:(.*) GB")
    reg_gpu_mem = re.compile(r"GPU Peak Memory:(.*) GB")
    reg_batch_size = re.compile(r"with input batch size (\d+)")
    regs = {
        'cpu': reg_cpu,
        'gpu': reg_gpu,
        'tflops': reg_flops,
        'cpu_mem': reg_cpu_mem,
        'gpu_mem': reg_gpu_mem,
        'batch_size': reg_batch_size
    }
    
    for k in regs:
        result = regs[k].findall(raw_str)
        if result:
            tmp = [float(_.strip()) for _ in result]
            measurements[k] = tmp
    reg2 = {
        'guard_checks': lambda x: guard_checks(x)
    }
    for k in reg2:
        result = reg2[k](raw_str)
        if result:
            measurements[k] = result

    return measurements


if __name__ == '__main__':
    # get current time and date like 202304141010
    current_time = time.strftime("%Y%m%d%H%M", time.localtime())
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='/home/yhao/d/tmp/run_all_speedup_aug4.log')
    parser.add_argument('-o', '--output', type=str,
                        default='filter_time_%s.csv' % current_time)
    args = parser.parse_args()
    work_multi_models(args.input, args.output)
