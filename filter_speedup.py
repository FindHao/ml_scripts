

import argparse
import re
from numpy import mean

def work_multi_models(input_file, output_file):
    content = ''
    with open(input_file, 'r') as fin:
        content = fin.read()
    content_s = [_ for _ in content.split(
        "@Yueming Hao origin") if _.strip()]
    speedups = {}
    for amodel in content_s:
        model_name = amodel.strip().split()[0].strip()
        amodel_s = [_ for _ in amodel.split(
            "@Yueming Hao optimize") if _.strip()]
        if len(amodel_s) != 2:
            continue
        origin_raw = amodel_s[0]
        opt_raw = amodel_s[1]
        origin_results = filter_time_bs(origin_raw)
        if not origin_results:
            print("No original results for %s" % model_name)
            continue
        opt_results = filter_time_bs(opt_raw)
        speedups[model_name] = [origin_results, opt_results]
    first_model = list(speedups.keys())[0]
    table_head = "model, origin cpu time, opt cpu time, total speedup"
    if 'gpu' in speedups[first_model][0]:
        table_head += ", origin gpu time, opt gpu time, gpu speedup"
    if 'tflops' in speedups[first_model][0]:
        table_head += ", origin tflops, opt tflops, tflops speedup"
    if 'cpu_mem' in speedups[first_model][0]:
        table_head += ", origin cpu mem, opt cpu mem, cpu mem speedup"
    if 'gpu_mem' in speedups[first_model][0]:
        table_head += ", origin gpu mem, opt gpu mem, gpu mem speedup"
    table_head += "\n"
    metrics_order = ['cpu', 'gpu', 'tflops', 'cpu_mem', 'gpu_mem']
    with open(output_file, 'w') as fout:
        fout.write(table_head)
        for model in speedups:
            fout.write("%s" % model)
            origin = speedups[model][0]
            opt = speedups[model][1]
            for metric in metrics_order:
                if metric not in origin:
                    continue
                if metric not in opt:
                    print("No metric %s results for %s opt " % (metric, model))
                    opt[metric] = 0
                if metric in ['cpu', 'gpu']:
                    tmp_speedup = origin[metric] / opt[metric] if opt[metric] > 0 else 0
                else:
                    tmp_speedup = opt[metric] / origin[metric] if origin[metric] > 0 else 0
                fout.write(", %.2f, %.2f, %.2f" % (origin[metric], opt[metric], tmp_speedup))
            fout.write("\n")


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
    regs = {
        'cpu': reg_cpu,
        'gpu': reg_gpu,
        'tflops': reg_flops,
        'cpu_mem': reg_cpu_mem,
        'gpu_mem': reg_gpu_mem
    }
    for k in regs:
        result = regs[k].findall(raw_str)
        if result:
            tmp = [float(_.strip()) for _ in result]
            measurements[k] = tmp
    return measurements


def filter_time_bs(raw_str):
    measurements = reg_filter(raw_str)
    if not measurements:
        return None
    mean_results = {}
    for k in measurements:
        mean_results[k] = mean(measurements[k])
    return mean_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='/home/yhao/d/tmp/run_all_speedup_aug4.log')
    parser.add_argument('-o', '--output', type=str,
                        default='/tmp/speedups.csv')
    args = parser.parse_args()
    work_multi_models(args.input, args.output)
