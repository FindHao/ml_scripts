

import argparse
import os
import re
from numpy import mean


def work_multi_models(input_file, output_file='/tmp/gpuactivetime.csv'):
    """
    only works for gpu active time now
    """
    content = ''
    input_file_path = os.path.abspath(input_file)
    hostname = os.uname()[1]
    with open(input_file_path, 'r') as fin:
        content = fin.read()
    content_s = [_ for _ in content.split(
        "@Yueming Hao origin") if _.strip()]
    timexratio = {}
    for amodel in content_s:
        model_name = amodel.strip().split()[0].strip()
        origin_raw = amodel
        og_gputime, og_gpuratio = filter_time_gpuactive(origin_raw)
        if not og_gputime or not og_gpuratio:
            print(f"Error when process {model_name}")
            continue
        timexratio[model_name] = [og_gputime, og_gpuratio]
    table_head = "model, gpu active time, gpu active time ratio%\n"
    with open(output_file, 'w') as fout:
        fout.write(f"{hostname}\n")
        fout.write(f"input file: {input_file_path}\n")
        fout.write(table_head)
        for model in timexratio:
            fout.write("%s, " % model)
            for v in timexratio[model]:
                fout.write("%.2f, " % v)
            fout.write('\n')


def filter_time_gpuactive(raw_str):
    gpu_active_time = []
    gpu_active_ratio = []
#     GPU active time: 106.609 ms
# GPU active time ratio: 47.93%
    reg1 = re.compile(
        r"GPU active time: (.*) ms\nGPU active time ratio: (.*)\%")
    results = reg1.findall(raw_str)
    if not results:
        return None, None
    for it in results:
        it = [float(_) for _ in it]
        gpu_active_time.append(it[0])
        gpu_active_ratio.append(it[1])
    return mean(gpu_active_time), mean(gpu_active_ratio)


def work_multi_models2(input_file, output_file):
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
    table_head = 'model'
    if 'gpu' in results[first_model]:
        table_head += ', gpu time'
    if 'cpu' in results[first_model]:
        table_head += ', cpu time'
    if 'tflops' in results[first_model]:
        table_head += ', tflops'
    if 'cpu_mem' in results[first_model]:
        table_head += ', cpu mem'
    if 'gpu_mem' in results[first_model]:
        table_head += ', gpu mem'
    table_head += '\n'
    metrics_order = ['cpu', 'gpu', 'tflops', 'cpu_mem', 'gpu_mem']
    with open(output_file, 'w') as fout:
        print("writing to file %s" % output_file)
        fout.write(table_head)
        for model in results:
            fout.write("%s, " % model)
            for metric in metrics_order:
                if metric in results[model]:
                    fout.write("%.2f, " % results[model][metric])
            fout.write('\n')


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='/home/yhao/d/tmp/run_all_speedup_aug4.log')
    parser.add_argument('-o', '--output', type=str,
                        default='/tmp/filter_time.csv')
    args = parser.parse_args()
    work_multi_models2(args.input, args.output)
