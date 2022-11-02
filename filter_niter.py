

import argparse
import re
from numpy import mean,std


def work_multi_models(input_file, output_file='/home/yhao/d/tmp/gpuactivetime.csv'):
    """
    only works for gpu active time now
    """
    content = ''
    with open(input_file, 'r') as fin:
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
    reg1 = re.compile(r"GPU active time: (.*) ms\nGPU active time ratio: (.*)\%")
    results = reg1.findall(raw_str)
    if not results:
        return None, None
    for it in results:
        it = [float(_) for _ in it]
        gpu_active_time.append(it[0])
        gpu_active_ratio.append(it[1])
    return mean(gpu_active_time), mean(gpu_active_ratio)



def work_multi_models2(input_file, w_tflops, output_file):
    if w_tflops:
        filter_time_func = filter_time_w_flops
    else:
        filter_time_func = filter_time_wo_flops
    content = ''
    with open(input_file, 'r') as fin:
        content = fin.read()
    content_s = [_ for _ in content.split(
        "@Yueming Hao origin") if _.strip()][1:]
    std_values = {}
    for amodel in content_s:
        model_name = amodel.strip().split()[0].strip()
        niter, cpu_time = filter_time_func(amodel)
        std_value = std(cpu_time)
        if std_value > 1:
            print(f"std value is {std_value:.2f} for {model_name}")
            print(cpu_time)
        std_values[model_name] = std_value
    table_head = 'model, std,'
    if w_tflops:
        table_head += 'tflops,'
    table_head += '\n'
    with open(output_file, 'w') as fout:
        print("writing to file %s" % output_file)
        fout.write(table_head)
        for model in std_values:
            fout.write("%s, " % model)
            fout.write("%.2f, " % std_values[model])
            fout.write('\n')
        pass

def filter_time_wo_flops(raw_str):
    niter = []
    cpu_time = []
    reg1 = re.compile(
        r"niter:(.+?)\n[\s\S]+?CPU Total Wall Time:(.*) milliseconds")
    results = reg1.findall(raw_str)
    reg2 = re.compile(
        r"niter:(.+?)\n[\s\S]+?CPU Wall Time per batch:(.*) milliseconds")
    results2 = reg2.findall(raw_str)
    if not results:
        if not results2:
            print("no results found!")
            return None, None, None
        else:
            results = results2
    for it in results:
        it = [_ for _ in it]
        niter.append(int(it[0]))
        cpu_time.append(float(it[1]))
    return niter, cpu_time


def filter_time_w_flops(raw_str):
    gpu_time = []
    cpu_time = []
    tflops = []
    reg1 = re.compile(
        r"GPU Time:(.*) milliseconds\nCPU Total Wall Time:(.*) milliseconds\nFLOPS:(.*) TFLOPs per second")
    results = reg1.findall(raw_str)
    reg2 = re.compile(r"GPU Time per batch:(.*) milliseconds\nCPU Wall Time per batch:(.*) milliseconds\nFLOPS:(.*) TFLOPs per second")
    results2 = reg2.findall(raw_str)
    if not results:
        if not results2:
            print("no results found!")
            return None, None, None
        else:
            results = results2
    for it in results:
        it = [float(_) for _ in it]
        gpu_time.append(it[0])
        cpu_time.append(it[1])
        tflops.append(it[2])
    return mean(gpu_time), mean(cpu_time), mean(tflops)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='/home/yhao/d/tmp/run_all_speedup_aug4.log')
    parser.add_argument('-t','--w_tflops', type=int, default=0)
    parser.add_argument('-o', '--output', type=str, default='/tmp/filter_time.csv')
    parser.add_argument('-g', '--w_gpu', type=int, default=0)
    args = parser.parse_args()
    work_multi_models2(args.input, args.w_tflops, args.output)