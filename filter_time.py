

import re
from numpy import mean


def work_multi_models(input_file, w_tflops=False):
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
    output_file = '/tmp/gpuactivetime.csv'
    table_head = "model, gpu time, gpu time ratio%\n"
    with open(output_file, 'w') as fout:
        fout.write(table_head)
        for model in timexratio:
            fout.write("%s, " % model)
            for v in timexratio[model]:
                fout.write("%.2f, " % v)
            fout.write('\n')
        pass

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



def work_multi_models2(input_file, wo_tflops=0):
    w_gpu = True
    w_tflops = False
    content = ''
    with open(input_file, 'r') as fin:
        content = fin.read()
    content_s = [_ for _ in content.split(
        "@Yueming Hao jit") if _.strip()]
    durations = {}
    for amodel in content_s:
        model_name = amodel.strip().split()[0].strip()
        gpu_time, cpu_time = filter_time_wo_flops(amodel)
        if gpu_time is None:
            print(f"Error when process model {model_name}")
            continue
        durations[model_name] = [gpu_time, cpu_time]
    output_file = '/tmp/jittime.csv'
    table_head = 'model, gpu time, cpu time\n'
    with open(output_file, 'w') as fout:
        fout.write(table_head)
        for model in durations:
            fout.write("%s, " % model)
            for v in durations[model]:
                fout.write("%.2f, " % v)
            fout.write('\n')
        pass

def filter_time_wo_flops(raw_str):
    gpu_time = []
    cpu_time = []
    reg1 = re.compile(
        r"GPU Time:(.*) milliseconds\nCPU Total Wall Time:(.*) milliseconds")
    results = reg1.findall(raw_str)
    if not results:
        print("no results found!")
        return None, None
    for it in results:
        it = [float(_) for _ in it]
        gpu_time.append(it[0])
        cpu_time.append(it[1])
    return mean(gpu_time), mean(cpu_time)


work_multi_models2('/home/yhao/d/tmp/run_all_jit_opt.log')
