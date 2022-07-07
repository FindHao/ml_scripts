

import re
from numpy import mean

def filter_time(raw_str):
    gpu_time = []
    cpu_time = []
    tflops = []
    reg1 = re.compile(
        r"GPU Time:(.*) milliseconds\nCPU Total Wall Time:(.*) milliseconds\nFLOPS: (.*) TFLOPs per second")
    results = reg1.findall(raw_str)
    for it in results:
        it = [float(_) for _ in it]
        gpu_time.append(it[0])
        cpu_time.append(it[1])
        tflops.append(it[2])
    return mean(gpu_time), mean(cpu_time), mean(tflops)

def work(input_file, wo_tflops=0):
    content = ''
    with open(input_file, 'r') as fin:
        content = fin.read()
    content_s = [_ for _ in content.split(
        "@Yueming Hao optimize") if _.strip()]
    if len(content_s) != 2:
        print("more than 2 item in content_s")
    origin_raw = content_s[0]
    opt_raw = content_s[1]

    if wo_tflops == 1:
        origin_gpu_time, origin_cpu_time,  = filter_time_wo_flops(origin_raw)
        opt_gpu_time, opt_cpu_time = filter_time_wo_flops(opt_raw)
        print("\t, Origin, Optimize, Speedup\nGPU Time:,\t%.2f, %.2f, %.2fX\nCPU Time:,\t%.2f, %.2f, %.2fX" % (origin_gpu_time,
          opt_gpu_time, origin_gpu_time/opt_gpu_time, origin_cpu_time, opt_cpu_time, origin_cpu_time/opt_cpu_time))
    else:
        origin_gpu_time, origin_cpu_time, origin_tflops = filter_time(origin_raw)
        opt_gpu_time, opt_cpu_time, opt_tflops = filter_time(opt_raw)
        print("\t, Origin, Optimize, Speedup\nGPU Time:,\t%.2f, %.2f, %.2fX\nCPU Time:,\t%.2f, %.2f, %.2fX\nTFLOPS:, \t%.2f, %.2f, %.2fX" % (origin_gpu_time,
          opt_gpu_time, origin_gpu_time/opt_gpu_time, origin_cpu_time, opt_cpu_time, origin_cpu_time/opt_cpu_time, origin_tflops, opt_tflops, opt_tflops/origin_tflops))
        




def filter_time_wo_flops(raw_str):
    gpu_time = []
    cpu_time = []
    reg1 = re.compile(
        r"GPU Time per batch:(.*) milliseconds\nCPU Wall Time per batch:(.*) milliseconds")
    results = reg1.findall(raw_str)
    for it in results:
        it = [float(_) for _ in it]
        gpu_time.append(it[0])
        cpu_time.append(it[1])
    return mean(gpu_time), mean(cpu_time)


work('/tmp/run.log', 1)
