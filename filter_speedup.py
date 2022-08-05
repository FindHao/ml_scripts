

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

def work_single_model(input_file, wo_tflops=0):
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
        

def work_multi_models(input_file, wo_tflops=0):
    w_gpu = True
    w_tflops = False
    content = ''
    with open(input_file, 'r') as fin:
        content = fin.read()
    content_s = [_ for _ in content.split(
        "@Yueming Hao origin") if _.strip()]
    speedups = {}
    for amodel in content_s:
        model_name = amodel.strip().split()[0].strip()
        amodel_s = [ _  for _ in amodel.split("@Yueming Hao optimize") if _.strip()]
        if len(amodel_s) != 2:
            continue
        origin_raw = amodel_s[0]
        opt_raw = amodel_s[1]
        origin_gpu_time, origin_cpu_time, origin_tflops = filter_time_bs(origin_raw, w_gpu, w_tflops)
        opt_gpu_time, opt_cpu_time, opt_tflops = filter_time_bs(opt_raw, w_gpu, w_tflops )
        speedups[model_name] = [[origin_gpu_time, origin_cpu_time, origin_tflops], [opt_gpu_time, opt_cpu_time, opt_tflops]]
    output_file = '/tmp/speedups.csv'
    table_head = ''
    formatted_speedups = {}
    if w_gpu:
        if w_tflops:
            # origin   | opt  | speedup
            table_head = "model, gpu time, cpu time, tflops, gpu time, cpu time, tflops, gpu speedup, total speedup, tflops speedup\n"
        else:
            table_head = "model, gpu time, cpu time, gpu time, cpu time, gpu speedup, total speedup\n"
    else:
        table_head = "model, cpu time, cpu time, total speedup\n"
    for model in speedups:
        origin = speedups[model][0]
        opt = speedups[model][1]
        gpu_speedup = origin[0]/opt[0] if opt[0] else 1
        cpu_speedup = origin[1] / opt[1] if opt[1] else 1
        flops_speedup = origin[2] / opt[2] if opt[2] else 1
        if w_gpu:
            if w_tflops:
                formatted_speedups[model] = [origin[0], origin[1], origin[2], opt[0], opt[1], opt[2], gpu_speedup, cpu_speedup, flops_speedup]
            else:
                formatted_speedups[model] = [origin[0], origin[1], opt[0], opt[1], gpu_speedup, cpu_speedup]
        else:
            formatted_speedups[model] = [origin[1], opt[1], cpu_speedup]
    with open(output_file, 'w') as fout:
        fout.write(table_head)
        for model in formatted_speedups:
            fout.write("%s, " % model)
            for v in formatted_speedups[model]:
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
        exit(-1)
    for it in results:
        it = [float(_) for _ in it]
        gpu_time.append(it[0])
        cpu_time.append(it[1])
    return mean(gpu_time), mean(cpu_time)

def reg_filter(raw_str, gpu=True, flops=False, bs=False):
    if bs:
        bs_str = " per batch"
        total = ''
    else:
        bs_str = ""
        total = "Total "
    reg_cpu = re.compile(r"CPU Wall Time%s:(.*) milliseconds" % bs_str)
    reg_cpu_gpu = re.compile(
        r"GPU Time%s:(.*) milliseconds\nCPU %sWall Time%s:(.*) milliseconds" % (bs_str, total, bs_str))
    reg_cpu_gpu_flops = re.compile(r"GPU Time%s:(.*) milliseconds\nCPU %sWall Time%s:(.*) milliseconds\nFLOPS:(.*) TFLOPs per second" % (bs_str, total, bs_str))
    reg = reg_cpu
    if flops:
        gpu=True
        reg = reg_cpu_gpu_flops
    elif gpu:
        reg = reg_cpu_gpu
    results = reg.findall(raw_str)
    return results

def filter_time_bs(raw_str, gpu=True, flops=False):
    gpu_time = []
    cpu_time = []
    flops_num = []
    results = reg_filter(raw_str, True, False, False)
    if not results:
        results = reg_filter(raw_str,True, False, True)
        if not results:
            print("error when processing ", raw_str)
            return [0, 0, 0]
    for it in results:
        it = [float(_.strip()) for _ in it]
        if gpu:
            gpu_time.append(it[0])
            cpu_time.append(it[1])
        else:
            cpu_time.append(it[0])
        if flops:
            flops_num.append(it[2])
    mean_gpu = mean(gpu_time) if gpu_time else 0
    mean_cpu = mean(cpu_time) if cpu_time else 0
    mean_flops = mean(flops_num) if flops_num else 0
    return [mean_gpu, mean_cpu, mean_flops]



work_multi_models('/home/yhao/d/tmp/run_all_speedup_aug4.log', wo_tflops=1)
