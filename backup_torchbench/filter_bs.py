import json
from threading import currentThread

# {model_name: {}}
results = {}


def work(file_path):
    fin = open(file_path)
    model_list = json.load(fin)
    fin.close()
    fout = open("proper_bs.csv", 'w')
    MAX_EXP = 30
    head_str = ''
    for i in range(MAX_EXP):
        head_str += f"{2**i}, "
    fout.write(f"model, {head_str} best_tflops_bs\n")
    for model in model_list:
        fout.write("%s, " % model['name'])
        latencies = {}
        tflops = {}
        tflops_gaps = [0] * MAX_EXP
        if model['results']['details']:
            for bs in model['results']['details']:
                # latencies[int(bs['batch_size'])] = float(bs['latency_ms'])
                tflops[int(bs['batch_size'])] = float(bs['tflops'])
            # if latencies:
            #     for batch_size_exp in range(8):
            #         batch_size = 2 ** batch_size_exp
            #         fout.write("%.2f, " % latencies.get(batch_size, 0))
            #     best_latency_bs = min(latencies, key=latencies.get)
            #     fout.write("%d, " % best_latency_bs)
            if tflops:
                
                for batch_size_exp in range(MAX_EXP):
                    batch_size = 2 ** batch_size_exp
                    last_batch_size = 2 ** (batch_size_exp-1)
                    current_tflops = tflops.get(batch_size, 0)
                    fout.write("%.2f, " % current_tflops)
                    tflops_gaps[batch_size_exp] = tflops.get(
                        batch_size, 0) - tflops.get(last_batch_size, 0)
                max_tflops = max(tflops.values())
                best_tflops = 0
                best_tflops_bs = 0
                last_tflops = 0
                special = False
                for batch_size_exp in range(MAX_EXP):
                    batch_size = 2 ** batch_size_exp
                    current_tflops = tflops.get(batch_size, 0)
                    if current_tflops != 0 and  last_tflops - current_tflops >= 0.05:
                        special = True
                        break
                    last_tflops = current_tflops
                for batch_size_exp in range(MAX_EXP):
                    batch_size = 2 ** batch_size_exp
                    current_tflops = tflops.get(batch_size, 0)
                    if batch_size_exp != MAX_EXP-1:
                        next_tflops = tflops.get(2**(batch_size_exp+1), 0)
                    if current_tflops >= max_tflops*0.99:
                        best_tflops = current_tflops
                        best_tflops_bs = batch_size
                        break
                fout.write("%d, " % best_tflops_bs)
                if special:
                    fout.write("special ")
        fout.write("\n")
    fout.close()


work("/home/yhao/d/ml_optimizations/tb-output-eval-full.json")
