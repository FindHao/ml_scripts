import json
from threading import currentThread

# {model_name: {}}
results = {}


def work(file_path):
    fin = open(file_path)
    model_list = json.load(fin)
    fin.close()
    fout = open("proper_bs.csv", 'w')
    fout.write("model, 1, 2, 4, 8, 16, 32, 64, 128, best_tflops_bs\n")
    for model in model_list:
        fout.write("%s, " % model['name'])
        latencies = {}
        tflops = {}
        tflops_gaps = [0] * 8
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
                last_tflops = 0
                for batch_size_exp in range(8):
                    batch_size = 2 ** batch_size_exp
                    last_batch_size = 2 ** (batch_size_exp-1)
                    current_tflops = tflops.get(batch_size, 0)
                    fout.write("%.2f, " % current_tflops)
                    tflops_gaps[batch_size_exp] = tflops.get(
                        batch_size, 0) - tflops.get(last_batch_size, 0)
                max_tflops = max(tflops.values())
                best_tflops = 0
                best_tflops_bs = 0
                for batch_size_exp in range(8):
                    batch_size = 2 ** batch_size_exp
                    current_tflops = tflops.get(batch_size, 0)
                    if batch_size_exp != 7:
                        next_tflops = tflops.get(2**(batch_size_exp+1), 0)
                    if current_tflops >= max_tflops*0.99:
                        best_tflops = current_tflops
                        best_tflops_bs = batch_size
                        break
                fout.write("%d, " % best_tflops_bs)
        fout.write("\n")
    fout.close()


work("/home/yhao/d/ml_optimizations/tb-output-eval.json")
