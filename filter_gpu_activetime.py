import argparse
from collections import defaultdict

def mean_wo_max_min(alist):
    # remove the max and min
    alist = sorted(alist)
    return sum(alist[1:-1]) / (len(alist) - 2)

def work_multi_models(input_path, output_path):

    content = ''
    with open(input_path, 'r') as fin:
        content = fin.read()
    content_s = [_ for _ in content.split("\n") if _.strip()]
    raw_logs = defaultdict(list)
    for aline in content_s:
        aline = aline.split(",")
        model_name = aline[0]
        # Model, memcpy, active, busy, total, memcpy ratio, active ratio, busy ratio
        # ignore the first model name and the last average occupancy
        raw_logs[model_name].append(aline[1:-1])
    results = defaultdict(list)
    for model_name, logs in raw_logs.items():
        results[model_name] = []
        for i in range(7):
            tmp = []
            for log in logs:
                tmp.append(float(log[i]))
            results[model_name].append(mean_wo_max_min(tmp))
    
    with open(output_path, 'w') as fout:
        for model_name, log in results.items():
            fout.write(f"{model_name},{','.join([str(_) for _ in log])}\n")
    print("Done! Output file: ", output_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='/home/yhao/d/tmp/run_all_speedup_aug4.log')
    parser.add_argument('-o', '--output', type=str,
                        default='missing_tflops.csv')
    args = parser.parse_args()
    work_multi_models(args.input, args.output)
