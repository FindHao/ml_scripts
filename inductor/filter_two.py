"""
This script is used to filter speedup by torch inductor
"""

"""
This script is used to filter the metric valus, such as execution time, memory usage etc., from the log file.
"""
import argparse
import os
import re
from numpy import mean
import time

def work_multi_inputs(origin_file, optimized_file, output_file):
    results_origin, summary_metric_origin = work_multi_models(origin_file)
    results_optimized, summary_metric_optimized = work_multi_models(optimized_file)
    speedups_change = {}
    for model in results_origin:
        speedups_change[model] = results_optimized[model] - results_origin[model]
    table_head = 'model, origin, optimized, speedup_change\n'
    with open(output_file, 'w') as fout:
        print("writing to file %s" % output_file)
        fout.write("origin file: %s\n" % origin_file)
        fout.write("optimized file: %s\n" % optimized_file)
        fout.write(table_head)
        for model in results_origin:
            fout.write('%s, %s, %s, %.2f\n' % (model, results_origin[model], results_optimized[model], speedups_change[model]))
        fout.write('\n\n\nmetric,mean_origin,mean_optimized,gmean_origin,gmean_optimized\n')
        for metric_name in summary_metric_origin:
            fout.write('%s,%s,%s,%s,%s\n' % (metric_name, summary_metric_origin[metric_name].get('mean', "None"), summary_metric_optimized[metric_name].get('mean', "None"), summary_metric_origin[metric_name].get('gmean', "None"), summary_metric_optimized[metric_name].get('gmean', "None")))


def work_multi_models(input_file):
    content = ''
    with open(input_file, 'r') as fin:
        content = fin.read()
    if "cuda train" in content:
        mode='train'
    elif "cuda eval" in content:
        mode='eval'
    split_str = "cuda %s" % mode
    content_s = [_ for _ in content.split(split_str) if _.strip()][1:]
    results = {}
    for amodel in content_s:
        model_name = amodel.strip().split()[0].strip()
        reg_speedup = re.compile(r"\n(\d+?\.?\d*?)x\n")
        speedup = reg_speedup.findall(amodel)
        if not speedup:
            continue
        results[model_name] = float(speedup[0].strip())
    last_content = content_s[-1]
    reg_summary = re.compile(r"\n\d+?\.?\d*?x\n([\s\S]*)")
    last_content = reg_summary.findall(last_content)[0]
    summary_metric = {}
    for metric_line in last_content.split('\n'):
        if not metric_line.strip():
            continue
        metric_line = metric_line.strip().split()
        metric_name = metric_line[0]
        summary_metric[metric_name] = {}
        if metric_name == 'compilation_latency':
            mean_name = metric_line[1].split('=')[0]
            mean_value = metric_line[1].split('=')[1] + ' ' + metric_line[2]
            summary_metric[metric_name][mean_name] = mean_value
        else:
            for mean_line in metric_line[1:]:
                mean_name = mean_line.split('=')[0]
                mean_value = mean_line.split('=')[1]
                summary_metric[metric_name][mean_name] = mean_value
    return results, summary_metric



if __name__ == '__main__':
    # get current time and date like 202304141010
    current_time = time.strftime("%Y%m%d%H%M", time.localtime())
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=str, required=True)
    parser.add_argument('--optimized', type=str, required=True)
    parser.add_argument('-o', '--output', type=str,
                        default='filter_speedup_change_inductorbench_%s.csv' % current_time)
    args = parser.parse_args()
    work_multi_inputs(args.origin, args.optimized, args.output)
