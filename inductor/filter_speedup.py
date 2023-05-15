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

def work_multi_models(input_file, output_file):
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

    table_head = 'model, speedup\n'
    with open(output_file, 'w') as fout:
        print("writing to file %s" % output_file)
        fout.write("input file: %s\n" % input_file)
        fout.write(table_head)
        for model in results:
            fout.write('%s, %s\n' % (model, results[model]))
        fout.write('\n\n\nmetric,mean,gmean\n')
        for metric_name in summary_metric:
            fout.write('%s,%s,%s\n' % (metric_name, summary_metric[metric_name].get('mean', ''), summary_metric[metric_name].get(
                'gmean', '')))



if __name__ == '__main__':
    # get current time and date like 202304141010
    current_time = time.strftime("%Y%m%d%H%M", time.localtime())
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='/home/yhao/d/tmp/run_all_speedup_aug4.log')
    parser.add_argument('-o', '--output', type=str,
                        default='filter_speedup_inductorbench_%s.csv' % current_time)
    args = parser.parse_args()
    work_multi_models(args.input, args.output)
