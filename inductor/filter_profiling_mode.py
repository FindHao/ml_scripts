
import argparse
import os
import logging
import re
from statistics import mean
logger = logging.getLogger(__name__)
# set logger level to INFO
logger.setLevel(logging.INFO)
models = []

search_strs = ['original speedup', 'multiple streams speedup', 'updated stream speedup']
reg = re.compile(r"(original speedup[\s\S]+?)(multiple streams speedup[\s\S]+?)(updated stream speedup[\s\S]+)")
def read_models(input_file):
    with open(input_file, 'r') as f:
        for line in f:
            models.append(line.strip().split(',')[0])

def filter_speedup_csv(input_file):
    content = ''
    with open(input_file, 'r') as f:
        content = f.read()
    if not all(search_str in content for search_str in search_strs):
        return False
    speedups = []
    results = reg.findall(content)
    # breakpoint()
    for part in results[0]:
        speedups.append(re.findall(r'(\d+\.\d+)x', part))
    assert len(speedups) == 3
    mean_speedups = []
    for alist in speedups:
        mean_speedups.append(mean([float(x) for x in alist]))
    return mean_speedups

def work(input_file, speedup_dir, output_file):
    read_models(input_file)
    model_results = []
    for model in models:
        model_files = [os.path.join(speedup_dir, file) for file in os.listdir(speedup_dir) if model in file]
        if len(model_files) == 0:
            logger.error('No speedup file for model: {}'.format(model))
            continue
        
        latest_file = max(model_files, key=os.path.getctime)
        results = filter_speedup_csv(latest_file)
        if results:
            original, multiple_streams, updated_stream = results
            changes = updated_stream / original
            model_results.append((model, original, multiple_streams, updated_stream, changes))
        else:
            logger.error('Speedup file {} is not valid'.format(latest_file))
    # sort by changes
    model_results.sort(key=lambda x: x[-1], reverse=True)
    with open(output_file, 'w') as f:
        f.write('model,original,multiple_streams,updated_stream,changes\n')
        for model, original, multiple_streams, updated_stream, changes in model_results:
            # limit the precision to 3 by format string
            f.write(f"{model},{original:.3f},{multiple_streams:.3f},{updated_stream:.3f},{changes:.3f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='the csv file saves successful updated models')
    parser.add_argument('--speedup-dir', '-s', type=str, required=True, help='the directory saves the speedup results')
    parser.add_argument('--output', '-o', type=str, required=True, help='the output file')
    args = parser.parse_args()
    work(args.input, args.speedup_dir, args.output)