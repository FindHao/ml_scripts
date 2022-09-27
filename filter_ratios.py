import argparse
import numpy as np

all_models = {}
all_models_avg = {}
memcpy_ratios = {}
active_ratios = {}
busy_ratios = {}

def parse_line_to_dict(line):
    if memcpy_ratios.get(line[0]) is None:
        memcpy_ratios[line[0]] = [float(line[5])]
    else:
        memcpy_ratios[line[0]].append(float(line[5]))
    if active_ratios.get(line[0]) is None:
        active_ratios[line[0]] = [float(line[6])]
    else:
        active_ratios[line[0]].append(float(line[6]))
    if busy_ratios.get(line[0]) is None:
        busy_ratios[line[0]] = [float(line[7])]
    else:
        busy_ratios[line[0]].append(float(line[7]))

def work(file_path, output_path):
    # load csv file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # remove header
    lines = lines[1:]
    # split lines
    lines = [line.split(',') for line in lines]
    # remove empty lines
    lines = [line for line in lines if line[0] != '']
    # parse every line to dict
    for line in lines:
        parse_line_to_dict(line)
    
    # for model_name in memcpy_ratios:
    with open(output_path, 'w') as fout:
        fout.write('model_name,memcpy_ratio,active_ratio,busy_ratio\n')
        for model_name in memcpy_ratios:
            fout.write('%s, %.2f, %.2f, %.2f\n' % (model_name, np.mean(memcpy_ratios[model_name]), np.mean(active_ratios[model_name]), np.mean(busy_ratios[model_name])))


if __name__ == '__main__':
    test_input = "/home/yhao24/ncsugdrive/data/pt_new/torchexpert_train_results_202209271621.csv"
    test_output = "model_ratios_train.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=test_input, help='input csv file')
    parser.add_argument('--output', type=str, default=test_output, help='output csv file')
    args = parser.parse_args()
    work(args.input, args.output)