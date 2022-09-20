import numpy as np

all_models = {}
all_models_avg = {}
memcpy_ratios = {}
active_ratios = {}
busy_ratios = {}
occupancy = {}

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
    if occupancy.get(line[0]) is None:
        occupancy[line[0]] = [float(line[8])]
    else:
        occupancy[line[0]].append(float(line[8]))

def work(file_path):
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
    with open('model_ratios_train.csv', 'w') as fout:
        fout.write('model_name,memcpy_ratio,active_ratio,busy_ratio,occupancy\n')
        for model_name in memcpy_ratios:
            fout.write('%s, %.2f, %.2f, %.2f, %.2f\n' % (model_name, np.mean(memcpy_ratios[model_name]), np.mean(active_ratios[model_name]), np.mean(busy_ratios[model_name]), np.mean(occupancy[model_name])))

work("/home/yhao/d/tmp/torchexpert_results_train_20220918.csv")