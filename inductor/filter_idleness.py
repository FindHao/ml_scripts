import argparse
import re
import os
import sys
sys.path.append('/home/yhao24/p/p9/TorchExpert')

from torchexpert import TorchExpert
from torchexpert import INPUT_MODE_JSON



def work(input_file, input_dir, output_file):
    output_file = os.path.abspath(output_file)
    if input_dir is not None:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('_profile.json'):
                    single_file = os.path.join(root, file)
                    print("processing file %s" % single_file)
                    model_name = os.path.basename(single_file).split('_profile.json')[0]
                    torchexpert = TorchExpert(output_csv_file=output_file, model_name=model_name, log_file="torchexpert.log", input_mode=INPUT_MODE_JSON)
                    torchexpert.analyze(single_file)
    elif input_file is not None:
        torchexpert = TorchExpert(output_csv_file=output_file)
        torchexpert.analyze(input_file)
    else:
        raise ValueError('Either input_file or input_dir should be specified.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # specify the file to check
    parser.add_argument('-i', '--input', type=str)
    # specify the dir to search for all files, it is exclusive with -i
    parser.add_argument('-d', '--dir', type=str)
    parser.add_argument('-o', '--output', type=str, default='output_code_idleness.csv')
    args = parser.parse_args()
    work(args.input, args.dir, args.output)
