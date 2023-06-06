
import argparse
import re
import time
import os

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n 
        self.sets = {x: {x} for x in range(n)} 

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootx = self.find(x)
        rooty = self.find(y)
        if rootx != rooty:
            if self.rank[rootx] < self.rank[rooty]:
                rootx, rooty = rooty, rootx
            self.sets[rootx].update(self.sets[rooty])
            del self.sets[rooty]
            self.parent[rooty] = rootx
            if self.rank[rootx] == self.rank[rooty]:
                self.rank[rootx] += 1
            self.count -= 1

    def total_count(self):
        return self.count

    def get_sets(self):
        return list(self.sets.values())


def compare_kernels(kernel1, kernel2):
    kernel1 = kernel1.split('\n')
    kernel1 = [_ for _ in kernel1 if _.strip() and not _.strip().startswith('#')]
    kernel2 = kernel2.split('\n')
    kernel2 = [_ for _ in kernel2 if _.strip() and not _.strip().startswith('#')]
    num_diff = 0
    num_same = 0
    num_total = 0
    diff_lines = []
    for line1, line2 in zip(kernel1, kernel2):
        if line1 != line2:
            num_diff += 1
            diff_lines.append((line1, line2))
        else:
            num_same += 1
        num_total += 1
    return num_diff, num_same, num_total, diff_lines

def check_output_file_headers(output_file):
    standard_headers = "model_name, sub_name, num_kernels, num_calls\n"
    if not os.path.exists(output_file):
        with open(output_file, 'w') as fout:
            fout.write(standard_headers)
    else:
        with open(output_file, 'r') as fin:
            headers1 = fin.readline()
            headers2 = fin.readline()
        if headers1 != standard_headers or headers2 != standard_headers:
            with open(output_file, 'w') as fout:
                fout.write(standard_headers)
        


def work(input_file, input_dir, output_file):
    check_output_file_headers(output_file)
    if input_dir:
        input_dir = os.path.abspath(input_dir)
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith("_profile.json"):
                    input_file_path = os.path.join(root, file)
                    work_single_file(input_file_path, output_file)
    elif input_file:
        work_single_file(input_file, output_file)
    else:
        raise ValueError("either input or dir should be specified")


def work_single_file(input_file, output_file):
    # BertForMaskedLM__0_forward_13.0
    # model_name_raw = os.path.basename(os.path.dirname(os.path.abspath(input_file)))
    print("processing file %s" % input_file)
    model_name_raw = os.path.basename(os.path.abspath(input_file))
    model_name, sub_name = model_name_raw.split('__')
    with open(input_file, 'r') as fin:
        content = fin.read()
    reg = re.compile(r"(.*) = async_compile.triton\('triton_', '''([\s\S]+?)'''\)")
    results = reg.findall(content)
    if not results:
        print("no kernels found in file %s" % input_file)
        return
    # num of generated kernels
    len_results = len(results)
    reg2 = re.compile(r"(.*)\.run\(")
    results2 = reg2.findall(content)
    # num of generated calls
    len_results2 = len(results2)
    with open(output_file, 'a') as fout:
        fout.write(f"{model_name}, {sub_name}, {len_results}, {len_results2}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # specify the file to check
    parser.add_argument('-i', '--input', type=str)
    # specify the dir to search for all files, it is exclusive with -i
    parser.add_argument('-d', '--dir', type=str)
    parser.add_argument('-o', '--output', type=str, default='kernel_nums.csv')
    args = parser.parse_args()
    work(args.input, args.dir, args.output)
