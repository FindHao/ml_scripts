
import argparse
import re
import time


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

def work(input_file, output_file):
    with open(input_file, 'r') as fin:
        content = fin.read()
    reg = re.compile(r"(.*) = async_compile.triton\('triton_', '''([\s\S]+?)'''\)")
    results = reg.findall(content)
    if not results:
        print("no kernels found in file %s" % input_file)
        return
    kernel_names = []
    tmp_results = []
    for node in results:
        kernel_names.append(node[0])
        tmp_results.append(node[1])
    len_results = len(tmp_results)
    results = tmp_results
    dsu = UnionFind(len_results)
    for i in range(len_results):
        for j in range(i+1, len_results):
            num_diff, num_same, num_total, diff_lines = compare_kernels(results[i], results[j])
            # if num_diff / num_total < 0.1:
            if num_diff <= 2:
                print("kernel %s and kernel %s are similar" % (kernel_names[i], kernel_names[j]))
                print("num_diff: %d, num_same: %d, num_total: %d" % (num_diff, num_same, num_total))
                dsu.union(i, j)
    print("=====================================")
    print("total kernels: %d" % len_results)
    print("total kernel sets: %d" % dsu.total_count())
    print("kernel sets: %s" % dsu.get_sets())
    print("output file: %s" % output_file)
    with open(output_file, 'w') as fout:
        fout.write("input file: %s\n" % input_file)
        for aset in dsu.get_sets():
            if len(aset) == 1:
                continue
            fout.write("=====================================\n")
            for idx in aset:
                fout.write("kernel %s:\n" % kernel_names[idx])
                fout.write("\n")


if __name__ == '__main__':
    current_time = time.strftime("%Y%m%d%H%M", time.localtime())
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='/scratch/yhao24/p9_inductor/pytorch/torch_compile_debug/run_2023_05_18_17_35_43_458243-pid_3201381/torchinductor/BertForMaskedLM__0_forward_13.0/output_code.py')
    parser.add_argument('-o', '--output', type=str,
                        default='redundant_kernels_%s.txt' % current_time)
    args = parser.parse_args()
    work(args.input, args.output)
