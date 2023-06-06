import re
from collections import defaultdict

# Union-Find (Disjoint Set) Data Structure
class DisjointSet:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1, item2):
        self.parent[self.find(item1)] = self.find(item2)

    def groups(self):
        roots = {self.find(item) for item in self.parent}
        groups = {root: [] for root in roots}

        for item in self.parent:
            groups[self.find(item)].append(item)

        return groups.values()


def analyze_file(filename):
    disjoint_set = DisjointSet()
    with open(filename, 'r') as file:
        content = file.read()

    # Extract the content of the `call` function
    call_func_pattern = r'def call\(args\):([\s\S]*?)(?=def |\Z)'
    call_func_content = re.search(call_func_pattern, content, re.MULTILINE)

    if call_func_content:
        # Search for lines with `XX.run()` and `extern_kernels.XX()`
        pattern = r'(\w+)\.(run|mm|bmm)\(([\w\,\s=]+)'
        lines = re.findall(pattern, call_func_content.group(1))

        # For each line, add connections to the disjoint set
        for line in lines[:-100]:
            func_name = line[0] + '.' + line[1]
            args = line[2].split(',')
            for arg in args:
                # Check if argument is "out=XX"
                if "out=" in arg:
                    arg = arg.split('=')[1].strip()
                else:
                    arg = arg.strip()

                disjoint_set.union(func_name, arg)

    # Print the disjoint sets
    disjoint_sets = disjoint_set.groups()
    print("Number of independent disjoint sets:", len(disjoint_sets))
    for index, dset in enumerate(disjoint_sets, start=1):
        print(f"Set {index}: {dset}")

# Analyze the given .py file
analyze_file('/home/users/yhao24/b/p9/pytorch/torch_compile_debug/run_2023_06_01_11_41_09_972435-pid_1451525/torchinductor/GPT2ForSequenceClassification__0_backward_76.1/output_code_opt.py')
