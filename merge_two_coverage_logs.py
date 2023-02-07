import argparse
import torch


def parse_func(func):
    if hasattr(func, '__module__'):
        module_name = func.__module__
        func_name = func.__name__
    else:
        if hasattr(func, '__qualname__'):
            func_name = func.__qualname__
            module_name = ''
        else:
            if type(func) == torch._C.Generator:
                func_name = 'torch._C.Generator'
                module_name = ''
            else:
                raise RuntimeError(
                    "no matched module and func name: ", func, type(func))
    return module_name, func_name


def generate_API_list():
    tmp_api_list = set()
    raw_all_apis = set(torch.overrides.get_testing_overrides().keys())
    # collect all items' attribute  `module` to a list
    for item in raw_all_apis:
        module_name, func_name = parse_func(item)
        # if (module_name, func_name) in api_list:
        # print("duplicated: ", (module_name, func_name))
        tmp_api_list.add((module_name, func_name))
    ignored_funcs = set(
        [_ for _ in torch.overrides.get_ignored_functions() if _ not in [True, False]])
    tmp_ignored_api_list = set()
    for item in ignored_funcs:
        module_name, func_name = parse_func(item)
        tmp_ignored_api_list.add((module_name, func_name))
    return tmp_api_list, tmp_ignored_api_list


API_LIST, IGNORED_API_LIST = generate_API_list()


def read_log(log_path):
    with open(log_path, 'r') as fin:
        content = fin.read()
        if content.find("missed apis") < 0:
            print("log format error, can't find 'missed apis': ", log_path)
        else:
            content = content[content.find("missed apis"):]
        lines = content.split('\n')
        cc = []
        for node in lines[2:]:
            if not node.strip():
                continue
            s = node.split(',')
            s = [_.strip() for _ in s]
            cc.append((s[0], s[1]))
        return set(cc)


if __name__ == "__main__":
    # add argument parser
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--log1", type=str, required=True, help="log1")
    parser.add_argument("--log2", type=str, required=True, help="log2")
    parser.add_argument("--output", type=str, required=True, help="output")
    args = parser.parse_args()
    cc1 = read_log(args.log1)
    cc2 = read_log(args.log2)
    cc = cc1.intersection(cc2)
    used_apis = API_LIST - cc
    missing_apis = API_LIST - used_apis
    # print used apis to file
    with open("apis.txt", 'w') as fout:
        fout.write("API coverage: %d / %d = %.2f\n" %(len(used_apis), len(API_LIST), len(used_apis) / len(API_LIST)))
        fout.write("used apis:\n")
        fout.write("module,func\n")
        for item in used_apis:
            fout.write("%s,%s\n" % (item[0], item[1]))
        fout.write("====================================\n")
        fout.write("missed apis:\n")
        fout.write("module,func\n")
        for item in missing_apis:
            fout.write("%s,%s\n" % (item[0], item[1]))
