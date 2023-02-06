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
f2=open("/home/yhao/d/p8/benchmark/.userbenchmark/api-coverage/logs/logs-20230206022122.json-api_coverage.csv", 'r')
f1 = open("/home/yhao/d/p8/benchmark/.userbenchmark/api-coverage/logs/logs-20230205151329.json-api_coverage.csv", 'r')
c1 = f1.readlines()
c2 = f2.readlines()
f1.close()
f2.close()
cc1 = []
for node in c1[4:]:
    if not node.strip():
        continue
    s = node.split(',')
    s = [_.strip() for _ in s]
    cc1.append((s[0], s[1]))
cc2 = []
for node in c2[4:]:
    if not node.strip():
        continue
    s = node.split(',')
    s = [_.strip() for _ in s]
    cc2.append((s[0], s[1]))
cc1 = set(cc1)
cc2 = set(cc2)
# combine two sets
cc = cc1.union(cc2)
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
    fout.write("missing apis:\n")
    fout.write("module,func\n")
    for item in missing_apis:
        fout.write("%s,%s\n" % (item[0], item[1]))


