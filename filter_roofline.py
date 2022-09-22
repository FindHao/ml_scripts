

def filter_from_csv(file_path):
    # sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained
    with open(file_path, 'r') as f:
        content = f.read()
    if content.find('"ID","Process ID","Process Name",') < 0:
        print("Invalid csv file")
        return
    content = content.split('"ID","Process ID","Process Name",')[1]
    lines = content.splitlines()
    lines[0] = '"ID","Process ID","Process Name",'+lines[0]
    head_line = lines[0]
    head_line = head_line.replace('"', '').split(',')
    kernels = []
    for line in lines:
        line = line.replace('"', '').split(',')
        kernel = {}
        for i in range(len(line)):
            kernel[head_line[i]] = line[i]
        kernels.append(kernel)


def compute_roofline(d):
    peak_fp32 = 2 * \
        float(d['sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained'])
    achieved_fp32 = 0.0
    achieved_fp32 += float(
        d["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed"])
    achieved_fp32 += float(
        d["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed"])
    achieved_fp32 += float(
        d["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed"])
    achieved_point_y = achieved_fp32
    achieved_point_x = achieved_fp32 * \
        float(d["smsp__cycles_elapsed.avg.per_second"])

    roofline_y = float(
        d['sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained']) * 2
    roofline_oblique_k = float(
        d['dram__bytes.sum.peak_sustained']) * float(d['dram__cycles_elapsed.avg.per_second'])
    intersection_x = roofline_y / roofline_oblique_k

    eligible_peak_flops = 0
    in_right_are = True
    if achieved_point_x > intersection_x:
        eligible_peak_flops = roofline_y
    else:
        eligible_peak_flops = achieved_point_x * roofline_oblique_k
        in_right_are = False
    return (in_right_are, eligible_peak_flops, achieved_point_y, peak_fp32)


def work():
    filter_from_csv("/home/yhao24/data/ncu/detectron2_maskrcnn_r_101_fpn.csv")
    print(compute_roofline())


work()
