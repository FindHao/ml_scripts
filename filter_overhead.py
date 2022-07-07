import re
from tkinter import N
from unittest import result
import numpy as np

def get_mean(reg, content):
    results = reg.findall(content)
    if not results:
        reg2 = re.compile(r"@Yueming Hao: Run (.*)")
        print("error for ", reg2.findall(content)[0])
        return None
    print(results)
    return np.mean([float(_) for _ in results[0]])

def time_convert(tmp_str):
    """
    convert linux time cmd's results to seconds.
    """
    tmp_str = tmp_str.strip()
    result = [ _ for _ in re.split('[ms]', tmp_str) if _.strip()]
    if len(result) != 2:
        print("error when convert time ", tmp_str)
        return None
    return float(result[0]) * 60 + float(result[1])


def get_mean_time_cmd(reg, content):
    results = reg.findall(content)
    if not results:
        reg2 = re.compile(r"@Yueming Hao: Run (.*)")
        print("error for ", reg2.findall(content)[0])
        return None
    results = [ time_convert(_) for _ in results[0] ]
    print(results)
    return np.mean(results)


def get_app_name(content):
    reg = re.compile(r"@Yueming Hao: Run ([a-zA-z_0-9]*)")
    results = reg.findall(content)
    return results[0]

# order = ['alexnet', 'BERT_pytorch', 'dcgan', 'dlrm', 'hf_Bart', 'hf_Bert', 'hf_BigBird', 'hf_GPT2', 'hf_Longformer', 'hf_Reformer', 'LearningToPaint', 'maml', 'mnasnet1_0', 'mobilenet_v2', 'mobilenet_v2_quantized_qat', 'mobilenet_v3_large', 'nvidia_deeprecommender', 'opacus_cifar10', 'pytorch_unet', 'resnet18', 'resnet50', 'resnet50_quantized_qat', 'resnext50_32x4d', 'shufflenet_v2_x1_0', 'soft_actor_critic', 'timm_efficientnet', 'timm_regnet', 'timm_resnest', 'timm_vision_transformer', 'timm_vovnet', 'tts_angular', 'vgg16']
all = ["fambench_dlrm", "fambench_xlmr", "detectron2_maskrcnn", "vision_maskrcnn", "timm_efficientnet", "timm_vision_transformer", "hf_Bert", "hf_GPT2", "hf_T5", "resnet50", "timm_resnest", "resnext50_32x4d", "hf_BigBird", "hf_Bart", "soft_actor_critic", "alexnet", "timm_vovnet", "mobilenet_v3_large", "vgg16", "shufflenet_v2_x1_0", "pytorch_unet", "dlrm", "mnasnet1_0", "resnet50_quantized_qat", "tts_angular", "hf_Reformer", "nvidia_deeprecommender", "mobilenet_v2_quantized_qat", "mobilenet_v2", "LearningToPaint", "hf_Longformer", "opacus_cifar10", "resnet18", "timm_regnet", "dcgan", "maml", "BERT_pytorch", "Super_SloMo", "pytorch_struct", "pplbench_beanmachine", "drq", "pyhpc_isoneutral_mixing", "hf_Albert", "attention_is_all_you_need_pytorch", "moco", "Background_Matting", "pyhpc_turbulent_kinetic_energy", "maml_omniglot", "pyhpc_equation_of_state", "timm_nfnet", "demucs", "densenet121", "pytorch_CycleGAN_and_pix2pix", "tacotron2", "squeezenet1_1", "fastNLP_Bert", "pytorch_stargan", "hf_DistilBert", "speech_transformer", "yolov3", "timm_efficientdet"]
def work(input_file):
    content = ''
    with open(input_file, 'r') as fin:
        content = fin.read()
    # reg=re.compile(r"Run (.+?) with dcgm[\s\S]+?([\d.]+?) +?TFLOPs per second")
    reg=re.compile(r"@Yueming Hao: Run [\s\S]+?CPU Total Wall Time: (.+?) milliseconds[\s\S]+?CPU Total Wall Time: (.+?) milliseconds[\s\S]+?CPU Total Wall Time: (.+?) milliseconds[\s\S]+?CPU Total Wall Time: (.+?) milliseconds[\s\S]+?CPU Total Wall Time: (.+?) milliseconds")
    reg_dcgm=re.compile(r"@Yueming Hao: Run.*with dcgm flops[\s\S]+?CPU Total Wall Time: (.+?) milliseconds[\s\S]+?CPU Total Wall Time: (.+?) milliseconds[\s\S]+?CPU Total Wall Time: (.+?) milliseconds[\s\S]+?CPU Total Wall Time: (.+?) milliseconds[\s\S]+?CPU Total Wall Time: (.+?) milliseconds")    
    content_s = [_ for _ in content.split("@Yueming Hao: start model tests") if _.strip()]
    # {name: [avg_time, avg_time_dcgm, overhead]}
    overheads = {}
    for part in content_s:
        avg_time = get_mean(reg, part)
        if avg_time is not None:
            avg_time_dcgm = get_mean(reg_dcgm, part)
            if avg_time_dcgm is None:
                continue
            app_name = get_app_name(part)
            overheads[app_name] = [avg_time, avg_time_dcgm, avg_time_dcgm / avg_time]
        
    with open("runlog.filter.overhead.csv", 'w') as fout:
        fout.write("app, execution time(ms), execution time dcgm(ms), overhead\n")
        for name in all:
            overhead = overheads.get(name, [-1, -1, -1])
            fout.write("%s, %.2f, %.2f, %.2f\n" % (name, overhead[0], overhead[1], overhead[2]))



def work_for_time_cmd(input_file):
    content = ''
    with open(input_file, 'r') as fin:
        content = fin.read()
    # reg=re.compile(r"Run (.+?) with dcgm[\s\S]+?([\d.]+?) +?TFLOPs per second")
    reg=re.compile(r"@Yueming Hao: Run [\s\S]+?real\t(.*)\n[\s\S]+?real\t(.*)\n[\s\S]+?real\t(.*)\n[\s\S]+?real\t(.*)\n[\s\S]+?real\t(.*)")
    reg_dcgm=re.compile(r"@Yueming Hao: Run.*with dcgm flops[\s\S]+?real\t(.*)\n[\s\S]+?real\t(.*)\n[\s\S]+?real\t(.*)\n[\s\S]+?real\t(.*)\n[\s\S]+?real\t(.*)")    
    content_s = [_ for _ in content.split("@Yueming Hao: start model tests") if _.strip()]
    overheads = {}
    for part in content_s:
        avg_time = get_mean_time_cmd(reg, part)
        if avg_time is not None:
            avg_time_dcgm = get_mean_time_cmd(reg_dcgm, part)
            if avg_time_dcgm is None:
                continue
            app_name = get_app_name(part)
            overheads[app_name] = [avg_time, avg_time_dcgm, avg_time_dcgm / avg_time]
        
    with open("runlog.filter.overhead.csv", 'w') as fout:
        fout.write("app, execution time(ms), execution time dcgm(ms), overhead\n")
        for name in all:
            overhead = overheads.get(name, [-1, -1, -1])
            fout.write("%s, %.2f, %.2f, %.2f\n" % (name, overhead[0], overhead[1], overhead[2]))

work("/home/yhao/d/testing/runlog_100ms_train_all.collect.txt")
# work_for_time_cmd("/home/yhao/d/testing/runlog_100ms_train_all.txt")
