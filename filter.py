import re


# order = ['alexnet', 'BERT_pytorch', 'dcgan', 'dlrm', 'hf_Bart', 'hf_Bert', 'hf_BigBird', 'hf_GPT2', 'hf_Longformer', 'hf_Reformer', 'LearningToPaint', 'maml', 'mnasnet1_0', 'mobilenet_v2', 'mobilenet_v2_quantized_qat', 'mobilenet_v3_large', 'nvidia_deeprecommender', 'opacus_cifar10', 'pytorch_unet', 'resnet18', 'resnet50', 'resnet50_quantized_qat', 'resnext50_32x4d', 'shufflenet_v2_x1_0', 'soft_actor_critic', 'timm_efficientnet', 'timm_regnet', 'timm_resnest', 'timm_vision_transformer', 'timm_vovnet', 'tts_angular', 'vgg16']
all = ["fambench_dlrm", "fambench_xlmr", "detectron2_maskrcnn", "vision_maskrcnn", "timm_efficientnet", "timm_vision_transformer", "hf_Bert", "hf_GPT2", "hf_T5", "resnet50", "timm_resnest", "resnext50_32x4d", "hf_BigBird", "hf_Bart", "soft_actor_critic", "alexnet", "timm_vovnet", "mobilenet_v3_large", "vgg16", "shufflenet_v2_x1_0", "pytorch_unet", "dlrm", "mnasnet1_0", "resnet50_quantized_qat", "tts_angular", "hf_Reformer", "nvidia_deeprecommender", "mobilenet_v2_quantized_qat", "mobilenet_v2", "LearningToPaint", "hf_Longformer", "opacus_cifar10", "resnet18", "timm_regnet", "dcgan", "maml", "BERT_pytorch", "Super_SloMo", "pytorch_struct", "pplbench_beanmachine", "drq", "pyhpc_isoneutral_mixing", "hf_Albert", "attention_is_all_you_need_pytorch", "moco", "Background_Matting", "pyhpc_turbulent_kinetic_energy", "maml_omniglot", "pyhpc_equation_of_state", "timm_nfnet", "demucs", "densenet121", "pytorch_CycleGAN_and_pix2pix", "tacotron2", "squeezenet1_1", "fastNLP_Bert", "pytorch_stargan", "hf_DistilBert", "speech_transformer", "yolov3", "timm_efficientdet"]
def work(input_file):
    content = ''
    with open(input_file, 'r') as fin:
        content = fin.read()
    tflops = {}
    reg=re.compile(r"Run (.+?) with dcgm[\s\S]+?([\d.]+?) +?TFLOPs per second")
    content_s = [_ for _ in content.split("@Yueming Hao: start model tests") if _.strip()]
    for part in content_s:
        results = reg.findall(part)
        if not results:
            reg2=re.compile(r"Run (.+?) with dcgm")
            print("error for ", reg2.findall(part)[0])
        else:
            line = results[0]
            tflops[line[0]] = line[1]
    with open("runlog.filter.csv", 'w') as fout:
        for name in all:
            fout.write("%s, %s\n" % (name, tflops.get(name, "error")))



work("/home/yhao/d/testing/runlog_100ms_train_all.txt")
