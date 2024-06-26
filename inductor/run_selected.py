import argparse
import datetime
import subprocess
import time
import re
import os
import subprocess
import time
import re
import os

# get current date time in a btter format
now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d_%H-%M-%S")
tmp_file = f"/tmp/inductor_tmp_{now}.txt"
default_profiler_trace_path = "/mnt/beegfs/users/yhao24/tmp/profile_all/"


def run_models(model_list, collections, test_accuracy=False, mode="inference", profile=False, disable_streams=False):
    result_dict = {}
    if test_accuracy:
        test_acc_or_perf = "--accuracy"
    else:
        test_acc_or_perf = "--performance"
    if mode == "inference":
        mode = "--inference"
        precision_place_holder = "--bfloat16"
    else:
        mode = "--training"
        precision_place_holder = "--amp"

    if disable_streams:
        stream_place_holder = "TORCHINDUCTOR_MULTIPLE_STREAMS=0"
    else:
        stream_place_holder = ""

    for model in model_list:

        if profile:
            profile_place_holder = f"--export-profiler-trace --profiler-trace-name={default_profiler_trace_path}{model}"
        else:
            profile_place_holder = ""
        command = f'{stream_place_holder} python benchmarks/dynamo/{collections}.py {test_acc_or_perf} {precision_place_holder} {profile_place_holder} -dcuda {mode} --inductor --disable-cudagraphs --only {model}'
        start_time = time.time()

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

        while True:
            if process.poll() is not None:  # process finished
                stdout, stderr = process.communicate()
                stdout = stdout.decode('utf-8')  # decoding the output
                stderr = stderr.decode('utf-8')  # decoding the error
                # write stdout and stderr to tmp_file
                with open(tmp_file, "a") as f:
                    f.write(f"model: {model}\n")
                    f.write(f"stdout: {stdout}\n")
                    f.write(f"stderr: {stderr}\n")
                if process.returncode != 0:  # process ended with an error
                    result_dict[model] = f"error is {stderr}"
                else:
                    if not test_accuracy:
                        match = re.search(r'(\d+?\.\d+?)x', stdout)
                        if match:
                            result_dict[model] = float(match.group(1))
                    else:
                        match = re.search('pass', stdout)
                        if match:
                            result_dict[model] = "pass"
                        elif re.search("fail_to_run", stdout):
                            result_dict[model] = "fail_to_run"
                        else:
                            result_dict[model] = "fail"
                break

            elif time.time() - start_time > 600:  # if process is running more than 10 minutes
                process.kill()  # killing the process
                result_dict[model] = "Execution time exceeded 10 minutes"
                break

            time.sleep(1)  # sleep for 1 second before next check

    return result_dict


def write_results(results, output_file):
    successed = []
    failed = []
    for key in results:
        if isinstance(results[key], float) or results[key] == "pass":
            successed.append(key)
        else:
            failed.append(key)
    with open(output_file, "a") as f:
        f.write("Successed:\n")
        for model in successed:
            f.write(f"{model}, {results[model]}\n")
        f.write("\nFailed:\n")
        for model in failed:
            f.write(f"{model}, {results[model]}\n")


timm_models_list = "adv_inception_v3 beit_base_patch16_224 botnet26t_256 cait_m36_384 coat_lite_mini convit_base convmixer_768_32 convnext_base crossvit_9_240 cspdarknet53 deit_base_distilled_patch16_224 dla102 dm_nfnet_f0 dpn107 eca_botnext26ts_256 eca_halonext26ts ese_vovnet19b_dw fbnetc_100 fbnetv3_b gernet_l ghostnet_100 gluon_inception_v3 gmixer_24_224 gmlp_s16_224 hrnet_w18 inception_v3 jx_nest_base lcnet_050 levit_128 mixer_b16_224 mixnet_l mnasnet_100 mobilenetv2_100 mobilenetv3_large_100 mobilevit_s nfnet_l0 pit_b_224 pnasnet5large poolformer_m36 regnety_002 repvgg_a2 res2net101_26w_4s res2net50_14w_8s res2next50 resmlp_12_224 resnest101e rexnet_100 sebotnet33ts_256 spnasnet_100 swin_base_patch4_window7_224 swsl_resnext101_32x16d tf_efficientnet_b0 tf_mixnet_l tinynet_a tnt_s_patch16_224 twins_pcpvt_base vit_base_patch16_224 volo_d1_224 xcit_large_24_p8_224"
timm_models_list = timm_models_list.split(" ")
timm_models_collection = "timm_models"

torchbench_list = "BERT_pytorch Background_Matting DALLE2_pytorch LearningToPaint Super_SloMo alexnet attention_is_all_you_need_pytorch basic_gnn_edgecnn basic_gnn_gcn basic_gnn_gin basic_gnn_sage cm3leon_generate dcgan demucs densenet121 detectron2_fcos_r_50_fpn dlrm doctr_det_predictor doctr_reco_predictor drq fastNLP_Bert functorch_dp_cifar10 functorch_maml_omniglot hf_Albert hf_Bart hf_Bert hf_Bert_large hf_BigBird hf_DistilBert hf_GPT2 hf_GPT2_large hf_Longformer hf_Reformer hf_T5 hf_T5_generate hf_T5_large lennard_jones llama maml maml_omniglot mnasnet1_0 mobilenet_v2 mobilenet_v3_large moco nanogpt_generate nvidia_deeprecommender opacus_cifar10 phlippe_densenet phlippe_resnet pyhpc_equation_of_state pyhpc_isoneutral_mixing pyhpc_turbulent_kinetic_energy pytorch_CycleGAN_and_pix2pix pytorch_stargan pytorch_unet resnet152 resnet18 resnet50 resnext50_32x4d shufflenet_v2_x1_0 soft_actor_critic speech_transformer squeezenet1_1 timm_efficientdet timm_efficientnet timm_nfnet timm_regnet timm_resnest timm_vision_transformer timm_vision_transformer_large timm_vovnet tts_angular vgg16 vision_maskrcnn yolov3"
torchbench_list = torchbench_list.split(" ")
torchbench_collection = "torchbench"

huggingface_list = "AlbertForMaskedLM AlbertForQuestionAnswering AllenaiLongformerBase BartForCausalLM BartForConditionalGeneration BertForMaskedLM BertForQuestionAnswering BlenderbotForCausalLM BlenderbotSmallForCausalLM BlenderbotSmallForConditionalGeneration CamemBert DebertaForMaskedLM DebertaForQuestionAnswering DebertaV2ForMaskedLM DebertaV2ForQuestionAnswering DistilBertForMaskedLM DistilBertForQuestionAnswering DistillGPT2 ElectraForCausalLM ElectraForQuestionAnswering GPT2ForSequenceClassification GoogleFnet LayoutLMForMaskedLM LayoutLMForSequenceClassification M2M100ForConditionalGeneration MBartForCausalLM MBartForConditionalGeneration MT5ForConditionalGeneration MegatronBertForCausalLM MegatronBertForQuestionAnswering MobileBertForMaskedLM MobileBertForQuestionAnswering OPTForCausalLM PLBartForCausalLM PLBartForConditionalGeneration PegasusForCausalLM PegasusForConditionalGeneration RobertaForCausalLM RobertaForQuestionAnswering Speech2Text2ForCausalLM T5ForConditionalGeneration T5Small TrOCRForCausalLM XGLMForCausalLM XLNetLMHeadModel YituTechConvBert"
huggingface_list = huggingface_list.split(" ")
huggingface_collection = "huggingface"

# target_models = "beit_base_patch16_224 cait_m36_384 crossvit_9_240 cspdarknet53 eca_botnext26ts_256 mobilevit_s pnasnet5large sebotnet33ts_256 Background_Matting attention_is_all_you_need_pytorch basic_gnn_sage cm3leon_generate detectron2_fcos_r_50_fpn hf_BigBird hf_GPT2 hf_Longformer hf_T5 llama nanogpt_generate pyhpc_turbulent_kinetic_energy yolov3 AllenaiLongformerBase OPTForCausalLM".split()
target_models = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--mode", type=str, default="inference")
    parser.add_argument("--output", type=str,
                        default=f"final_results_{now}.txt")
    parser.add_argument("--work-dir", type=str, required=True,
                        help="The directory where the pytorch is located")
    parser.add_argument("--profile", action="store_true",
                        help="profile trace path")
    parser.add_argument("--disable_streams", action="store_true",)
    args = parser.parse_args()
    if args.accuracy and args.performance:
        raise ValueError("Please choose either accuracy or performance")
    if not args.accuracy and not args.performance:
        print("No --accuracy or --performance is chosen, default to performance")
        test_accuracy = False
    elif args.accuracy:
        test_accuracy = True
    else:
        test_accuracy = False
    mode = args.mode
    directory = args.work_dir
    os.chdir(directory)
    output_file = args.output
    start_time = datetime.datetime.now()
    with open(output_file, "w") as f:
        f.write(start_time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    # get full path
    print(f"write results to {os.path.abspath(output_file)}")
    if target_models:
        timm_models_list = [_ for _ in timm_models_list if _ in target_models]
        torchbench_list = [_ for _ in torchbench_list if _ in target_models]
        huggingface_list = [_ for _ in huggingface_list if _ in target_models]

    for collection, model_list in zip([timm_models_collection, torchbench_collection, huggingface_collection], [timm_models_list, torchbench_list, huggingface_list]):
        results = run_models(model_list, collection,
                             test_accuracy, mode, args.profile, args.disable_streams)
        write_results(results, output_file)
    end_time = datetime.datetime.now()
    with open(output_file, "a") as f:
        f.write(end_time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        hours = (end_time - start_time).total_seconds() // 3600
        minutes = ((end_time - start_time).total_seconds() % 3600) // 60
        f.write(f"Duration: {hours} hours {minutes} minutes\n")
