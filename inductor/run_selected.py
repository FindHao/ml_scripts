import subprocess
import time
import re
import os
import subprocess
import time
import re
import os

def run_models(model_list, directory, collections):
    # change current working directory
    os.chdir(directory)  # replace with your actual path

    result_dict = {}

    for model in model_list:
        command = f'python benchmarks/dynamo/{collections}.py --performance --amp -dcuda --inference --inductor --disable-cudagraphs --only {model}'
        start_time = time.time()

        # Opening a file to write the output
        with open("temp.txt", "w") as outfile:
            # Running the command
            process = subprocess.Popen(command, stdout=outfile, stderr=subprocess.PIPE, shell=True)

            while True:
                if process.poll() is not None:  # process finished
                    stderr = process.stderr.read().decode('utf-8')  # decoding the output

                    if process.returncode != 0:  # process ended with an error
                        result_dict[model] = f"error is {stderr}"
                    else:
                        with open("temp.txt", "r") as infile:
                            stdout = infile.read()
                            # Finding the required string
                            match = re.search(r'(\d+?\.\d+?)x', stdout)
                            if match:
                                result_dict[model] = float(match.group(1))  # storing the float value
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
        if isinstance(results[key], float):
            successed.append(key)
        else:
            failed.append(key)
    with open(output_file, "w") as f:
        f.write("Successed:\n")
        for model in successed:
            f.write(f"{model}, {results[model]}\n")
        f.write("\nFailed:\n")
        for model in failed:
            f.write(f"{model}, {results[model]}\n")


directory = '/home/users/yhao24/p9_inductor/pytorch'  # replace with your actual directory

output_file = "/home/users/yhao24/p9_inductor/pytorch/torch_compile_debug/tmp/final_results.txt"

model_list = "SelecSls42b coat_lite_mini tnt_s_patch16_224 deit_base_distilled_patch16_224 jx_nest_base vit_base_patch16_224 res2net101_26w_4s mobilevit_s repvgg_a2 tf_mixnet_l convit_base resmlp_12_224 levit_128 volo_d1_224 cait_m36_384 swin_base_patch4_window7_224 res2net50_14w_8s mixnet_l gmixer_24_224 res2next50 convmixer_768_32 regnety_002 adv_inception_v3 xcit_large_24_p8_224".split()
model_list = "SelecSls42b adv_inception_v3 beit_base_patch16_224 botnet26t_256 cait_m36_384 coat_lite_mini convit_base convmixer_768_32 convnext_base crossvit_9_240 cspdarknet53 deit_base_distilled_patch16_224 dla102 dm_nfnet_f0 dpn107 eca_botnext26ts_256 eca_halonext26ts ese_vovnet19b_dw fbnetc_100 fbnetv3_b gernet_l ghostnet_100 gluon_inception_v3 gmixer_24_224 gmlp_s16_224 hrnet_w18 inception_v3 jx_nest_base lcnet_050 levit_128 mixer_b16_224 mixnet_l mnasnet_100 mobilenetv2_100 mobilenetv3_large_100 mobilevit_s nfnet_l0 pit_b_224 pnasnet5large poolformer_m36 regnety_002 repvgg_a2 res2net101_26w_4s res2net50_14w_8s res2next50 resmlp_12_224 resnest101e rexnet_100 sebotnet33ts_256 spnasnet_100 swin_base_patch4_window7_224 swsl_resnext101_32x16d tf_efficientnet_b0 tf_mixnet_l tinynet_a tnt_s_patch16_224 twins_pcpvt_base vit_base_patch16_224 volo_d1_224 xcit_large_24_p8_224"
collection="timm_models"
results = run_models(model_list, directory, "timm_models")
write_results(results, output_file)

model_list = "hf_T5_large hf_Albert hf_Longformer maml speech_transformer demucs vision_maskrcnn nanogpt_generate hf_Bert pytorch_CycleGAN_and_pix2pix Super_SloMo timm_vision_transformer functorch_dp_cifar10 nvidia_deeprecommender hf_T5 DALLE2_pytorch llama cm3leon_generate timm_efficientdet hf_Bert_large pytorch_stargan fastNLP_Bert detectron2_fcos_r_50_fpn functorch_maml_omniglot moco attention_is_all_you_need_pytorch pyhpc_turbulent_kinetic_energy timm_vision_transformer_large".split()
model_list = "BERT_pytorch Background_Matting DALLE2_pytorch LearningToPaint Super_SloMo alexnet attention_is_all_you_need_pytorch basic_gnn_edgecnn basic_gnn_gcn basic_gnn_gin basic_gnn_sage cm3leon_generate dcgan demucs densenet121 detectron2_fcos_r_50_fpn dlrm doctr_det_predictor doctr_reco_predictor drq fastNLP_Bert functorch_dp_cifar10 functorch_maml_omniglot hf_Albert hf_Bart hf_Bert hf_Bert_large hf_BigBird hf_DistilBert hf_GPT2 hf_GPT2_large hf_Longformer hf_Reformer hf_T5 hf_T5_generate hf_T5_large lennard_jones llama maml maml_omniglot mnasnet1_0 mobilenet_v2 mobilenet_v3_large moco nanogpt_generate nvidia_deeprecommender opacus_cifar10 phlippe_densenet phlippe_resnet pyhpc_equation_of_state pyhpc_isoneutral_mixing pyhpc_turbulent_kinetic_energy pytorch_CycleGAN_and_pix2pix pytorch_stargan pytorch_unet resnet152 resnet18 resnet50 resnext50_32x4d shufflenet_v2_x1_0 soft_actor_critic speech_transformer squeezenet1_1 timm_efficientdet timm_efficientnet timm_nfnet timm_regnet timm_resnest timm_vision_transformer timm_vision_transformer_large timm_vovnet tts_angular vgg16 vision_maskrcnn yolov3"
collection="torchbench"
results = run_models(model_list, directory, "torchbench")
write_results(results, output_file)


model_list = "MBartForConditionalGeneration BlenderbotForCausalLM MegatronBertForQuestionAnswering DebertaV2ForQuestionAnswering BartForCausalLM AlbertForMaskedLM MobileBertForMaskedLM MobileBertForQuestionAnswering YituTechConvBert AllenaiLongformerBase XLNetLMHeadModel MT5ForConditionalGeneration MegatronBertForCausalLM".split()
collection="huggingface"
model_list = "AlbertForMaskedLM AlbertForQuestionAnswering AllenaiLongformerBase BartForCausalLM BartForConditionalGeneration BertForMaskedLM BertForQuestionAnswering BlenderbotForCausalLM BlenderbotSmallForCausalLM BlenderbotSmallForConditionalGeneration CamemBert DebertaForMaskedLM DebertaForQuestionAnswering DebertaV2ForMaskedLM DebertaV2ForQuestionAnswering DistilBertForMaskedLM DistilBertForQuestionAnswering DistillGPT2 ElectraForCausalLM ElectraForQuestionAnswering GPT2ForSequenceClassification GoogleFnet LayoutLMForMaskedLM LayoutLMForSequenceClassification M2M100ForConditionalGeneration MBartForCausalLM MBartForConditionalGeneration MT5ForConditionalGeneration MegatronBertForCausalLM MegatronBertForQuestionAnswering MobileBertForMaskedLM MobileBertForQuestionAnswering OPTForCausalLM PLBartForCausalLM PLBartForConditionalGeneration PegasusForCausalLM PegasusForConditionalGeneration RobertaForCausalLM RobertaForQuestionAnswering Speech2Text2ForCausalLM T5ForConditionalGeneration T5Small TrOCRForCausalLM XGLMForCausalLM XLNetLMHeadModel YituTechConvBert"
results = run_models(model_list, directory, "huggingface")
write_results(results, output_file)