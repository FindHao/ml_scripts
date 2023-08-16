import os
import json
import shutil
import subprocess
import time
niter=None
pt_path=None
user_key = os.getenv("PUSHOVER_USER_KEY")
app_key = os.getenv("PUSHOVER_TORCH_KEY")
assert user_key is not None, "PUSHOVER_USER_KEY is not set"
assert app_key is not None, "PUSHOVER_TORCH_KEY is not set"

import requests

def get_system_proxies():
    # Get system proxies. Typically, these might be set in your OS environment.
    return {
        "http": os.environ.get("http_proxy"),
        "https": os.environ.get("https_proxy"),
    }

def notify(message):
    url = "https://api.pushover.net/1/messages.json"
    headers = {
        "Content-type": "application/x-www-form-urlencoded"
    }
    data = {
        "token": app_key,
        "user": user_key,
        "message": message,
    }
    proxies = get_system_proxies()
    response = requests.post(url, headers=headers, data=data, proxies=proxies)
    if response.status_code == 200:
        print("Notification sent successfully!")
    else:
        print(f"Failed to send notification. Status code: {response.status_code}, Message: {response.text}")



def get_latest_folder(path):
    return max([os.path.join(path, d) for d in os.listdir(path) if d.startswith('run')], key=os.path.getctime)

collections = {
    "timm_models": ["xcit_large_24_p8_224", "cait_m36_384", "cspdarknet53", "convit_base"],
    "torchbench": ["hf_BigBird", "drq", "pyhpc_turbulent_kinetic_energy", "doctr_reco_predictor", "pytorch_stargan", "hf_T5", "Super_SloMo", "Background_Matting", "llama", "DALLE2_pytorch", "hf_T5_generate", "nanogpt_generate", "detectron2_fasterrcnn_r_101_c4", "detectron2_fasterrcnn_r_101_dc5", "detectron2_fasterrcnn_r_101_fpn", "detectron2_fasterrcnn_r_50_c4", "detectron2_fasterrcnn_r_50_dc5", "detectron2_fasterrcnn_r_50_fpn", "detectron2_fcos_r_50_fpn", "detectron2_maskrcnn_r_101_c4", "detectron2_maskrcnn_r_101_fpn", "detectron2_maskrcnn_r_50_c4", "detectron2_maskrcnn_r_50_fpn", "resnet18"]
}


def execute_command(model_name, debug_folder_path):
    target_collection = None
    # search for model_name in collections
    for collection_name, collection in collections.items():
        if model_name in collection:
            target_collection = collection_name
            break
    assert target_collection is not None, f"Model {model_name} not found in collections"
    cmd = f"TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_MULTIPLE_STREAMS=1 TORCHINDUCTOR_STREAM_ACCURACY_FIX=1 DEBUG_FOLDER={model_name}_checkpoints TORCH_COMPILE_DEBUG_DIR={debug_folder_path} python benchmarks/dynamo/{target_collection}.py --accuracy --bfloat16 -dcuda --inference --inductor --disable-cudagraphs --only {model_name}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=pt_path)
    stdout, stderr = process.communicate()
    return stdout.decode(), stderr.decode()

def work(model_name, json_path, base_debug_folder):
    while True:
        # check if json_path exists
        if not os.path.exists(json_path):
            original_content = {}
        else:
            with open(json_path, 'r') as f:
                original_content = json.load(f)
        for _ in range(niter):
            if os.path.exists(json_path):
                with open(json_path, 'w') as f:
                    json.dump(original_content, f)
            stdout, stderr = execute_command(model_name, base_debug_folder)
            latest_folder = get_latest_folder(f"{base_debug_folder}/torch_compile_debug")
            # Dump stdout and stderr
            with open(os.path.join(latest_folder, 'stdout.log'), 'w') as f:
                f.write(stdout)
            with open(os.path.join(latest_folder, 'stderr.log'), 'w') as f:
                f.write(stderr)
            # read the latest json file.
            with open(json_path, 'r') as f:
                content = json.load(f)
            # write the latest json file to the latest folder
            with open(os.path.join(latest_folder, 'checkpoint_after.json'), 'w') as f:
                json.dump(content, f)
            # rewrite it with the original content. it is useful when current run fails. we can directly copy it to the global json file to reproduce the error.
            with open(os.path.join(latest_folder, 'checkpoint.json'), 'w') as f:
                json.dump(original_content, f)

            if 'cur_graph' not in original_content:
                with open(os.path.join(latest_folder, 'first_run'), 'w') as f:
                    f.write('This file indicates the first run that is used to generate the stream assignments.')
                if 'pass' not in stdout:
                    return False, f"Error: 'pass' not found in stdout. Check {latest_folder} for details."
                break
            cur_graph = content["cur_graph"]
            this_time_node = content[cur_graph]["this_time_node"]
            print(f"Current graph: {cur_graph}, this_time_node: {this_time_node} pass")
            with open(os.path.join(latest_folder, f"{cur_graph}___{this_time_node}"), 'w') as f:
                f.write('This file indicates the current graph and its time node.')

            if 'pass' not in stdout:
                return False, f"Error: 'pass' not found in stdout. Check {latest_folder} for details."


            if 'finished' in content:
                return True, f"{model_name} passed all accuracy tests!"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--pt_path', type=str, default='/home/users/yhao24/p9_inductor/pytorch')
    parser.add_argument('--niter', type=int, default=10)
    args = parser.parse_args()
    model_name = args.model_name
    niter = args.niter
    pt_path = args.pt_path
    assert os.path.exists(pt_path), f"Path {pt_path} does not exist"
    # Configurable paths
    base_debug_folder = f"/tmp/yhao/{model_name}_debug"
    json_path = os.path.join(base_debug_folder, f"torch_compile_debug/{model_name}_checkpoints/checkpoints.json")
    start_time = time.time()
    success, return_str = work(model_name, json_path, base_debug_folder)
    end_time = time.time()
    duration = end_time - start_time
    # format duration to hours, minutes, seconds
    duration = time.strftime("%H:%M:%S", time.gmtime(duration))
    if success:
        print(f"{model_name} passed all accuracy tests in {duration}!")
        notify(f"{model_name} passed all accuracy tests in {duration}!")
    else:
        print(return_str)
        notify(f"{return_str}. Test takes {duration}.")
