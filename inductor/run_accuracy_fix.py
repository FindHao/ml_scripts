import os
import json
import subprocess

niter=1

def get_latest_folder(path):
    return max([os.path.join(path, d) for d in os.listdir(path) if d.startswith('run')], key=os.path.getctime)


def execute_command(model_name, debug_folder_path):
    cmd = f"TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_MULTIPLE_STREAMS=1 TORCHINDUCTOR_STREAM_ACCURACY_FIX=1 DEBUG_FOLDER={model_name}_checkpoints TORCH_COMPILE_DEBUG_DIR={debug_folder_path} python benchmarks/dynamo/torchbench.py --accuracy --bfloat16 -dcuda --inference --inductor --disable-cudagraphs --only {model_name}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="/home/users/yhao24/p9_inductor/pytorch")
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
            with open(os.path.join(latest_folder, 'checkpoint.json'), 'w') as f:
                json.dump(original_content, f)

            if 'pass' not in stdout:
                print(f"Error: 'pass' not found in stdout. Check {latest_folder} for details.")
                return

            # Checkpoints.json
            with open(json_path, 'r') as f:
                content = json.load(f)
            # breakpoint()
            if 'finished' in content:
                print(f"{model_name} passed all accuracy tests!")
                return

            cur_graph = content["cur_graph"]
            this_time_node = content[cur_graph]["this_time_node"]
            print(f"Current graph: {cur_graph}, this_time_node: {this_time_node} pass")
            with open(os.path.join(latest_folder, f"{cur_graph}___{this_time_node}"), 'w') as f:
                f.write('This file indicates the current graph and its time node.')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet18')
    args = parser.parse_args()
    model_name = args.model_name
    # Configurable paths
    base_debug_folder = f"/tmp/yhao/{model_name}_debug"
    json_path = os.path.join(base_debug_folder, f"torch_compile_debug/{model_name}_checkpoints/checkpoints.json")
    work(model_name, json_path, base_debug_folder)
