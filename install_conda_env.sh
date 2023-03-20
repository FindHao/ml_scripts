#!/bin/bash
# the conda environment you want to create
conda_env=${conda_env:-pt_jan02}
# where the python wheels are downloaded
wheel_path=${wheel_path:-/home/yhao/d/p8/ml_optimizations/.downloads/20230102}
# where you want to install torchbench
work_path=${work_path:-/home/yhao/d/p8/opts}
# where the conda is installed
conda_dir=${conda_dir:-/home/yhao/d/conda}
# whether to install torchbench
enable_torchbench=${enable_torchbench:-1}
# cuda_path
cuda_env=${cuda_env:-/data/yhao/setenvs/cuda11.6.sh}
# python_version
python_version=${python_version:-3.10}
# install stable version
stable_version=${stable_version:-0}
# where torchbench is installed
torhcbench_path=${torchbench_path:-${work_path}/benchmark_dev2}

source ${conda_dir}/bin/activate
source ${cuda_env}

check_folder_exist() {
    if [ ! -d $1 ]; then
        echo "$1 not exist"
        exit 1
    fi
}

# if stable_version is 1, don't need to check wheel_path
if [ ${stable_version} -eq 0 ]; then
    check_folder_exist $wheel_path
fi
check_folder_exist $work_path

# check if the environment is already installed
if [ $(conda env list | grep -c ${conda_env}) -ne 0 ]; then
    echo "Conda environment ${conda_env} already exists. Will use it directly."
else
    echo "Conda environment ${conda_env} does not exist. Will create it."
    # create the conda environment
    conda create -y -n ${conda_env} python=${python_version}
fi

conda activate ${conda_env}
conda install -y git-lfs
pip install requests bs4 argparse oauthlib pyyaml

# install pytorch
if [ ${stable_version} -eq 1 ]; then
    echo "Install stable version of pytorch"
    pip3 install torch torchvision torchaudio torchtext
else
    pip install ${wheel_path}/*.whl
fi

# check if the last command is successful
if [ $? -ne 0 ]; then
    echo "Failed to install pytorch. Exit."
    exit 1
fi

if [ ${enable_torchbench} -ne 1 ]; then
    echo "Skip torchbench installation."
    exit 0
fi

cd ${work_path}
# install torchbench
git clone --recursive git@github.com:pytorch/benchmark.git ${torhcbench_path}
cd ${torhcbench_path}
pip install pyyaml
python install.py
