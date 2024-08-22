#!/bin/bash
# simple usage: work_path=/home/yhao/pt ./compile_torch.sh
set -e
work_path=${work_path:-/home/yhao/p9}
# clean_install=1 will remove the existing pytorch folder and re-clone it
# if not, it will just update the existing pytorch and dependent packages
clean_install=${clean_install:-0}
# clean_torch=1 will run python setup.py clean to remove previous pytorch build files
clean_torch=${clean_torch:-0}
torch_only=${torch_only:-0}

# for specific commit or branch 
torch_commit=${torch_commit:-""}
torch_branch=${torch_branch:-"main"}

# torchbench installation takes a long time, so it can be disabled
no_torchbench=${no_torchbench:-0} 
# disable ROCM when working on servers with NVIDIA GPUs and AMD GPUs
export USE_ROCM=0
export USE_NCCL=1
# export ROCR_VISIBLE_DEVICES=3
# export CUDA_VISIBLE_DEVICES=1
# function to check the return value of the previous command
check_return_value() {
    if [ $? -ne 0 ]; then
        echo "Error: $1"
        exit 1
    fi
}

# print configs
echo "work_path: ${work_path}"
echo "clean_install: ${clean_install}"
echo "clean_torch: ${clean_torch}"
echo "torch_only: ${torch_only}"
echo "torch_branch: ${torch_branch}"
echo "torch_commit: ${torch_commit}"


# if you have an error named like version `GLIBCXX_3.4.30' not found, you can add `-c conda-forge` to the following command. And also for your `conda create -n pt_compiled -c conda-forge python=3.10` command

conda install -y magma-cuda121 cache cmake ninja mkl mkl-include libpng libjpeg-turbo graphviz -c pytorch

cd $work_path
if [ $clean_install -eq 1 ]; then
    rm -rf pytorch text vision audio benchmark data
    git clone --recursive git@github.com:pytorch/pytorch.git
    git clone --recursive git@github.com:pytorch/text.git
    git clone --recursive git@github.com:pytorch/data.git
    git clone --recursive git@github.com:pytorch/vision.git
    git clone --recursive git@github.com:pytorch/audio.git
    git clone --recursive git@github.com:pytorch/benchmark.git
fi

cd $work_path/pytorch
git pull
if [ -n "$torch_commit" ]; then
    git checkout $torch_commit
    echo "warnging: you are using a specific commit. don't forget to create a new branch if you want to make changes"
else
    git checkout $torch_branch
fi
git submodule sync
git submodule update --init --recursive
pip install -r requirements.txt
make triton

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
if [ $clean_torch -eq 1 ]; then
    python setup.py clean
fi
python setup.py develop

function notify_finish() {
    if command -v notify &>/dev/null; then
        notify "PyTorch Compilation is done"
    fi
}

if [ $torch_only -eq 1 ]; then
    notify_finish
    exit 0
fi

function upgrade_pack() {
    cd $work_path/$1
    git pull
    git submodule sync
    git submodule update --init --recursive
    pip uninstall -y $1
    python setup.py clean
    python setup.py install
    echo "$1 installation is done"
}

# install torchdata
cd $work_path
upgrade_pack data

# install torchtext
cd $work_path
export CC=$(which gcc)
export CXX=$(which g++)
upgrade_pack text

# install torchvision
export FORCE_CUDA=1
upgrade_pack vision

# install torchaudio
upgrade_pack audio

if [ $no_torchbench -eq 1 ]; then
    notify_finish
    exit 0
fi
# install torchbench
pip install pyyaml
cd $work_path/benchmark
git pull
git submodule sync
git submodule update --init --recursive
python install.py
echo "torchbench installation is done"
notify_finish