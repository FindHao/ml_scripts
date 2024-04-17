#!/bin/bash

work_path=${work_path:-/scratch/yhao24/p9_inductor}
# disable ROCM when working on servers with NVIDIA GPUs and AMD GPUs
export USE_ROCM=0
export USE_NCCL=1
export ROCR_VISIBLE_DEVICES=3
export CUDA_VISIBLE_DEVICES=1

# write a function to check the return value of the previous command
check_return_value() {
    if [ $? -ne 0 ]; then
        echo "Error: $1"
        exit 1
    fi
}

# if you have an error named like version `GLIBCXX_3.4.30' not found, you can add `-c conda-forge` to the following command. And also for your `conda create -n pt_compiled -c conda-forge python=3.10` command

conda install -y magma-cuda121 cmake ninja mkl mkl-include libpng libjpeg-turbo graphviz -c pytorch

# !!! warning need to use same numpy version with torchbench!!!!! 
# https://github.com/pytorch/benchmark/blob/main/requirements.txt
pip install numpy==1.23.5
cd $work_path
git clone --recursive git@github.com:pytorch/pytorch.git
cd pytorch
git submodule sync
git submodule update --init --recursive
pip install -r requirements.txt
make triton

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop

check_return_value "pytorch installation failed"

# install torchdata
cd $work_path
git clone git@github.com:pytorch/data.git
cd data
git submodule update --init --recursive
pip uninstall -y  torchdata
python setup.py clean
python setup.py install
check_return_value "torchdata installation failed"
echo "pytorch installation is done"

# install torchtext
cd $work_path
git clone git@github.com:pytorch/text.git
cd text
git submodule update --init --recursive
pip uninstall -y  torchtext
python setup.py clean
python setup.py install
check_return_value "torchtext installation failed"
echo "torchtext installation is done"


# install torchvision
export FORCE_CUDA=TRUE
cd $work_path
git clone git@github.com:pytorch/vision.git
cd vision
git submodule update --init --recursive
pip uninstall -y torchvision
python setup.py clean
python setup.py install
check_return_value "torchvision installation failed"
echo "torchvision installation is done"

# install torchaudio
cd $work_path
git clone git@github.com:pytorch/audio.git
cd audio
git submodule update --init --recursive
pip uninstall -y torchaudio
python setup.py clean
python setup.py install
check_return_value "torchaudio installation failed"
echo "torchaudio installation is done"

cd $work_path
git clone git@github.com:pytorch/benchmark.git
pip install pyyaml
cd benchmark
python install.py
check_return_value "torchbench installation failed"
echo "torchbench installation is done"
