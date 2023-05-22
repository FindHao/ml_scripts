#!/bin/bash

work_path=${work_path:-/scratch/yhao24/p9_inductor}
# disable ROCM when working on servers with NVIDIA GPUs and AMD GPUs
export USE_ROCM=0
export USE_NCCL=1

conda install -c pytorch magma-cuda121
conda install cmake ninja
conda install mkl mkl-include
conda install libpng

# !!! warning need to use same numpy version with torchbench!!!!!

cd $work_path/benchmark
pip install -r requirements.txt

git clone --recursive git@github.com:pytorch/pytorch.git
cd pytorch
git submodule sync
git submodule update --init --recursive
pip install -r requirements.txt
make triton

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop

# install torchtext
cd $work_path
git clone git@github.com:pytorch/text.git
cd text
git submodule update --init --recursive
python setup.py clean install

# install torchvision
cd $work_path
git clone git@github.com:pytorch/vision.git
cd vision
python setup.py install

# install torchaudio
cd $work_path
git clone git@github.com:pytorch/audio.git
cd audio
python setup.py install