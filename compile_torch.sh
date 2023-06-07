#!/bin/bash

work_path=${work_path:-/scratch/yhao24/p9_inductor}
# disable ROCM when working on servers with NVIDIA GPUs and AMD GPUs
export USE_ROCM=0
export USE_NCCL=1
export ROCR_VISIBLE_DEVICES=3
export CUDA_VISIBLE_DEVICES=1

conda install -c pytorch magma-cuda121 -y
conda install -y cmake ninja mkl mkl-include libpng -y

# !!! warning need to use same numpy version with torchbench!!!!! 
pip install numpy==1.21.2

git clone --recursive git@github.com:pytorch/pytorch.git
cd pytorch
git submodule sync
git submodule update --init --recursive
pip install -r requirements.txt
make triton

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop

# install torchdata
cd $work_path
git clone git@github.com:pytorch/data.git
cd data
git submodule update --init --recursive
pip uninstall -y  torchdata
python setup.py clean
python setup.py install

# install torchtext
cd $work_path
git clone git@github.com:pytorch/text.git
cd text
git submodule update --init --recursive
pip uninstall -y  torchtext
python setup.py clean
python setup.py install


# install torchvision
# FORCE_CUDA doesn't work
export FORCE_CUDA=TRUE
cd $work_path
git clone git@github.com:pytorch/vision.git
cd vision
git submodule update --init --recursive
pip uninstall -y torchvision
python setup.py clean
python setup.py install

# install torchaudio
cd $work_path
git clone git@github.com:pytorch/audio.git
cd audio
git submodule update --init --recursive
pip uninstall -y torchaudio
python setup.py clean
python setup.py install

cd $work_path
git clone git@github.com:pytorch/benchmark.git
pip install pyyaml
cd benchmark
python install.py