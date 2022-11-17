#!/bin/bash

# check variable exsitence
if [ -z "$cuda116_path" ]; then
    echo "cuda116_path is not set"
    exit 1
fi
if [ -z "$cuda117_path" ]; then
    echo "cuda117_path is not set"
    exit 1
fi

wget -q https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run -P ./.downloads/cu116
cd ./.downloads/cu116
chmod +x cuda_11.6.2_510.47.03_linux.run
./cuda_11.6.2_510.47.03_linux.run --toolkit --toolkitpath=${cuda116_path} --silent
wget -q https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.2/local_installers/11.5/cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz -O cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz
tar xf cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz
cuda_path=${cuda116_path} cudnn_path=$(pwd)/cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive ./install_cudnn.sh

wget -q https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run -P ./.downloads/cu117
cd ./.downloads/cu117
chmod +x cuda_11.7.0_515.43.04_linux.run
./cuda_11.7.0_515.43.04_linux.run --toolkit --toolkitpath=${cuda117_path} --silent
wget -q https://ossci-linux.s3.amazonaws.com/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz -O cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
tar xf cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
cuda_path=${cuda117_path} cudnn_path=$(pwd)/cudnn-linux-x86_64-8.5.0.96_cuda11-archive ./install_cudnn.sh




