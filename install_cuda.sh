#!/bin/bash
# usage: t_cuda=118 t_cuda_path=/install/path/cuda-11.8  ./install_cuda.sh

# cudnn official redist link
# https://developer.download.nvidia.com/compute/redist/cudnn/

# for variable in t_cuda t_cudnn t_cuda_path
# do 
#     if [ -z "${!variable}" ]; then
#         echo "${variable} not set"
#         exit 1
#     fi
# done

# check t_cuda is equal to 116 or 117 or 118
if [ "$t_cuda" == "116" ] ; then
    cuda_download_link="https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run"
    cudnn_download_link="https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.2/local_installers/11.5/cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz"
    cuda_file_name="cuda_11.6.2_510.47.03_linux.run"
    cudnn_file_name="cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive"
    cudnn_file_name_with_ext="cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz"
fi
if [ "$t_cuda" == "117" ] ; then
    cuda_download_link="https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run"
    cudnn_download_link="https://developer.download.nvidia.com/compute/redist/cudnn/v8.5.0/local_installers/11.7/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz"
    cuda_file_name="cuda_11.7.0_515.43.04_linux.run"
    cudnn_file_name="cudnn-linux-x86_64-8.5.0.96_cuda11-archive"
    cudnn_file_name_with_ext="cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz"
fi
if [ "$t_cuda" == "118" ] ; then
    cuda_download_link="https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run"
    cudnn_download_link="https://developer.download.nvidia.com/compute/redist/cudnn/v8.5.0/local_installers/11.7/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz"
    cuda_file_name="cuda_11.8.0_520.61.05_linux.run"
    cudnn_file_name="cudnn-linux-x86_64-8.5.0.96_cuda11-archive"
    cudnn_file_name_with_ext="cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz"
fi
if [ "$t_cuda" == "1187" ] ; then
    cuda_download_link="https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run"
    cudnn_download_link="https://developer.download.nvidia.com/compute/redist/cudnn/v8.7.0/local_installers/11.8/cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz"
    cuda_file_name="cuda_11.8.0_520.61.05_linux.run"
    cudnn_file_name="cudnn-linux-x86_64-8.7.0.84_cuda11-archive"
    cudnn_file_name_with_ext=$cudnn_file_name."tar.xz"
fi
if [ "$t_cuda" == "121" ] ; then
    cuda_download_link="https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run"
    cudnn_download_link="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.8.1.3_cuda12-archive.tar.xz"
    cuda_file_name="cuda_12.1.0_530.30.02_linux.run"
    cudnn_file_name="cudnn-linux-x86_64-8.8.1.3_cuda12-archive"
    cudnn_file_name_with_ext="cudnn-linux-x86_64-8.8.1.3_cuda12-archive.tar.xz"
fi

if [ -z "$cuda_download_link" ]; then
    echo "t_cuda is only available for 116, 117, 118, 1187, 121"
    exit 1
fi
if [ -z "$t_cuda_path" ]; then
    echo "t_cuda_path not set"
    exit 1
fi

echo "This script will install cuda $t_cuda to $t_cuda_path."

mkdir -p .downloads/$t_cuda

function download_and_install() {
    echo "Downloading cudatoolkit"
    wget -c $cuda_download_link -O .downloads/$t_cuda/$cuda_file_name
    echo "Downloading cudnn"
    wget -c $cudnn_download_link -O .downloads/$t_cuda/$cudnn_file_name_with_ext &
    /usr/bin/bash .downloads/$t_cuda/$cuda_file_name --silent --toolkit --toolkitpath=$t_cuda_path 
    wait
    mkdir .downloads/$t_cuda/$cudnn_file_name
    tar -xf .downloads/$t_cuda/$cudnn_file_name_with_ext -C .downloads/$t_cuda/
    cp .downloads/$t_cuda/$cudnn_file_name/include/cudnn*.h $t_cuda_path/include
    cp .downloads/$t_cuda/$cudnn_file_name/lib/libcudnn* $t_cuda_path/lib64
    chmod a+r $t_cuda_path/include/cudnn*.h $t_cuda_path/lib64/libcudnn*
}

download_and_install
