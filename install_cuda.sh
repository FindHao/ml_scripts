#!/bin/bash
# usage: t_cuda=118 t_cuda_path=/install/path/cuda-11.8  ./install_cuda.sh

# cudnn official redist link
# https://developer.download.nvidia.com/compute/redist/cudnn/
# new pytorch official cudnn version https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_cudnn.sh
# another useful script: https://github.com/pytorch/builder/blob/main/common/install_cuda.sh
# for variable in t_cuda t_cudnn t_cuda_path
# do 
#     if [ -z "${!variable}" ]; then
#         echo "${variable} not set"
#         exit 1
#     fi
# done

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
    cuda_download_link="https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run"
    cuda_file_name="cuda_12.1.1_530.30.02_linux.run"
    cudnn_file_name="cudnn-linux-x86_64-8.9.2.26_cuda12-archive"
    cudnn_download_link="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/${cudnn_file_name}.tar.xz"
    cudnn_file_name_with_ext="${cudnn_file_name}.tar.xz"
    # https://developer.download.nvidia.com/compute/redist/nccl/v2.20.5/nccl_2.20.5-1+cuda12.2_x86_64.txz
    nccl_file_name="nccl_2.20.5-1+cuda12.2_x86_64"
    nccl_download_link="https://developer.download.nvidia.com/compute/redist/nccl/v2.20.5/${nccl_file_name}.txz"
    nccl_file_name_with_ext="${nccl_file_name}.txz"
fi

if [ "$t_cuda" == "124" ] ; then
    cuda_download_link="https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run"
    cuda_file_name="cuda_12.4.0_550.54.14_linux.run"
    cudnn_file_name="cudnn-linux-x86_64-8.9.2.26_cuda12-archive"
    cudnn_download_link="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/${cudnn_file_name}.tar.xz"
    cudnn_file_name_with_ext="${cudnn_file_name}.tar.xz"
    nccl_branch="v2.20.5-1"
    cusparselt_version="052"
fi

function compile_nccl() {
    git clone -b ${nccl_branch} --depth 1 https://github.com/NVIDIA/nccl.git nccl-${nccl_branch}
    cd nccl-${nccl_branch}
    make -j src.build
    cp -a build/include/* $t_cuda_path/include/
    cp -a build/lib/* $t_cuda_path/lib64/
    cd ..
    rm -rf nccl-${nccl_branch}
}


function install_cusparselt_052 {
    # cuSparseLt license: https://docs.nvidia.com/cuda/cusparselt/license.html
    mkdir tmp_cusparselt && cd tmp_cusparselt
    wget -q https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.5.2.1-archive.tar.xz
    tar xf libcusparse_lt-linux-x86_64-0.5.2.1-archive.tar.xz
    cp -a libcusparse_lt-linux-x86_64-0.5.2.1-archive/include/* $t_cuda_path/include/
    cp -a libcusparse_lt-linux-x86_64-0.5.2.1-archive/lib/* $t_cuda_path/lib64/
    cd ..
    rm -rf tmp_cusparselt
}


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
    if [ ! -z "$nccl_download_link" ]; then
        echo "Downloading nccl"
        wget -c $nccl_download_link -O .downloads/$t_cuda/$nccl_file_name_with_ext
        tar -xf .downloads/$t_cuda/$nccl_file_name_with_ext -C .downloads/$t_cuda/
        cp -r .downloads/$t_cuda/$nccl_file_name/include/* $t_cuda_path/include/
        cp -r .downloads/$t_cuda/$nccl_file_name/lib/* $t_cuda_path/lib64/
        chmod a+r $t_cuda_path/include/nccl*.h $t_cuda_path/lib64/libnccl*
    fi
    if [ ! -z "$nccl_branch" ]; then
        compile_nccl
    fi
    if [ ! -z "$cusparselt_version" ]; then
        eval install_cusparselt_${cusparselt_version}
    fi
}

download_and_install
