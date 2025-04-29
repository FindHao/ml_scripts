#!/bin/bash
# Usage: CUDA_INSTALL_PREFIX=/home/yhao/opt ./install_cuda.sh 12.8
# Notice: Part of this script is synced with https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_cuda.sh
set -ex

# set a user specific tmp directory. this can avoid the segmentation fault issue caused
# by /tmp/cuda-installer.log permission issue.
USER_TMPDIR="${HOME}/tmp/cuda_install"
mkdir -p "${USER_TMPDIR}"
export TMPDIR="${USER_TMPDIR}"

CUDNN_VERSION=9.5.1.17

# Make cuda install path configurable. By default, it is /usr/local.
CUDA_INSTALL_PREFIX=${CUDA_INSTALL_PREFIX:-$HOME/opt}
# Remove trailing slash if present
CUDA_INSTALL_PREFIX=${CUDA_INSTALL_PREFIX%/}
SKIP_PRUNE=${SKIP_PRUNE:-1}
CUDA_VERSION=${CUDA_VERSION:-12.8}

# Clean up any leftover temporary directories from previous failed installations
cleanup_temp_dirs() {
  echo "Debug: Starting cleanup"
  [ -d "tmp_cusparselt" ] && rm -rf tmp_cusparselt
  [ -d "tmp_cudnn" ] && rm -rf tmp_cudnn
  [ -d "nccl" ] && rm -rf nccl
  echo "Debug: Cleanup finished"
}

cleanup_temp_dirs
echo "Debug: After cleanup"

function install_cusparselt {
  # cuSparseLt license: https://docs.nvidia.com/cuda/cusparselt/license.html
  mkdir tmp_cusparselt && pushd tmp_cusparselt

  if [[ ${CUDA_VERSION:0:4} =~ ^12\.[5-8]$ ]]; then
    arch_path='sbsa'
    export TARGETARCH=${TARGETARCH:-$(uname -m)}
    if [ ${TARGETARCH} = 'amd64' ] || [ "${TARGETARCH}" = 'x86_64' ]; then
      arch_path='x86_64'
    fi
    CUSPARSELT_NAME="libcusparse_lt-linux-${arch_path}-0.6.3.2-archive"
    curl --retry 3 -OLs https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-${arch_path}/${CUSPARSELT_NAME}.tar.xz
  elif [[ ${CUDA_VERSION:0:4} == "12.4" ]]; then
    arch_path='sbsa'
    export TARGETARCH=${TARGETARCH:-$(uname -m)}
    if [ ${TARGETARCH} = 'amd64' ] || [ "${TARGETARCH}" = 'x86_64' ]; then
      arch_path='x86_64'
    fi
    CUSPARSELT_NAME="libcusparse_lt-linux-${arch_path}-0.6.2.3-archive"
    curl --retry 3 -OLs https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-${arch_path}/${CUSPARSELT_NAME}.tar.xz
  elif [[ ${CUDA_VERSION:0:4} == "11.8" ]]; then
    CUSPARSELT_NAME="libcusparse_lt-linux-x86_64-0.4.0.7-archive"
    curl --retry 3 -OLs https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/${CUSPARSELT_NAME}.tar.xz
  else
    echo "Not sure which libcusparselt version to install for this ${CUDA_VERSION}"
    exit 1
  fi

  tar xf ${CUSPARSELT_NAME}.tar.xz
  cp -a ${CUSPARSELT_NAME}/include/* ${CUDA_INSTALL_PREFIX}/cuda/include/
  cp -a ${CUSPARSELT_NAME}/lib/* ${CUDA_INSTALL_PREFIX}/cuda/lib64/
  popd
  rm -rf tmp_cusparselt
}

function install_nccl {
  NCCL_VERSION=""
  if [[ ${CUDA_VERSION:0:2} == "11" ]]; then
    NCCL_VERSION=$(curl -s https://github.com/pytorch/pytorch/raw/refs/heads/main/.ci/docker/ci_commit_pins/nccl-cu11.txt)
  elif [[ ${CUDA_VERSION:0:2} == "12" ]]; then
    NCCL_VERSION=$(curl -s https://github.com/pytorch/pytorch/raw/refs/heads/main/.ci/docker/ci_commit_pins/nccl-cu12.txt)
  else
    echo "Unexpected CUDA_VERSION ${CUDA_VERSION}"
    exit 1
  fi

  if [[ -n "${NCCL_VERSION}" ]]; then
    # NCCL license: https://docs.nvidia.com/deeplearning/nccl/#licenses
    # Follow build: https://github.com/NVIDIA/nccl/tree/master?tab=readme-ov-file#build
    git clone -b $NCCL_VERSION --depth 1 https://github.com/NVIDIA/nccl.git
    pushd nccl
    make -j src.build CUDA_HOME=${CUDA_INSTALL_PREFIX}/cuda
    cp -a build/include/* ${CUDA_INSTALL_PREFIX}/cuda/include/
    cp -a build/lib/* ${CUDA_INSTALL_PREFIX}/cuda/lib64/
    popd
    rm -rf nccl
    if [ "$(id -u)" -eq 0 ]; then
      ldconfig
    fi
  fi
}

function install_118 {
  CUDNN_VERSION=9.1.0.70
  echo "Installing CUDA 11.8 and cuDNN ${CUDNN_VERSION} and NCCL and cuSparseLt"
  rm -rf ${CUDA_INSTALL_PREFIX}/cuda-11.8 ${CUDA_INSTALL_PREFIX}/cuda
  # install CUDA 11.8.0 in the same container
  wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run -O cuda_11.8.0_520.61.05_linux.run
  chmod +x cuda_11.8.0_520.61.05_linux.run
  ./cuda_11.8.0_520.61.05_linux.run --toolkit --silent --toolkitpath=${CUDA_INSTALL_PREFIX}/cuda-${CUDA_VERSION}
  rm -f cuda_11.8.0_520.61.05_linux.run
  rm -f ${CUDA_INSTALL_PREFIX}/cuda && ln -s ${CUDA_INSTALL_PREFIX}/cuda-11.8 ${CUDA_INSTALL_PREFIX}/cuda

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  mkdir tmp_cudnn && cd tmp_cudnn
  wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda11-archive.tar.xz -O cudnn-linux-x86_64-${CUDNN_VERSION}_cuda11-archive.tar.xz
  tar xf cudnn-linux-x86_64-${CUDNN_VERSION}_cuda11-archive.tar.xz
  cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda11-archive/include/* ${CUDA_INSTALL_PREFIX}/cuda/include/
  cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda11-archive/lib/* ${CUDA_INSTALL_PREFIX}/cuda/lib64/
  cd ..
  rm -rf tmp_cudnn

  install_nccl
  install_cusparselt

  if [ "$(id -u)" -eq 0 ]; then
    ldconfig
  fi
}

function install_124 {
  CUDNN_VERSION=9.1.0.70
  echo "Installing CUDA 12.4.1 and cuDNN ${CUDNN_VERSION} and NCCL and cuSparseLt"
  rm -rf ${CUDA_INSTALL_PREFIX}/cuda-12.4 ${CUDA_INSTALL_PREFIX}/cuda
  # install CUDA 12.4.1 in the same container
  wget -q https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run -O cuda_12.4.1_550.54.15_linux.run
  chmod +x cuda_12.4.1_550.54.15_linux.run
  ./cuda_12.4.1_550.54.15_linux.run --toolkit --silent --toolkitpath=${CUDA_INSTALL_PREFIX}/cuda-${CUDA_VERSION}
  rm -f cuda_12.4.1_550.54.15_linux.run
  rm -f ${CUDA_INSTALL_PREFIX}/cuda && ln -s ${CUDA_INSTALL_PREFIX}/cuda-12.4 ${CUDA_INSTALL_PREFIX}/cuda

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  mkdir tmp_cudnn && cd tmp_cudnn
  wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz -O cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
  tar xf cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
  cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/include/* ${CUDA_INSTALL_PREFIX}/cuda/include/
  cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/lib/* ${CUDA_INSTALL_PREFIX}/cuda/lib64/
  cd ..
  rm -rf tmp_cudnn

  install_nccl
  install_cusparselt

  if [ "$(id -u)" -eq 0 ]; then
    ldconfig
  fi
}

function install_126 {
  echo "Installing CUDA 12.6.3 and cuDNN ${CUDNN_VERSION} and NCCL and cuSparseLt"
  rm -rf ${CUDA_INSTALL_PREFIX}/cuda-12.6 ${CUDA_INSTALL_PREFIX}/cuda
  # install CUDA 12.6.3 in the same container
  wget -q https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run -O cuda_12.6.3_560.35.05_linux.run
  chmod +x cuda_12.6.3_560.35.05_linux.run
  ./cuda_12.6.3_560.35.05_linux.run --toolkit --silent --toolkitpath=${CUDA_INSTALL_PREFIX}/cuda-${CUDA_VERSION}
  rm -f cuda_12.6.3_560.35.05_linux.run
  rm -f ${CUDA_INSTALL_PREFIX}/cuda && ln -s ${CUDA_INSTALL_PREFIX}/cuda-12.6 ${CUDA_INSTALL_PREFIX}/cuda

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  mkdir tmp_cudnn && cd tmp_cudnn
  wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz -O cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
  tar xf cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
  cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/include/* ${CUDA_INSTALL_PREFIX}/cuda/include/
  cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/lib/* ${CUDA_INSTALL_PREFIX}/cuda/lib64/
  cd ..
  rm -rf tmp_cudnn

  install_nccl
  install_cusparselt

  # Only run ldconfig if we are root
  if [ "$(id -u)" -eq 0 ]; then
    ldconfig
  fi
}

function install_128 {
  CUDNN_VERSION=9.8.0.87
  echo "Installing CUDA 12.8.0 and cuDNN ${CUDNN_VERSION} and NCCL and cuSparseLt"
  rm -rf ${CUDA_INSTALL_PREFIX}/cuda-12.8 ${CUDA_INSTALL_PREFIX}/cuda
  # install CUDA 12.8.0 in the same container
  wget -q https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
  chmod +x cuda_12.8.0_570.86.10_linux.run
  ./cuda_12.8.0_570.86.10_linux.run --toolkit --silent --toolkitpath=${CUDA_INSTALL_PREFIX}/cuda-${CUDA_VERSION}
  rm -f cuda_12.8.0_570.86.10_linux.run
  rm -f ${CUDA_INSTALL_PREFIX}/cuda && ln -s ${CUDA_INSTALL_PREFIX}/cuda-12.8 ${CUDA_INSTALL_PREFIX}/cuda

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  mkdir tmp_cudnn && cd tmp_cudnn
  wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz -O cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
  tar xf cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
  cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/include/* ${CUDA_INSTALL_PREFIX}/cuda/include/
  cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/lib/* ${CUDA_INSTALL_PREFIX}/cuda/lib64/
  cd ..
  rm -rf tmp_cudnn

  install_nccl
  install_cusparselt

  if [ "$(id -u)" -eq 0 ]; then
    ldconfig
  fi
}

function prune_118 {
  echo "Pruning CUDA 11.8 and cuDNN"
  #####################################################################################
  # CUDA 11.8 prune static libs
  #####################################################################################
  export NVPRUNE="${CUDA_INSTALL_PREFIX}/cuda-11.8/bin/nvprune"
  export CUDA_LIB_DIR="${CUDA_INSTALL_PREFIX}/cuda-11.8/lib64"

  export GENCODE="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
  export GENCODE_CUDNN="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"

  if [[ -n "$OVERRIDE_GENCODE" ]]; then
    export GENCODE=$OVERRIDE_GENCODE
  fi

  # all CUDA libs except CuDNN and CuBLAS (cudnn and cublas need arch 3.7 included)
  ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis" |
    xargs -I {} bash -c \
      "echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

  # prune CuDNN and CuBLAS
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

  #####################################################################################
  # CUDA 11.8 prune visual tools
  #####################################################################################
  export CUDA_BASE="${CUDA_INSTALL_PREFIX}/cuda-11.8/"
  rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins 
}

function prune_124 {
  echo "Pruning CUDA 12.4"
  #####################################################################################
  # CUDA 12.4 prune static libs
  #####################################################################################
  export NVPRUNE="${CUDA_INSTALL_PREFIX}/cuda-12.4/bin/nvprune"
  export CUDA_LIB_DIR="${CUDA_INSTALL_PREFIX}/cuda-12.4/lib64"

  export GENCODE="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
  export GENCODE_CUDNN="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"

  if [[ -n "$OVERRIDE_GENCODE" ]]; then
    export GENCODE=$OVERRIDE_GENCODE
  fi
  if [[ -n "$OVERRIDE_GENCODE_CUDNN" ]]; then
    export GENCODE_CUDNN=$OVERRIDE_GENCODE_CUDNN
  fi

  # all CUDA libs except CuDNN and CuBLAS
  ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis" |
    xargs -I {} bash -c \
      "echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

  # prune CuDNN and CuBLAS
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

  #####################################################################################
  # CUDA 12.4 prune visual tools
  #####################################################################################
  export CUDA_BASE="${CUDA_INSTALL_PREFIX}/cuda-12.4/"
  rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins 
}

function prune_126 {
  echo "Pruning CUDA 12.6"
  #####################################################################################
  # CUDA 12.6 prune static libs
  #####################################################################################
  export NVPRUNE="${CUDA_INSTALL_PREFIX}/cuda-12.6/bin/nvprune"
  export CUDA_LIB_DIR="${CUDA_INSTALL_PREFIX}/cuda-12.6/lib64"

  export GENCODE="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
  export GENCODE_CUDNN="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"

  if [[ -n "$OVERRIDE_GENCODE" ]]; then
    export GENCODE=$OVERRIDE_GENCODE
  fi
  if [[ -n "$OVERRIDE_GENCODE_CUDNN" ]]; then
    export GENCODE_CUDNN=$OVERRIDE_GENCODE_CUDNN
  fi

  # all CUDA libs except CuDNN and CuBLAS
  ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis" |
    xargs -I {} bash -c \
      "echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

  # prune CuDNN and CuBLAS
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

  #####################################################################################
  # CUDA 12.6 prune visual tools
  #####################################################################################
  export CUDA_BASE="/usr/local/cuda-12.6/"
  rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins 
}

VALID_VERSIONS=("11.8" "12.4" "12.6" "12.8")

# Make it compatible with previous usage
while test $# -gt 0; do
  if [[ " ${VALID_VERSIONS[@]} " =~ " $1 " ]]; then
    CUDA_VERSION=$1
  else
    echo "bad argument $1"
    exit 1
  fi
  shift
done

# Check if the CUDA version is valid
if [[ ! " ${VALID_VERSIONS[@]} " =~ " ${CUDA_VERSION} " ]]; then
  echo "CUDA_VERSION must be 11.8, 12.4, 12.6, or 12.8"
  exit 1
fi
echo "Debug: Version check passed"

if [ ! -d "$CUDA_INSTALL_PREFIX" ]; then
  echo "The directory specified by CUDA_INSTALL_PREFIX does not exist: $CUDA_INSTALL_PREFIX"
  exit 1
fi

version_no_dot="${CUDA_VERSION//./}"
eval install_${version_no_dot}
if [ "$SKIP_PRUNE" -eq 0 ]; then
  eval prune_${version_no_dot}
fi

# clean up the temp directory
cleanup_temp_dirs
rm -rf "${USER_TMPDIR}"

echo "CUDA installation complete"
