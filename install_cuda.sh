#!/bin/bash
# Usage: CUDA_INSTALL_PREFIX=/home/yhao/opt ./install_cuda.sh 12.8
# Notice: Part of this script is synced with https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_cuda.sh
set -ex

# Make cuda install path configurable. By default, it is $HOME/opt.
CUDA_INSTALL_PREFIX=${CUDA_INSTALL_PREFIX:-$HOME/opt}
# Remove trailing slash if present
CUDA_INSTALL_PREFIX=${CUDA_INSTALL_PREFIX%/}
SKIP_PRUNE=${SKIP_PRUNE:-1}
CUDA_VERSION=${CUDA_VERSION:-12.8}

# Set a user specific tmp directory to avoid permissions issues
USER_TMPDIR="/tmp/${USER}/cuda_install"
mkdir -p "${USER_TMPDIR}"
export TMPDIR="${USER_TMPDIR}"

# Determine target architecture only once
export TARGETARCH=${TARGETARCH:-$(uname -m)}
if [ "${TARGETARCH}" = 'aarch64' ] || [ "${TARGETARCH}" = 'arm64' ]; then
  ARCH_PATH='sbsa'
else
  ARCH_PATH='x86_64'
fi

# Clean up any leftover temporary directories from previous failed installations
cleanup_temp_dirs() {
  echo "Cleaning up temporary directories..."
  [ -d "tmp_cusparselt" ] && rm -rf tmp_cusparselt
  [ -d "tmp_cudnn" ] && rm -rf tmp_cudnn
  [ -d "nccl" ] && rm -rf nccl
}

# Error handling function
error_exit() {
  echo "ERROR: $1" >&2
  cleanup_temp_dirs
  rm -rf "${USER_TMPDIR}"
  exit 1
}

# Check if a command exists
command_exists() {
  command -v "$1" &> /dev/null
}

# Check required dependencies
check_dependencies() {
  echo "Checking dependencies..."
  for cmd in wget curl git make; do
    if ! command_exists $cmd; then
      error_exit "$cmd is required but not installed. Please install it and try again."
    fi
  done
}

# Install CUDA toolkit
function install_cuda {
  version=$1
  runfile=$2
  major_minor=${version%.*}
  
  echo "Installing CUDA ${version}..."
  rm -rf ${CUDA_INSTALL_PREFIX}/cuda-${major_minor} ${CUDA_INSTALL_PREFIX}/cuda
  
  if [[ ${ARCH_PATH} == 'sbsa' ]]; then
    runfile="${runfile}_sbsa"
  fi
  runfile="${runfile}.run"
  
  if ! wget -q https://developer.download.nvidia.com/compute/cuda/${version}/local_installers/${runfile} -O ${runfile}; then
    error_exit "Failed to download CUDA installer"
  fi
  
  chmod +x ${runfile}
  if ! ./${runfile} --toolkit --silent --toolkitpath=${CUDA_INSTALL_PREFIX}/cuda-${major_minor}; then
    echo "Failed to install CUDA toolkit. The installer file ${runfile} is preserved for troubleshooting."
    error_exit "CUDA installation failed"
  fi
  
  rm -f ${runfile}
  rm -f ${CUDA_INSTALL_PREFIX}/cuda && ln -s ${CUDA_INSTALL_PREFIX}/cuda-${major_minor} ${CUDA_INSTALL_PREFIX}/cuda
}

# Install cuDNN
function install_cudnn {
  cuda_major_version=$1
  cudnn_version=$2
  
  echo "Installing cuDNN ${cudnn_version} for CUDA ${cuda_major_version}..."
  mkdir tmp_cudnn && cd tmp_cudnn || error_exit "Failed to create temporary directory for cuDNN"
  
  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  filepath="cudnn-linux-${ARCH_PATH}-${cudnn_version}_cuda${cuda_major_version}-archive"
  if ! wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-${ARCH_PATH}/${filepath}.tar.xz; then
    cd ..
    rm -rf tmp_cudnn
    error_exit "Failed to download cuDNN"
  fi
  
  if ! tar xf ${filepath}.tar.xz; then
    cd ..
    rm -rf tmp_cudnn
    error_exit "Failed to extract cuDNN"
  fi
  
  cp -a ${filepath}/include/* ${CUDA_INSTALL_PREFIX}/cuda/include/
  cp -a ${filepath}/lib/* ${CUDA_INSTALL_PREFIX}/cuda/lib64/
  cd ..
  rm -rf tmp_cudnn
}

# Install cuSparseLt
function install_cusparselt {
  echo "Installing cuSparseLt for CUDA ${CUDA_VERSION}..."
  # cuSparseLt license: https://docs.nvidia.com/cuda/cusparselt/license.html
  mkdir tmp_cusparselt && pushd tmp_cusparselt || error_exit "Failed to create temporary directory for cuSparseLt"

  local cusparselt_version
  if [[ ${CUDA_VERSION:0:4} =~ ^12\.[5-8]$ ]]; then
    cusparselt_version="0.6.3.2"
  elif [[ ${CUDA_VERSION:0:4} == "12.4" ]]; then
    cusparselt_version="0.6.2.3"
  elif [[ ${CUDA_VERSION:0:4} == "11.8" ]]; then
    cusparselt_version="0.4.0.7"
    # Override ARCH_PATH for 11.8 as it only supports x86_64
    ARCH_PATH="x86_64"
  else
    popd
    rm -rf tmp_cusparselt
    error_exit "Unsupported CUDA version ${CUDA_VERSION} for cuSparseLt"
  fi

  CUSPARSELT_NAME="libcusparse_lt-linux-${ARCH_PATH}-${cusparselt_version}-archive"
  if ! curl --retry 3 -OLs https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-${ARCH_PATH}/${CUSPARSELT_NAME}.tar.xz; then
    popd
    rm -rf tmp_cusparselt
    error_exit "Failed to download cuSparseLt"
  fi

  if ! tar xf ${CUSPARSELT_NAME}.tar.xz; then
    popd
    rm -rf tmp_cusparselt
    error_exit "Failed to extract cuSparseLt"
  fi
  
  cp -a ${CUSPARSELT_NAME}/include/* ${CUDA_INSTALL_PREFIX}/cuda/include/
  cp -a ${CUSPARSELT_NAME}/lib/* ${CUDA_INSTALL_PREFIX}/cuda/lib64/
  popd
  rm -rf tmp_cusparselt
}

# Install NCCL
function install_nccl {
  echo "Installing NCCL for CUDA ${CUDA_VERSION}..."
  NCCL_VERSION=""
  if [[ ${CUDA_VERSION:0:2} == "11" ]]; then
    NCCL_VERSION=$(curl -sL https://github.com/pytorch/pytorch/raw/refs/heads/main/.ci/docker/ci_commit_pins/nccl-cu11.txt)
  elif [[ ${CUDA_VERSION:0:2} == "12" ]]; then
    NCCL_VERSION=$(curl -sL https://github.com/pytorch/pytorch/raw/refs/heads/main/.ci/docker/ci_commit_pins/nccl-cu12.txt)
  else
    error_exit "Unexpected CUDA_VERSION ${CUDA_VERSION}"
  fi

  if [[ -z "${NCCL_VERSION}" ]]; then
    error_exit "NCCL_VERSION is empty"
  fi

  # NCCL license: https://docs.nvidia.com/deeplearning/nccl/#licenses
  # Follow build: https://github.com/NVIDIA/nccl/tree/master?tab=readme-ov-file#build
  if ! git clone -b $NCCL_VERSION --depth 1 https://github.com/NVIDIA/nccl.git; then
    error_exit "Failed to clone NCCL repository"
  fi
  
  pushd nccl || error_exit "Failed to enter NCCL directory"
  if ! make -j src.build CUDA_HOME=${CUDA_INSTALL_PREFIX}/cuda; then
    popd
    rm -rf nccl
    error_exit "Failed to build NCCL"
  fi
  
  cp -a build/include/* ${CUDA_INSTALL_PREFIX}/cuda/include/
  cp -a build/lib/* ${CUDA_INSTALL_PREFIX}/cuda/lib64/
  popd
  rm -rf nccl
  
  if [ "$(id -u)" -eq 0 ]; then
    ldconfig
  fi
}

# Install CUDA 11.8
function install_118 {
  local CUDNN_VERSION=9.1.0.70
  echo "Beginning installation for CUDA 11.8"
  install_cuda "11.8.0" "cuda_11.8.0_520.61.05_linux"
  install_cudnn "11" "${CUDNN_VERSION}"
  install_nccl
  install_cusparselt

  if [ "$(id -u)" -eq 0 ]; then
    ldconfig
  fi
}

# Install CUDA 12.4
function install_124 {
  local CUDNN_VERSION=9.1.0.70
  echo "Beginning installation for CUDA 12.4"
  install_cuda "12.4.1" "cuda_12.4.1_550.54.15_linux"
  install_cudnn "12" "${CUDNN_VERSION}"
  install_nccl
  install_cusparselt

  if [ "$(id -u)" -eq 0 ]; then
    ldconfig
  fi
}

# Install CUDA 12.6
function install_126 {
  local CUDNN_VERSION=9.5.1.17
  echo "Beginning installation for CUDA 12.6"
  install_cuda "12.6.3" "cuda_12.6.3_560.35.05_linux"
  install_cudnn "12" "${CUDNN_VERSION}"
  install_nccl
  install_cusparselt

  if [ "$(id -u)" -eq 0 ]; then
    ldconfig
  fi
}

# Install CUDA 12.8
function install_128 {
  local CUDNN_VERSION=9.8.0.87
  echo "Beginning installation for CUDA 12.8"
  install_cuda "12.8.0" "cuda_12.8.0_570.86.10_linux"
  install_cudnn "12" "${CUDNN_VERSION}"
  install_nccl
  install_cusparselt

  if [ "$(id -u)" -eq 0 ]; then
    ldconfig
  fi
}

# Create a unified prune function
function prune_cuda {
  local cuda_version=$1
  local major_minor=$2
  
  echo "Pruning CUDA ${major_minor}"
  
  # Set appropriate GENCODE flags based on CUDA version
  if [[ ${major_minor} == "11.8" ]]; then
    GENCODE="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
    GENCODE_CUDNN="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
  else
    GENCODE="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
    GENCODE_CUDNN="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
  fi
  
  # Override GENCODE if specified
  if [[ -n "$OVERRIDE_GENCODE" ]]; then
    GENCODE=$OVERRIDE_GENCODE
  fi
  
  # Override GENCODE_CUDNN if specified (only for CUDA 12.x)
  if [[ -n "$OVERRIDE_GENCODE_CUDNN" && ${major_minor:0:2} == "12" ]]; then
    GENCODE_CUDNN=$OVERRIDE_GENCODE_CUDNN
  fi
  
  # Set up paths
  export NVPRUNE="${CUDA_INSTALL_PREFIX}/cuda-${major_minor}/bin/nvprune"
  export CUDA_LIB_DIR="${CUDA_INSTALL_PREFIX}/cuda-${major_minor}/lib64"
  export CUDA_BASE="${CUDA_INSTALL_PREFIX}/cuda-${major_minor}/"
  
  # Prune static libs
  ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis" |
    xargs -I {} bash -c \
      "echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"
  
  # Prune CuDNN and CuBLAS
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a
  
  # Prune visual tools
  rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins
}

# Map version-specific prune functions to unified function
function prune_118 {
  prune_cuda "118" "11.8"
}

function prune_124 {
  prune_cuda "124" "12.4"
}

function prune_126 {
  prune_cuda "126" "12.6"
}

# Main execution block
{
  # Valid CUDA versions
  VALID_VERSIONS=("11.8" "12.4" "12.6" "12.8")

  # Check dependencies
  check_dependencies

  # Clean up any leftovers from previous runs
  cleanup_temp_dirs

  # Parse arguments
  while test $# -gt 0; do
    if [[ " ${VALID_VERSIONS[@]} " =~ " $1 " ]]; then
      CUDA_VERSION=$1
    else
      error_exit "Bad argument: $1. CUDA_VERSION must be one of: ${VALID_VERSIONS[*]}"
    fi
    shift
  done

  # Validate CUDA version
  if [[ ! " ${VALID_VERSIONS[@]} " =~ " ${CUDA_VERSION} " ]]; then
    error_exit "CUDA_VERSION must be one of: ${VALID_VERSIONS[*]}"
  fi

  # Validate installation directory
  if [ ! -d "$CUDA_INSTALL_PREFIX" ]; then
    error_exit "The directory specified by CUDA_INSTALL_PREFIX does not exist: $CUDA_INSTALL_PREFIX"
  fi

  # Run installation
  version_no_dot="${CUDA_VERSION//./}"
  eval install_${version_no_dot}
  
  # Run pruning if requested
  if [ "$SKIP_PRUNE" -eq 0 ]; then
    eval prune_${version_no_dot}
  fi

  # Final cleanup
  cleanup_temp_dirs
  rm -rf "${USER_TMPDIR}"

  echo "CUDA ${CUDA_VERSION} installation completed successfully"
}
