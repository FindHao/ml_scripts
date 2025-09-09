#!/bin/bash
# Usage: CUDA_INSTALL_PREFIX=/home/yhao/opt ./install_cuda.sh 12.8
# Notice: Part of this script is synced with https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_cuda.sh
set -ex

# Debug settings - log all execution steps
exec 5>/tmp/cuda_install_debug.txt
BASH_XTRACEFD="5"
PS4='${LINENO}: '

echo "🚀 ===== CUDA Installation Script Started ====="

# Basic settings
CUDA_INSTALL_PREFIX=${CUDA_INSTALL_PREFIX:-$HOME/opt}
CUDA_INSTALL_PREFIX=${CUDA_INSTALL_PREFIX%/}
CUDA_VERSION=${CUDA_VERSION:-12.8}
NVSHMEM_VERSION=${NVSHMEM_VERSION:-3.3.24}
INSTALL_NCCL=${INSTALL_NCCL:-1}

echo "CUDA_INSTALL_PREFIX=${CUDA_INSTALL_PREFIX}"
echo "CUDA_VERSION=${CUDA_VERSION}"
echo "INSTALL_NCCL=${INSTALL_NCCL}"

# Create temporary directory
USER_TMPDIR="/tmp/${USER}/cuda_install"
mkdir -p "${USER_TMPDIR}"
export TMPDIR="${USER_TMPDIR}"

# Error handling function
function error_exit {
  echo "❌ ERROR: $1" >&2
  # Save error message to log file
  echo "$(date): ERROR - $1" >>"${USER_TMPDIR}/error.log"
  # Leave error marker file for debugging
  echo "$1" >"${USER_TMPDIR}/error_message"
  touch "${USER_TMPDIR}/error_occurred"
  exit 1
}

# Cleanup function
function cleanup_temp_dirs {
  echo "Cleaning up temporary directories..."
  [ -d "tmp_cusparselt" ] && rm -rf tmp_cusparselt
  [ -d "tmp_cudnn" ] && rm -rf tmp_cudnn
  [ -d "nccl" ] && rm -rf nccl
  touch "${USER_TMPDIR}/cleanup_completed"
}

# Detect architecture
export TARGETARCH=${TARGETARCH:-$(uname -m)}
if [ "${TARGETARCH}" = 'aarch64' ] || [ "${TARGETARCH}" = 'arm64' ]; then
  ARCH_PATH='sbsa'
else
  ARCH_PATH='x86_64'
fi

echo "🏗️  Architecture: ${ARCH_PATH}"

# Check if command exists
function command_exists {
  command -v "$1" &>/dev/null
}

# Check dependencies
function check_dependencies {
  echo "Checking dependencies..."
  local missing_deps=()

  for cmd in wget curl git make; do
    if ! command_exists $cmd; then
      missing_deps+=($cmd)
    fi
  done

  if [ ${#missing_deps[@]} -gt 0 ]; then
    error_exit "Missing required dependencies: ${missing_deps[*]}"
  fi

  echo "✅ All dependency checks passed"
}

# Check if file or directory exists
function check_exists {
  if [ ! -e "$1" ]; then
    error_exit "Path does not exist: $1"
  fi
}

# Check directory
check_exists "${CUDA_INSTALL_PREFIX}"

# Check dependencies
check_dependencies

# Real CUDA installation function
function install_cuda {
  local version=$1
  local runfile=$2
  local major_minor=${version%.*}

  echo "Installing CUDA ${version}..."
  rm -rf ${CUDA_INSTALL_PREFIX}/cuda-${major_minor} ${CUDA_INSTALL_PREFIX}/cuda

  if [[ ${ARCH_PATH} == 'sbsa' ]]; then
    runfile="${runfile}_sbsa"
  fi
  runfile="${runfile}.run"

  echo "Downloading CUDA installation file: ${runfile}"
  # Use --progress=dot:giga parameter to show download progress
  # Use -t 3 option to set retry count to 3 times
  if ! wget --progress=dot:giga -t 3 -q https://developer.download.nvidia.com/compute/cuda/${version}/local_installers/${runfile} -O ${runfile}; then
    error_exit "CUDA installation file download failed: ${runfile}"
  fi

  echo "CUDA installation file download complete, preparing to install..."
  chmod +x ${runfile}

  # Create installation progress marker
  touch "${USER_TMPDIR}/cuda_${version}_download_complete"

  echo "Executing CUDA installation script..."
  if ! ./${runfile} --toolkit --silent --toolkitpath=${CUDA_INSTALL_PREFIX}/cuda-${major_minor}; then
    echo "CUDA installation failed, keeping installation file for troubleshooting"
    error_exit "CUDA ${version} installation failed"
  fi

  # Create installation complete marker
  touch "${USER_TMPDIR}/cuda_${version}_install_complete"

  echo "CUDA installation complete, creating symbolic link..."
  rm -f ${CUDA_INSTALL_PREFIX}/cuda && ln -s ${CUDA_INSTALL_PREFIX}/cuda-${major_minor} ${CUDA_INSTALL_PREFIX}/cuda

  echo "Removing installation file..."
  rm -f ${runfile}

  echo "CUDA ${version} installation completed"
  return 0
}

# Real cuDNN installation function
function install_cudnn {
  local cuda_major_version=$1
  local cudnn_version=$2

  echo "Installing cuDNN ${cudnn_version} for CUDA ${cuda_major_version}..."

  # Create temporary directory
  mkdir -p tmp_cudnn
  cd tmp_cudnn || error_exit "Failed to create cuDNN temporary directory"

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  local filepath="cudnn-linux-${ARCH_PATH}-${cudnn_version}_cuda${cuda_major_version}-archive"
  echo "Downloading cuDNN: ${filepath}.tar.xz"

  if ! wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-${ARCH_PATH}/${filepath}.tar.xz; then
    cd ..
    rm -rf tmp_cudnn
    error_exit "cuDNN download failed: ${filepath}.tar.xz"
  fi

  echo "cuDNN download complete, preparing to extract..."
  if ! tar xf ${filepath}.tar.xz; then
    cd ..
    rm -rf tmp_cudnn
    error_exit "cuDNN extraction failed: ${filepath}.tar.xz"
  fi

  echo "cuDNN extraction complete, installing to CUDA directory..."
  cp -a ${filepath}/include/* ${CUDA_INSTALL_PREFIX}/cuda/include/
  cp -a ${filepath}/lib/* ${CUDA_INSTALL_PREFIX}/cuda/lib64/

  echo "cuDNN installation complete, cleaning up temporary files..."
  cd ..
  rm -rf tmp_cudnn

  # Create cuDNN installation complete marker
  touch "${USER_TMPDIR}/cudnn_${cudnn_version}_installed"

  echo "cuDNN ${cudnn_version} installation completed"
  return 0
}

# Real NCCL installation function
function install_nccl {
  if [ "$INSTALL_NCCL" -eq 0 ]; then
    echo "NCCL installation skipped as per INSTALL_NCCL=${INSTALL_NCCL}"
    return 0
  fi
  
  echo "Installing NCCL for CUDA ${CUDA_VERSION}..."
  local NCCL_VERSION=""

  echo "Getting NCCL version information..."
  if [[ ${CUDA_VERSION:0:2} == "12" ]]; then
    NCCL_VERSION=$(curl -sL https://github.com/pytorch/pytorch/raw/refs/heads/main/.ci/docker/ci_commit_pins/nccl-cu12.txt)
  elif [[ ${CUDA_VERSION:0:2} == "13" ]]; then
    NCCL_VERSION=$(curl -sL https://github.com/pytorch/pytorch/raw/refs/heads/main/.ci/docker/ci_commit_pins/nccl-cu13.txt)
  else
    error_exit "Unsupported CUDA version: ${CUDA_VERSION}"
  fi

  if [[ -z "${NCCL_VERSION}" ]]; then
    error_exit "Failed to get NCCL version information"
  fi

  echo "Retrieved NCCL version: ${NCCL_VERSION}"
  echo "${NCCL_VERSION}" >"${USER_TMPDIR}/nccl_version.txt"

  # NCCL license: https://docs.nvidia.com/deeplearning/nccl/#licenses
  # Follow build: https://github.com/NVIDIA/nccl/tree/master?tab=readme-ov-file#build
  echo "Cloning NCCL repository..."
  if ! git clone -b $NCCL_VERSION --depth 1 https://github.com/NVIDIA/nccl.git; then
    error_exit "NCCL repository clone failed"
  fi

  echo "Starting NCCL compilation..."
  pushd nccl || error_exit "Failed to enter NCCL directory"

  if ! make -j src.build CUDA_HOME=${CUDA_INSTALL_PREFIX}/cuda; then
    popd
    rm -rf nccl
    error_exit "NCCL compilation failed"
  fi

  echo "NCCL compilation complete, installing to CUDA directory..."
  cp -a build/include/* ${CUDA_INSTALL_PREFIX}/cuda/include/
  cp -a build/lib/* ${CUDA_INSTALL_PREFIX}/cuda/lib64/

  popd
  rm -rf nccl

  if [ "$(id -u)" -eq 0 ]; then
    ldconfig
  fi

  # Create NCCL installation complete marker
  touch "${USER_TMPDIR}/nccl_installed"

  echo "NCCL installation completed"
  return 0
}

# Real cuSparseLt installation function
function install_cusparselt {
  echo "Installing cuSparseLt for CUDA ${CUDA_VERSION}..."
  mkdir -p tmp_cusparselt
  pushd tmp_cusparselt || error_exit "Failed to create cuSparseLt temporary directory"

  local cusparselt_version
  local arch_path=${ARCH_PATH}

  local CUSPARSELT_NAME
  if [[ ${CUDA_VERSION:0:2} == "13" ]]; then
    cusparselt_version="0.8.0.4"
    CUSPARSELT_NAME="libcusparse_lt-linux-${arch_path}-${cusparselt_version}_cuda13-archive"
  elif [[ ${CUDA_VERSION:0:4} =~ ^12\.[5-9]$ ]]; then
    cusparselt_version="0.7.1.0"
    CUSPARSELT_NAME="libcusparse_lt-linux-${arch_path}-${cusparselt_version}-archive"
  else
    popd
    rm -rf tmp_cusparselt
    error_exit "Unsupported CUDA version: ${CUDA_VERSION}"
  fi

  echo "${cusparselt_version}" >"${USER_TMPDIR}/cusparselt_version.txt"

  echo "Downloading cuSparseLt: ${CUSPARSELT_NAME}.tar.xz"

  if ! curl --retry 3 -OLs https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-${arch_path}/${CUSPARSELT_NAME}.tar.xz; then
    popd
    rm -rf tmp_cusparselt
    error_exit "cuSparseLt download failed"
  fi

  echo "cuSparseLt download complete, preparing to extract..."
  if ! tar xf ${CUSPARSELT_NAME}.tar.xz; then
    popd
    rm -rf tmp_cusparselt
    error_exit "cuSparseLt extraction failed"
  fi

  echo "cuSparseLt extraction complete, installing to CUDA directory..."
  cp -a ${CUSPARSELT_NAME}/include/* ${CUDA_INSTALL_PREFIX}/cuda/include/
  cp -a ${CUSPARSELT_NAME}/lib/* ${CUDA_INSTALL_PREFIX}/cuda/lib64/

  popd
  rm -rf tmp_cusparselt

  touch "${USER_TMPDIR}/cusparselt_installed"

  echo "cuSparseLt installation completed"
  return 0
}

# nvSHMEM installation function
function install_nvshmem {
  local cuda_major_version=$1 # e.g. "12"
  local nvshmem_version=${NVSHMEM_VERSION}
  local arch_path=${ARCH_PATH}

  case "${arch_path}" in
    sbsa)
      dl_arch="aarch64"
      ;;
    x86_64)
      dl_arch="x64"
      ;;
    *)
      dl_arch="${arch_path}"
      ;;
  esac

  local tmpdir="tmp_nvshmem"
  mkdir -p "${tmpdir}" && cd "${tmpdir}"

  # nvSHMEM license: https://docs.nvidia.com/nvshmem/api/sla.html
  local filename="libnvshmem-linux-${arch_path}-${nvshmem_version}_cuda${cuda_major_version}-archive"
  local suffix=".tar.xz"
  local url="https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-${arch_path}/${filename}${suffix}"

  echo "Downloading nvSHMEM: ${filename}${suffix}"
  if ! wget -q "${url}"; then
    cd ..
    rm -rf "${tmpdir}"
    error_exit "nvSHMEM download failed: ${filename}${suffix}"
  fi

  echo "Extracting nvSHMEM..."
  if ! tar xf "${filename}${suffix}"; then
    cd ..
    rm -rf "${tmpdir}"
    error_exit "nvSHMEM extraction failed: ${filename}${suffix}"
  fi

  echo "Installing nvSHMEM to CUDA directory..."
  cp -a "${filename}/include/"* ${CUDA_INSTALL_PREFIX}/cuda/include/
  cp -a "${filename}/lib/"*     ${CUDA_INSTALL_PREFIX}/cuda/lib64/

  cd ..
  rm -rf "${tmpdir}"

  # Create nvshmem installation complete marker
  touch "${USER_TMPDIR}/nvshmem_${nvshmem_version}_installed"

  echo "nvSHMEM ${nvshmem_version} for CUDA ${cuda_major_version} (${arch_path}) installed."
  return 0
}

# CUDA 12.6 installation function
function install_126 {
  local CUDNN_VERSION=9.10.2.21
  echo "Starting installation for CUDA 12.6..."

  echo "📦 STEP 1: Installing CUDA toolkit..."
  install_cuda "12.6.3" "cuda_12.6.3_560.35.05_linux" || error_exit "CUDA 12.6.3 toolkit installation failed"

  echo "🧠 STEP 2: Installing cuDNN..."
  install_cudnn "12" "${CUDNN_VERSION}" || error_exit "cuDNN installation failed"

  echo "🔗 STEP 3: Installing NCCL..."
  install_nccl || error_exit "NCCL installation failed"

  echo "⚡ STEP 4: Installing cuSparseLt..."
  install_cusparselt || error_exit "cuSparseLt installation failed"

  echo "💾 STEP 5: Installing nvSHMEM..."
  install_nvshmem "12" || error_exit "nvSHMEM installation failed"

  if [ "$(id -u)" -eq 0 ]; then
    ldconfig
  fi

  echo "✅ CUDA 12.6 installation completed"
  return 0
}

# CUDA 12.8 installation function
function install_128 {
  local CUDNN_VERSION=9.8.0.87
  echo "Starting installation for CUDA 12.8..."

  echo "📦 STEP 1: Installing CUDA toolkit..."
  install_cuda "12.8.1" "cuda_12.8.1_570.124.06_linux" || error_exit "CUDA 12.8.1 toolkit installation failed"

  echo "🧠 STEP 2: Installing cuDNN..."
  install_cudnn "12" "${CUDNN_VERSION}" || error_exit "cuDNN installation failed"

  echo "🔗 STEP 3: Installing NCCL..."
  install_nccl || error_exit "NCCL installation failed"

  echo "⚡ STEP 4: Installing cuSparseLt..."
  install_cusparselt || error_exit "cuSparseLt installation failed"

  echo "💾 STEP 5: Installing nvSHMEM..."
  install_nvshmem "12" || error_exit "nvSHMEM installation failed"

  if [ "$(id -u)" -eq 0 ]; then
    ldconfig
  fi

  echo "✅ CUDA 12.8 installation completed"
  return 0
}

# Add CUDA 12.9 installation function
function install_129 {
  local CUDNN_VERSION=9.10.2.21
  echo "Starting installation for CUDA 12.9..."

  echo "📦 STEP 1: Installing CUDA toolkit..."
  install_cuda "12.9.1" "cuda_12.9.1_575.57.08_linux" || error_exit "CUDA 12.9.1 toolkit installation failed"

  echo "🧠 STEP 2: Installing cuDNN..."
  install_cudnn "12" "${CUDNN_VERSION}" || error_exit "cuDNN installation failed"

  echo "🔗 STEP 3: Installing NCCL..."
  install_nccl || error_exit "NCCL installation failed"

  echo "⚡ STEP 4: Installing cuSparseLt..."
  install_cusparselt || error_exit "cuSparseLt installation failed"

  echo "💾 STEP 5: Installing nvSHMEM..."
  install_nvshmem "12" || error_exit "nvSHMEM installation failed"

  if [ "$(id -u)" -eq 0 ]; then
    ldconfig
  fi

  echo "✅ CUDA 12.9 installation completed"
  return 0
}

# CUDA 13.0 installation function
function install_130 {
  local CUDNN_VERSION=9.13.0.50
  echo "Starting installation for CUDA 13.0..."

  echo "📦 STEP 1: Installing CUDA toolkit..."
  install_cuda "13.0.0" "cuda_13.0.0_580.65.06_linux" || error_exit "CUDA 13.0.0 toolkit installation failed"

  echo "🧠 STEP 2: Installing cuDNN..."
  install_cudnn "13" "${CUDNN_VERSION}" || error_exit "cuDNN installation failed"

  echo "🔗 STEP 3: Installing NCCL..."
  install_nccl || error_exit "NCCL installation failed"

  echo "⚡ STEP 4: Installing cuSparseLt..."
  install_cusparselt || error_exit "cuSparseLt installation failed"

  echo "💾 STEP 5: Installing nvSHMEM..."
  install_nvshmem "13" || error_exit "nvSHMEM installation failed"

  if [ "$(id -u)" -eq 0 ]; then
    ldconfig
  fi

  echo "✅ CUDA 13.0 installation completed"
  return 0
}

 

# Main execution logic
echo "🔧 ===== Parsing command line arguments ====="
VALID_VERSIONS=("12.6" "12.8" "12.9" "13.0")

# Parse command line arguments
while test $# -gt 0; do
  echo "Processing argument: $1"
  if [[ " ${VALID_VERSIONS[@]} " =~ " $1 " ]]; then
    CUDA_VERSION=$1
    echo "Setting CUDA version to: $CUDA_VERSION"
  else
    error_exit "Invalid argument: $1, CUDA version must be one of: ${VALID_VERSIONS[*]}"
  fi
  shift
done

# Validate CUDA version
echo "Validating CUDA version: $CUDA_VERSION"
if [[ ! " ${VALID_VERSIONS[@]} " =~ " ${CUDA_VERSION} " ]]; then
  error_exit "CUDA version must be one of: ${VALID_VERSIONS[*]}"
fi

# Perform cleanup before installation
cleanup_temp_dirs

# Perform installation
echo "⚙️  ===== Starting installation of CUDA ${CUDA_VERSION} ====="
version_no_dot="${CUDA_VERSION//./}"
echo "Calling install_${version_no_dot} function"

# Use set +e to temporarily disable error exit, for capturing errors and logging
set +e
eval "install_${version_no_dot}"
INSTALL_RESULT=$?
set -e

if [ $INSTALL_RESULT -ne 0 ]; then
  error_exit "Installation failed, exit code: $INSTALL_RESULT"
fi

 

# Final cleanup
cleanup_temp_dirs

# Version verification
echo "🔍 ===== Verifying CUDA installation ====="
if [ -f "${CUDA_INSTALL_PREFIX}/cuda/bin/nvcc" ]; then
  echo "✅ nvcc found - checking version:"
  echo "📋 $( "${CUDA_INSTALL_PREFIX}/cuda/bin/nvcc" --version | head -1 )"
  echo ""
else
  echo "⚠️  Warning: nvcc not found at ${CUDA_INSTALL_PREFIX}/cuda/bin/nvcc"
fi

if [ -f "${CUDA_INSTALL_PREFIX}/cuda/bin/nvidia-smi" ]; then
  echo "✅ nvidia-smi found - checking version:"
  echo "📋 $( "${CUDA_INSTALL_PREFIX}/cuda/bin/nvidia-smi" --version | head -1 )"
  echo ""
else
  echo "ℹ️  Note: nvidia-smi not found (this is normal for toolkit-only installation)"
fi

echo "🎉 ===== Script execution completed ====="
touch "${USER_TMPDIR}/script_completed_successfully"
echo "✅ CUDA ${CUDA_VERSION} has been successfully installed to ${CUDA_INSTALL_PREFIX}"
echo "📊 ========================================="
echo " 📋 CUDA & Related Libraries Installation Summary"
echo "📊 ========================================="
echo "  CUDA        : ${CUDA_VERSION}"
echo "  cuDNN       : ${CUDNN_VERSION:-(see install function)}"
if [ "$INSTALL_NCCL" -eq 1 ]; then
  if [ -f "${USER_TMPDIR}/nccl_version.txt" ]; then
    NCCL_VERSION_PRINT=$(cat "${USER_TMPDIR}/nccl_version.txt")
    echo "  NCCL        : ${NCCL_VERSION_PRINT}"
  else
    echo "  NCCL        : (not found)"
  fi
else
  echo "  NCCL        : (skipped)"
fi
if [ -f "${USER_TMPDIR}/cusparselt_version.txt" ]; then
  CUSPARSELT_VERSION_PRINT=$(cat "${USER_TMPDIR}/cusparselt_version.txt")
  echo "  cuSparseLt  : ${CUSPARSELT_VERSION_PRINT}"
else
  echo "  cuSparseLt  : (not found)"
fi
echo "  nvSHMEM     : ${NVSHMEM_VERSION}"
echo "🔗 -----------------------------------------"
echo "📁  Install Path : ${CUDA_INSTALL_PREFIX}/cuda"
echo "💡  Usage:"
echo "    export PATH=${CUDA_INSTALL_PREFIX}/cuda/bin:\$PATH"
echo "    export LD_LIBRARY_PATH=${CUDA_INSTALL_PREFIX}/cuda/lib64:\$LD_LIBRARY_PATH"
echo "📊 ========================================="
