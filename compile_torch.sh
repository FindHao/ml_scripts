#!/bin/bash
# simple usage: work_path=/home/yhao/pt ./compile_torch.sh
# make sure you have activated the correct conda environment before running this script

# Add strict mode for better error handling
set -euo pipefail
IFS=$'\n\t'

# Consolidate and organize environment variables at the top
declare -r MAX_JOBS=256
declare -r DEFAULT_WORK_PATH="/home/yhao/p9"

# Convert environment variables to more robust declarations
declare -r work_path=${work_path:-"$DEFAULT_WORK_PATH"}
declare -r clean_install=${clean_install:-0}
declare -r clean_upgrade=${clean_upgrade:-0}
declare -r clean_torch=${clean_torch:-0}
declare -r torch_only=${torch_only:-0}
declare -r debug=${debug:-0}
declare -r torch_commit=${torch_commit:-""}
declare -r torch_branch=${torch_branch:-"main"}
declare -r torch_pull=${torch_pull:-0}
declare -r no_torchbench=${no_torchbench:-0}

# GPU-related exports
export USE_ROCM=0
export USE_NCCL=1

# Improve error handling function
function error_exit() {
    local message="$1"
    echo "ERROR: $message" >&2
    exit 1
}

# Improve the git_upgrade_pack function with error handling
function git_upgrade_pack() {
    local package_name="$1"
    echo "Upgrading package: $package_name"
    cd "$work_path/$package_name" || error_exit "Failed to change directory to $package_name"
    git pull || error_exit "Failed to pull latest changes for $package_name"
    git submodule sync || error_exit "Failed to sync submodules for $package_name"
    git submodule update --init --recursive || error_exit "Failed to update submodules for $package_name"
}

# Improve the upgrade_pack function
function upgrade_pack() {
    local package_name="$1"
    echo "Installing package: $package_name"
    git_upgrade_pack "$package_name"
    pip uninstall -y "torch$package_name" || true # Don't fail if package isn't installed
    python setup.py clean || error_exit "Failed to clean $package_name"
    python setup.py install || error_exit "Failed to install $package_name"
    echo "$package_name installation completed successfully"
}

# print configs
echo "work_path: ${work_path}"
echo "clean_install: ${clean_install}"
echo "clean_torch: ${clean_torch}"
echo "torch_only: ${torch_only}"
echo "torch_branch: ${torch_branch}"
echo "torch_commit: ${torch_commit}"

# Extract CUDA version from nvcc --version with fallback
CUDA_VERSION=$(nvcc --version | grep "release" | sed -E 's/.*release ([0-9]+\.[0-9]+).*/\1/' | sed 's/\.//')
# Check if the version was detected correctly
if [[ -z "$CUDA_VERSION" ]] || ! [[ "$CUDA_VERSION" =~ ^[0-9]+$ ]]; then
    # Set default version if detection fails
    CUDA_VERSION="126"
    echo "WARNING: Could not detect CUDA version properly. Defaulting to CUDA 12.6 (CUDA_VERSION=${CUDA_VERSION})"
else
    echo "Detected CUDA version: ${CUDA_VERSION}"
fi
conda install -y magma-cuda${CUDA_VERSION} -c pytorch
conda install -y ccache cmake ninja mkl mkl-include libpng libjpeg-turbo -c conda-forge
# graphviz

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

# Improve directory handling
cd "$work_path" || error_exit "Failed to change to work directory"

# Improve clean install section with error handling
if [ "$clean_install" -eq 1 ]; then
    echo "Performing clean installation..."
    rm -rf pytorch text vision audio benchmark data
    for repo in pytorch text data vision audio benchmark; do
        git clone --recursive "git@github.com:pytorch/${repo}.git" || error_exit "Failed to clone $repo"
    done
fi

function notify_finish() {
    echo "PyTorch compilation completed successfully"
    if command -v notify &>/dev/null; then
        notify "PyTorch Compilation is done" || true # Don't fail if notify fails
    fi
}

pip uninstall -y torch
# install pytorch
cd $work_path/pytorch
git fetch
if [ -n "$torch_commit" ]; then
    git checkout $torch_commit
    echo "warnging: you are using a specific commit. don't forget to create a new branch if you want to make changes"
else
    git checkout $torch_branch
fi
if [ $torch_pull -eq 1 ] && [ -z "$torch_commit" ]; then
    git pull
fi
git submodule sync
git submodule update --init --recursive
pip install -r requirements.txt
make triton

if [ $clean_torch -eq 1 ]; then
    python setup.py clean
fi

if [ $debug -eq 1 ]; then
    debug_prefix="env DEBUG=1"
else
    debug_prefix=""
fi

${debug_prefix} python setup.py develop

if [ $torch_only -eq 1 ]; then
    notify_finish
    exit 0
fi

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

# Add trap for cleanup on script exit
trap 'echo "Script execution interrupted"; exit 1' INT TERM
