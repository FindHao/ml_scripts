#!/bin/bash
# simple usage: work_path=/home/yhao/pt ./compile_torch.sh
# make sure you have activated the correct conda environment before running this script

# Add strict mode for better error handling
set -euo pipefail
IFS=$'\n\t'

# Start timing the script execution
start_time=$(date +%s)

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

# Function to format time
function format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
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
    local package_start_time=$(date +%s)
    echo "Installing package: $package_name"
    git_upgrade_pack "$package_name"
    pip uninstall -y "torch$package_name" || true # Don't fail if package isn't installed
    python setup.py clean || error_exit "Failed to clean $package_name"
    python setup.py install || error_exit "Failed to install $package_name"
    local package_end_time=$(date +%s)
    local package_duration=$((package_end_time - package_start_time))
    echo "$package_name installation completed successfully in $(format_time $package_duration)"
}

# print configs
echo "work_path: ${work_path}"
echo "clean_install: ${clean_install}"
echo "clean_torch: ${clean_torch}"
echo "torch_only: ${torch_only}"
echo "torch_branch: ${torch_branch}"
echo "torch_commit: ${torch_commit}"

# Function to get CUDA version from nvcc
function get_cuda_version_from_nvcc() {
    if command -v nvcc &>/dev/null; then
        local nvcc_output
        nvcc_output=$(nvcc --version)
        if [[ "$nvcc_output" =~ release[[:space:]]([0-9]+\\.[0-9]+) ]]; then
            echo "${BASH_REMATCH[1]}"
        else
            echo "Could not parse CUDA version from nvcc output." >&2
            # Default to a common version or ask user, here defaulting to 11.8 as a fallback
            echo "12.8"
        fi
    else
        echo "nvcc not found. Please ensure CUDA toolkit is installed and nvcc is in PATH." >&2
        # Default to a common version or ask user, here defaulting to 11.8 as a fallback
        echo "12.8"
    fi
}

# Determine CUDA version
cuda_version=$(get_cuda_version_from_nvcc)
echo "Detected CUDA version: $cuda_version"

conda install -y ccache cmake==3.31.6 ninja -c conda-forge
conda install -y libpng libjpeg-turbo -c conda-forge
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# Download and install Magma for the detected CUDA version
echo "Downloading install_magma_conda.sh..."
wget https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_magma_conda.sh -O /tmp/install_magma_conda.sh || error_exit "Failed to download install_magma_conda.sh"
chmod +x /tmp/install_magma_conda.sh
echo "Running install_magma_conda.sh for CUDA $cuda_version..."
cd /tmp/
./install_magma_conda.sh "$cuda_version" || error_exit "install_magma_conda.sh failed"
echo "MAGMA installation/check completed."

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

if [ -f "$HOME/.notify.sh" ]; then
    source "$HOME/.notify.sh"
fi

function notify_finish() {
    end_time=$(date +%s)
    total_duration=$((end_time - start_time))
    formatted_duration=$(format_time $total_duration)
    echo "PyTorch compilation completed successfully in $formatted_duration"
    if command -v notify &>/dev/null; then
        notify "PyTorch Compilation finished in $formatted_duration" || true # Don't fail if notify fails
    fi
}

pip uninstall -y torch
# install pytorch
cd $work_path/pytorch
torch_start_time=$(date +%s)
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
torch_end_time=$(date +%s)
torch_duration=$((torch_end_time - torch_start_time))
echo "PyTorch core installation completed in $(format_time $torch_duration)"

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
torchbench_start_time=$(date +%s)
pip install pyyaml
cd $work_path/benchmark
git pull
git submodule sync
git submodule update --init --recursive
python install.py
torchbench_end_time=$(date +%s)
torchbench_duration=$((torchbench_end_time - torchbench_start_time))
echo "torchbench installation completed in $(format_time $torchbench_duration)"
notify_finish

# Add trap for cleanup on script exit
trap 'echo "Script execution interrupted"; exit 1' INT TERM
