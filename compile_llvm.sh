#!/bin/bash

# Script to clone LLVM and compile a specific commit version
# Usage: ./compile_llvm.sh
# Environment variables:
#   COMMIT_HASH - LLVM commit hash to build (default: 2619c2ed584cdf3b38e6743ed3c785223f06e3f7)
#   BUILD_DIR - Directory to build LLVM in (default: $HOME/g/opt/llvm-build)
#   LLVM_DIR - Directory of LLVM repository (default: $HOME/g/llvm-project)
# install dependencies
# conda install -c conda-forge git cmake ninja compilers make

set -e

# Start timing
START_TIME=$(date +%s)

# Default values
COMMIT_HASH=${COMMIT_HASH:-"2619c2ed584cdf3b38e6743ed3c785223f06e3f7"}
BUILD_DIR=${BUILD_DIR:-"$HOME/g/opt/llvm-build"}
LLVM_DIR=${LLVM_DIR:-"$HOME/g/llvm-project"}

echo "=== Compiling LLVM commit: $COMMIT_HASH ==="
echo "=== Build directory: $BUILD_DIR ==="

# Check if LLVM repository exists, clone if not
if [ ! -d "$LLVM_DIR" ]; then
    echo "=== Cloning LLVM repository ==="
    git clone https://github.com/llvm/llvm-project.git "$LLVM_DIR"
else
    echo "=== LLVM repository already exists at $LLVM_DIR ==="
fi

# Navigate to LLVM repository
cd "$LLVM_DIR"

# Fetch latest changes
echo "=== Fetching latest changes ==="
git fetch

# Checkout specified commit
echo "=== Checking out commit: $COMMIT_HASH ==="
git checkout $COMMIT_HASH
git submodule sync
git submodule update --init --recursive


if [ -d "$BUILD_DIR" ]; then
    echo "=== Removing existing build directory ==="
    rm -rf "$BUILD_DIR"
fi
echo "=== Creating build directory ==="
mkdir -p "$BUILD_DIR"

# Navigate to build directory
cd "$BUILD_DIR"

# Configure with CMake
echo "=== Configuring with CMake ==="
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
    "$LLVM_DIR/llvm"

# Build
echo "=== Building LLVM ==="
ninja

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

# Format the time string
if [ $HOURS -gt 0 ]; then
    TIME_STRING="${HOURS}h ${MINUTES}m ${SECONDS}s"
elif [ $MINUTES -gt 0 ]; then
    TIME_STRING="${MINUTES}m ${SECONDS}s"
else
    TIME_STRING="${SECONDS}s"
fi

echo "=== Build completed successfully in $TIME_STRING ==="
echo "LLVM binaries can be found in $BUILD_DIR/bin"

function notify_finish() {
    if command -v notify &>/dev/null; then
        notify "LLVM build completed successfully in $TIME_STRING! LLVM binaries can be found in $BUILD_DIR/bin" || true # Don't fail if notify fails
    fi
}
notify_finish
