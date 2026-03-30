# This file is not a runnable script but a set of instructions to install Triton.
# Sometimes, pip install triton does not work due to various issues. This script will let triton installation use
# the conda environment to install the required dependencies.

conda create -n pta python=3.13
conda activate pta


# option 1
conda install libstdcxx-ng lld zstd zlib libgcc-ng ninja ccache cmake -y -c conda-forge
# option 2
conda install -c conda-forge gcc=13.2.0
# optional
conda install clang=21 clangxx=21 -y -c conda-forge
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CXXFLAGS="-I$CONDA_PREFIX/include"
export LDFLAGS="-L$CONDA_PREFIX/lib"

# if reports error `Host compiler does not support '-fuse-ld=lld'.`
# set -DLLVM_ENABLE_LLD=OFF in `scripts/build-llvm-project.sh`

rm -rf .llvm-project build
# either compile llvm again 
make dev-install-llvm
# or directly try the pip install 
pip install -e .
