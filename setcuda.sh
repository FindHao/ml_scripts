#!/bin/bash
export CUDA_PATH=/scratch/opt/cuda-12.1
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/compute-sanitizer:$LD_LIBRARY_PATH 
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH
source /scratch/setenvs/setncu_nsys.sh
