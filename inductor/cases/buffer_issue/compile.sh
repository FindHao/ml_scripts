#!/bin/bash
# make sure pytorch_path is defined.
# example: export pytorch_path=/home/yhao/p9/pytorch
set -e
if [ -z "$pytorch_path" ]; then
    echo "Please define pytorch_path in your environment."
    exit 1
fi
# check conda env python
# example export conda_env_path=/home/yhao/.conda/envs/py_compiled_may14
conda_env_path=$(conda info --envs | grep '*' | awk '{print $NF}')
# check cuda path
if [ -z "$cuda_path" ]; then
    echo "Please define cuda_path in your environment."
    exit 1
fi

echo "pytorch_path: $pytorch_path"
echo "conda_env_path: $conda_env_path"
echo "cuda_path: $cuda_path"

# get current script absolut path
SCRIPT=$(readlink -f "$0")
# get current script directory
SCRIPTPATH=$(dirname "$SCRIPT")


# // Compile cmd
# //
# // g++ /home/yhao/p9/ml_scripts/inductor/cases/buffer_issue/aoti_impl.cpp -fPIC -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -D_GLIBCXX_USE_CXX11_ABI=1 -I/home/yhao/p9/pytorch/torch/include -I/home/yhao/p9/pytorch/torch/include/torch/csrc/api/include -I/home/yhao/p9/pytorch/torch/include/TH -I/home/yhao/p9/pytorch/torch/include/THC -I/home/yhao/opt/cuda-12.1/include -I/home/yhao/.conda/envs/py_compiled_may14/include/python3.11 -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma -DCPU_CAPABILITY_AVX512 -D USE_CUDA -O3 -DNDEBUG -ffast-math -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -march=native -fopenmp -D C10_USING_CUSTOM_GENERATED_MACROS -c -o /tmp/torchinductor_yhao/cfiwgqmhnw54zqatavj57riugdirk75nrpqhdgyyaa4dlx62na34/cxurg4jagc5ladi6scephaotyisxw7dooonxt62ckzujcwjhxxhu.o
# // Link cmd
# // g++ /tmp/torchinductor_yhao/cfiwgqmhnw54zqatavj57riugdirk75nrpqhdgyyaa4dlx62na34/cxurg4jagc5ladi6scephaotyisxw7dooonxt62ckzujcwjhxxhu.o /tmp/torchinductor_yhao/cfiwgqmhnw54zqatavj57riugdirk75nrpqhdgyyaa4dlx62na34/c4oymiquy7qobjgx36tejs35zeqt24qpemsnzgtfeswmrw6csxbk.o -shared -fPIC -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -D_GLIBCXX_USE_CXX11_ABI=1 -I/home/yhao/p9/pytorch/torch/include -I/home/yhao/p9/pytorch/torch/include/torch/csrc/api/include -I/home/yhao/p9/pytorch/torch/include/TH -I/home/yhao/p9/pytorch/torch/include/THC -I/home/yhao/opt/cuda-12.1/include -I/home/yhao/.conda/envs/py_compiled_may14/include/python3.11 -L/home/yhao/p9/pytorch/torch/lib -L/home/yhao/opt/cuda-12.1/lib64 -L/home/yhao/.conda/envs/py_compiled_may14/lib -ltorch -ltorch_cpu -lgomp -lc10_cuda -lcuda -ltorch_cuda -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma -DCPU_CAPABILITY_AVX512 -D USE_CUDA -O3 -DNDEBUG -ffast-math -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -march=native -fopenmp -D C10_USING_CUSTOM_GENERATED_MACROS -o /tmp/torchinductor_yhao/cfiwgqmhnw54zqatavj57riugdirk75nrpqhdgyyaa4dlx62na34/cxurg4jagc5ladi6scephaotyisxw7dooonxt62ckzujcwjhxxhu.so

# compile
g++ ${SCRIPTPATH}/aoti_impl.cpp -fPIC -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -D_GLIBCXX_USE_CXX11_ABI=1 -I${pytorch_path}/torch/include -I$pytorch_path/torch/include/torch/csrc/api/include -I$pytorch_path/torch/include/TH -I$pytorch_path/torch/include/THC -I${cuda_path}/include -I${conda_env_path}/include/python3.11 -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma -DCPU_CAPABILITY_AVX512 -D USE_CUDA -O3 -DNDEBUG -ffast-math -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -march=native -fopenmp -D C10_USING_CUSTOM_GENERATED_MACROS -c -o $SCRIPTPATH/aoti_impl.o


# link
g++ $SCRIPTPATH/aoti_impl.o /tmp/torchinductor_yhao/cfiwgqmhnw54zqatavj57riugdirk75nrpqhdgyyaa4dlx62na34/c4oymiquy7qobjgx36tejs35zeqt24qpemsnzgtfeswmrw6csxbk.o -shared -fPIC -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -D_GLIBCXX_USE_CXX11_ABI=1 -I${pytorch_path}/torch/include -I$pytorch_path/torch/include/torch/csrc/api/include -I$pytorch_path/torch/include/TH -I$pytorch_path/torch/include/THC -I${cuda_path}/include -I${conda_env_path}/include/python3.11  -L$pytorch_path/torch/lib -L${cuda_path}/lib64 -L${conda_env_path}/lib -ltorch -ltorch_cpu -lgomp -lc10_cuda -lcuda -ltorch_cuda -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma -DCPU_CAPABILITY_AVX512 -D USE_CUDA -O3 -DNDEBUG -ffast-math -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -march=native -fopenmp -D C10_USING_CUSTOM_GENERATED_MACROS -o $SCRIPTPATH/aoti_impl.so
