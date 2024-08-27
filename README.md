# ml_scripts

This is a collection of scripts including cuda installation, nightly pytorch installation, torchbench tests, and more.

## cuda installation

```
CUDA_VERSION=12.1 CUDA_INSTALL_PREFIX=/install/path  ./install_cuda.sh
```
It will automatically download and install cuda 12.1, cudnn, and nccl to the path `/install/path/cuda-12.1`.


## PyTorch + TorchBench Compilation

```bash
# set cuda path before install
# set conda env with python 3.10
conda create -n pt_compiled python=3.10
conda activate pt_compiled
# compile pytorch and other packages
work_path=~/project/ compile_torch.sh
```

## torchbench scripts

For scripts that source `run_base.sh`, the run command is like
```
work_path=/home/yhao/d/p8 env1=pt_jan02 env2=pt_oct26_cu117 cuda_env1=/home/yhao/setenvs/set11.6.sh mode=train  ./run_all_speedup_cuda.sh
```
The detailed explanation of the arguments is in `run_base.sh`.

