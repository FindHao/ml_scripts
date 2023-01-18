# ml_scripts

This is a collection of scripts including cuda installation, nightly pytorch installation, torchbench tests, and more.

## torchbench scripts

For scripts that source `run_base.sh`, the run command is like
```
work_path=/home/yhao/d/p8 env1=pt_jan02 env2=pt_oct26_cu117 cuda_env1=/home/yhao/setenvs/set11.6.sh mode=train  ./run_all_speedup_cuda.sh
```
The detailed explanation of the arguments is in `run_base.sh`.

## cuda installation

```
t_cuda=118 t_cuda_path=/install/path  ./install_cuda.sh
```
It will automatically download and install cuda 11.8 and cudnn 8.5.0 to the path `/install/path/cuda-11.8`.
