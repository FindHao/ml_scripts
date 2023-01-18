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

## nightly pytorch installation

```
down_nightly_torch.py --date 20230102
```
This script will automatically download and analyze the package dependencies of the nightly pytorch build from the given date. The date format is `YYYYMMDD`. e.g., torchvision built in 20230102 may rely on torch 20230101 while torchtext 20230102 may rely on torch 20230102. All packages will be downloaded to `.downloads/cu116`. If no conflict dependcy found, you can use the following cmomand to install the nightly torch.

```
cd .downloads/ && mkdir 20230102 && cd 20230102
for file in `ls ../cu116/*20230102*`; do 
ln -s $file
done
ln -s ../cu116/pytorch-triton-XXX

pip install ./*
```