#!/bin/bash 

conda_dir=${tb_conda_dir:-/home/yhao/d/conda}
source ${conda_dir}/bin/activate
conda create -n pt_oct26 python=3.8 -y
conda activate pt_oct26

conda env list
pip install requests bs4 argparse oauthlib pyyaml 
pip install .downloads/*.whl .downloads/cu116/*.whl 
conda install git-lfs -y
cd ../benchmark
python install.py
conda deactivate

cd ../ml_optimizations

# cu117
conda create -n pt_oct26_cu117 python=3.8 -y
conda activate pt_oct26_cu117
pip install requests bs4 argparse oauthlib pyyaml
pip install .downloads/*.whl .downloads/cu117/*.whl 
conda install git-lfs -y
cd ../benchmark
python install.py
