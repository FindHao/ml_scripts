#!/bin/bash

wget -i cu116.txt -P ./.downloads/cu116
wget -i cu117.txt -P ./.downloads/cu117
wget https://download.pytorch.org/whl/nightly/torchtext-0.14.0.dev20221026-cp38-cp38-linux_x86_64.whl -P ./.downloads/
wget https://download.pytorch.org/whl/nightly/torchdata-0.6.0.dev20221026-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -P ./.downloads/