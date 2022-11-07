#!/bin/bash

if [ ! -d $cudnn_path ]; then
    echo "cudnn_path not exist"
    exit 1
fi

if [ ! -d $cuda_path ]; then
    echo "cuda_path not exist"
    exit 1
fi

cp $cudnn_path/include/cudnn*.h $cuda_path/include
cp $cudnn_path/lib64/libcudnn* $cuda_path/lib64
chmod a+r $cuda_path/include/cudnn*.h $cuda_path/lib64/libcudnn*