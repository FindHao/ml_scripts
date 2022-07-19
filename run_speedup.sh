#!/bin/bash



model=speech_transformer
output=/tmp/run.log
echo "" > $output
cd /home/yhao/d/benchmark
func(){
    for i in {1..20} ; do
        python run.py -d cuda -t train --precision fp32  $model >> $output 2>&1
    done
}

source /home/yhao/d/conda/bin/activate
conda activate pt
echo "@Yueming Hao origin" >>$output
func
conda activate optimize
echo "@Yueming Hao optimize" >>$output
func