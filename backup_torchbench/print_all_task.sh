#!/bin/bash
# This script prints all models' categories in torchbenchmark/models
benchmark_path=${benchmark_path:-"/home/yhao24/p/p8/benchmark"}
all_models=`ls $benchmark_path/torchbenchmark/models`
cd $benchmark_path/
output="/tmp/model_class.txt"
for model in $all_models
do
    # if model ends with .md, skip it
    if [[ $model == *.md ]]; then
        continue
    fi
    echo -n "$model, " >> $output
    python3 -c "from torchbenchmark.models.$model import Model; print(Model.task)" >> $output
done