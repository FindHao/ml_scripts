#!/bin/bash

var_date=$(date +%Y%m%d_%H%M%S)
output_file="run_test_$var_date.log"
for i in {1..20}; do
  python benchmarks/dynamo/torchbench.py --performance --amp -dcuda --training --inductor --only attention_is_all_you_need_pytorch --cold-start-latency >>$output_file 2>&1
done
notify "Inductor test finished. the log is saved to run_test_$var_date.log"
