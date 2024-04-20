
# write a bash function to set precision based on mode
function set_precision() {
    if [ "$mode" = "inference" ]; then
        precision="bfloat16"
    elif [ "$mode" = "training" ]; then
        precision="amp"
    else
        echo "mode not supported"
    fi
}

# sat
log_path="/tmp/yhao/"
profile_path="/home/users/yhao24/b/tmp/profile"
collection="torchbench"

# gil
log_path="/scratch/yhao/logs/runlog"
profile_path="/scratch/yhao/logs/profile"
collection="torchbench"

# dev 
log_path="/tmp/yhao/"
profile_path="/home/yhao/p9/profile"
collection="torchbench"

mode="inference"

mode="training"

set_precision

TORCHINDUCTOR_BYPASS_TINY=0

# profile
# TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/${collection}.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_${model}_multiple  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1
# TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/${collection}.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_${model}_single  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1


TORCH_COMPILE_DEBUG=1  TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/torchbench.py  --performance --${precision} -dcuda --${mode}  --inductor --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1

TORCHINDUCTOR_MULTIPLE_STREAMS_PROFILING=1
# profile 
TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/${collection}.py  --performance --${precision} -dcuda --${mode} --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_multiple  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1
TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/${collection}.py  --performance --${precision} -dcuda --${mode} --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_single  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1

TORCHINDUCTOR_MULTIPLE_STREAMS=0 python benchmarks/dynamo/torchbench.py  --performance --${precision} -dcuda --${mode} --inductor  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1

# pure run
TORCHINDUCTOR_MULTIPLE_STREAMS=1 python benchmarks/dynamo/${collection}.py  --accuracy --${precision} -dcuda --${mode} --inductor  --disable-cudagraphs   --only ${model}
TORCHINDUCTOR_MULTIPLE_STREAMS=1 python benchmarks/dynamo/${collection}.py  --performance --${precision} -dcuda --${mode} --inductor  --disable-cudagraphs   --only ${model}
TORCHINDUCTOR_MULTIPLE_STREAMS=0 python benchmarks/dynamo/${collection}.py  --performance --${precision} -dcuda --${mode} --inductor  --disable-cudagraphs  --only ${model}

# loading exisiting stream assignment
TORCHINDUCTOR_LOAD_EXISTING_STREAM_ASSIGNMENT=/tmp/yhao/debug2023/resnet18_stream_assignment.update.json TORCHINDUCTOR_MULTIPLE_STREAMS=1 
/tmp/yhao/debug2023/resnet50_stream_assignment.update.json
# debug

TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/${collection}.py  --performance --${precision} -dcuda --${mode} --inductor  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/${collection}.py  --performance --${precision} -dcuda --${mode} --inductor  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1




## cpp wrapper

TORCHINDUCTOR_MULTIPLE_STREAMS=1 python benchmarks/dynamo/torchbench.py  --accuracy --${precision} -dcuda --inference --inductor  --disable-cudagraphs --cpp-wrapper  --only ${model}
TORCHINDUCTOR_MULTIPLE_STREAMS=1 python benchmarks/dynamo/torchbench.py  --performance --${precision} -dcuda --inference --inductor  --disable-cudagraphs --cpp-wrapper  --only ${model}
TORCHINDUCTOR_MULTIPLE_STREAMS=0 python benchmarks/dynamo/torchbench.py  --performance --${precision} -dcuda --inference --inductor  --disable-cudagraphs --cpp-wrapper  --only ${model}

TORCHINDUCTOR_MULTIPLE_STREAMS=0 python benchmarks/dynamo/torchbench.py  --performance --${precision} -dcuda --inference --inductor --disable-cudagraphs --cpp-wrapper  --only ${model}
 
TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/torchbench.py --cpp-wrapper   --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_multiple_cpp  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1
TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/torchbench.py --cpp-wrapper  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_single_cpp  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1


TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/torchbench.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_multiple_cpp  --disable-cudagraphs --cpp-wrapper    --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/torchbench.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_single_cpp  --disable-cudagraphs --cpp-wrapper   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1







single_stream=0 mode=inference test=acc work_path=/home/users/yhao24/gil/p9 ./run_all.sh;
single_stream=0 mode=inference test=perf work_path=/home/users/yhao24/gil/p9 ./run_all.sh;
single_stream=1 mode=inference test=perf work_path=/home/users/yhao24/gil/p9 ./run_all.sh;








# timm_models
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/timm_models.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_multiple  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/timm_models.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_single  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1

TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/timm_models.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_multiple  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1
TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/timm_models.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_single  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1

TORCHINDUCTOR_MULTIPLE_STREAMS=1 python benchmarks/dynamo/timm_models.py  --accuracy --${precision} -dcuda --inference --inductor  --disable-cudagraphs   --only ${model}
TORCHINDUCTOR_MULTIPLE_STREAMS=1 python benchmarks/dynamo/timm_models.py  --performance --${precision} -dcuda --inference --inductor  --disable-cudagraphs   --only ${model}
TORCHINDUCTOR_MULTIPLE_STREAMS=0 python benchmarks/dynamo/timm_models.py  --performance --${precision} -dcuda --inference --inductor  --disable-cudagraphs  --only ${model}

TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/timm_models.py  --performance --${precision} -dcuda --inference --inductor  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/timm_models.py  --performance --${precision} -dcuda --inference --inductor  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1


TORCHDYNAMO_REPRO_AFTER=aot TORCHDYNAMO_REPRO_LEVEL=4 TORCHINDUCTOR_MULTIPLE_STREAMS=1 python benchmarks/dynamo/timm_models.py  --accuracy --${precision} -dcuda --inference --inductor  --disable-cudagraphs   --only ${model}

## cpp wrapper

TORCHINDUCTOR_MULTIPLE_STREAMS=1 python benchmarks/dynamo/timm_models.py  --accuracy --${precision} -dcuda --inference --inductor  --disable-cudagraphs --cpp-wrapper  --only ${model}
TORCHINDUCTOR_MULTIPLE_STREAMS=1 python benchmarks/dynamo/timm_models.py  --performance --${precision} -dcuda --inference --inductor  --disable-cudagraphs --cpp-wrapper  --only ${model}
TORCHINDUCTOR_MULTIPLE_STREAMS=0 python benchmarks/dynamo/timm_models.py  --performance --${precision} -dcuda --inference --inductor  --disable-cudagraphs --cpp-wrapper --only ${model}


# huggingface
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/huggingface.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_multiple  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/huggingface.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_single  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1

TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=0 TORCHINDUCTOR_STREAM_PRINT_GRAPH=0 TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/huggingface.py  --performance --amp -dcuda --training --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_multiple  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=0 TORCHINDUCTOR_STREAM_PRINT_GRAPH=0 TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/huggingface.py  --performance --amp -dcuda --training --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_single  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1


TORCHINDUCTOR_MULTIPLE_STREAMS=0 python benchmarks/dynamo/huggingface.py  --performance --${precision} -dcuda --inference --inductor  --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_single  --disable-cudagraphs   --only ${model}  

TORCHINDUCTOR_MULTIPLE_STREAMS=1 python benchmarks/dynamo/huggingface.py  --performance --${precision} -dcuda --inference --inductor  --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_multi  --disable-cudagraphs   --only ${model}  


TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/huggingface.py  --accuracy --${precision} -dcuda --inference --inductor --disable-cudagraphs  --only ${model} 
TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/huggingface.py  --performance --${precision} -dcuda --inference --inductor --disable-cudagraphs  --only ${model} 
TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/huggingface.py  --performance --${precision} -dcuda --inference --inductor --disable-cudagraphs  --only ${model} 


TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/huggingface.py  --accuracy --${precision} -dcuda --inference --inductor --disable-cudagraphs  --only ${model} 
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/huggingface.py  --performance --${precision} -dcuda --inference --inductor --disable-cudagraphs  --only ${model} 

## cpp wrapper

TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/huggingface.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --cpp-wrapper --profiler_trace_name ${profile_path}/$(mydate)_multiple  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/huggingface.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --cpp-wrapper --profiler_trace_name ${profile_path}/$(mydate)_single  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1


TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/huggingface.py  --accuracy --${precision} -dcuda --inference --inductor --disable-cudagraphs  --cpp-wrapper --only ${model} 
TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/huggingface.py  --performance --${precision} -dcuda --inference --inductor --disable-cudagraphs  --cpp-wrapper --only ${model} 
TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/huggingface.py  --performance --${precision} -dcuda --inference --inductor --disable-cudagraphs --cpp-wrapper  --only ${model} 

# export trace
TORCHINDUCTOR_MULTIPLE_STREAMS=0 python benchmarks/dynamo/huggingface.py  --performance --${precision} -dcuda --inference --inductor --cpp-wrapper --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_single_cpp  --disable-cudagraphs   --only ${model}  

TORCHINDUCTOR_MULTIPLE_STREAMS=1 python benchmarks/dynamo/huggingface.py  --performance --${precision} -dcuda --inference --inductor --cpp-wrapper  --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_multi_cpp  --disable-cudagraphs   --only ${model}  



# devgpu

TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/timm_models.py  --performance --${precision} -dcuda --inference --inductor  --disable-cudagraphs   --only ${model} >  ~/p9/tmp/${model}_$(mydate).log 2>&1# torchbench

TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/torchbench.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_multiple  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/torchbench.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_single  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1

TORCH_COMPILE_DEBUG=1  TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/torchbench.py  --performance --${precision} -dcuda --inference --inductor --cpp-wrapper --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1


TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/torchbench.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_multiple  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1
TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/torchbench.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_single  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1

TORCHINDUCTOR_MULTIPLE_STREAMS=0 python benchmarks/dynamo/torchbench.py  --performance --${precision} -dcuda --inference --inductor  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1

# timm_models
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/timm_models.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_multiple  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_GRAPH_DIAGRAM=1 TORCHINDUCTOR_STREAM_PRINT_GRAPH=1 TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/timm_models.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_single  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1

TORCHINDUCTOR_MULTIPLE_STREAMS=1  python benchmarks/dynamo/timm_models.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_multiple  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1
TORCHINDUCTOR_MULTIPLE_STREAMS=0  python benchmarks/dynamo/timm_models.py  --performance --${precision} -dcuda --inference --inductor --export-profiler-trace --profiler_trace_name ${profile_path}/$(mydate)_single  --disable-cudagraphs   --only ${model} >  ${log_path}/${model}_$(mydate).log 2>&1



# run all 
export TORCHINDUCTOR_BYPASS_TINY=0 
mode=training test=perf cpp_wrapper=0 log_path=/home/users/yhao24/b/p9/inductor_logs/nobypass_$(mydate) ./run_all.sh
mode=training test=perf cpp_wrapper=0 single_stream=1 log_path=/home/users/yhao24/b/p9/inductor_logs/nobypass_$(mydate) ./run_all.sh
mode=training test=acc cpp_wrapper=0 log_path=/home/users/yhao24/b/p9/inductor_logs/nobypass_$(mydate) ./run_all.sh
mode=inference test=perf cpp_wrapper=0 log_path=/home/users/yhao24/b/p9/inductor_logs/nobypass_$(mydate) ./run_all.sh
mode=inference test=perf cpp_wrapper=0 single_stream=1 log_path=/home/users/yhao24/b/p9/inductor_logs/nobypass_$(mydate) ./run_all.sh
mode=inference test=acc cpp_wrapper=0 log_path=/home/users/yhao24/b/p9/inductor_logs/nobypass_$(mydate) ./run_all.sh
export TORCHINDUCTOR_BYPASS_TINY=1
mode=training test=perf cpp_wrapper=0 log_path=/home/users/yhao24/b/p9/inductor_logs/bypass ./run_all.sh;
mode=training test=perf cpp_wrapper=0 single_stream=1 log_path=/home/users/yhao24/b/p9/inductor_logs/bypass ./run_all.sh;
mode=training test=acc cpp_wrapper=0 log_path=/home/users/yhao24/b/p9/inductor_logs/bypass ./run_all.sh;
mode=inference test=perf cpp_wrapper=0 log_path=/home/users/yhao24/b/p9/inductor_logs/bypass ./run_all.sh;
mode=inference test=perf cpp_wrapper=0 single_stream=1 log_path=/home/users/yhao24/b/p9/inductor_logs/bypass ./run_all.sh;
mode=inference test=acc cpp_wrapper=0 log_path=/home/users/yhao24/b/p9/inductor_logs/bypass ./run_all.sh;

export TORCHINDUCTOR_BYPASS_TINY=0 

# run all on dev

work_path=/home/yhao/p9 conda_dir=/home/yhao/miniconda3 mode=training test=acc cpp_wrapper=0 log_path=/home/yhao/p9/logs ./run_all.sh;
work_path=/home/yhao/p9 conda_dir=/home/yhao/miniconda3 mode=training test=perf cpp_wrapper=0 log_path=/home/yhao/p9/logs ./run_all.sh;
work_path=/home/yhao/p9 conda_dir=/home/yhao/miniconda3 mode=training test=perf cpp_wrapper=0 single_stream=1 log_path=/home/yhao/p9/logs ./run_all.sh;
work_path=/home/yhao/p9 conda_dir=/home/yhao/miniconda3 mode=inference test=acc cpp_wrapper=0 log_path=/home/yhao/p9/logs ./run_all.sh;
work_path=/home/yhao/p9 conda_dir=/home/yhao/miniconda3 mode=inference test=perf cpp_wrapper=0 log_path=/home/yhao/p9/logs ./run_all.sh;
work_path=/home/yhao/p9 conda_dir=/home/yhao/miniconda3 mode=inference test=perf cpp_wrapper=0 single_stream=1 log_path=/home/yhao/p9/logs ./run_all.sh;
