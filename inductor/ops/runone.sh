mydate=$(date +%Y%m%d%H%M%S)
target_dir=./embedding_$(mydate)
mkdir -p $target_dir

for i in {0..15}; do
  python run.py --op embedding --mode fwd --num-inputs 1 --input-id $i --precision fp32 --metrics latency,speedup --cudagraph --csv --output $target_dir/${i}.csv
done

export TORCH_COMPILE_DEBUG=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=0

for i in {0..15}; do
  python run.py --op embedding --mode fwd --num-inputs 1 --input-id $i --precision fp32 --metrics latency,speedup --cudagraph
done
