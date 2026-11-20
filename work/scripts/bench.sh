#!/bin/bash
# One arm of the residual speed comparison on one 8x H100 node: their production speed arm
# (ep_ckptoff_mn5 e_db family, rendered by their launcher; flags in bench/base_args.txt) on mock
# data, adapted to one node: 32 experts (not 112), EP 4, the all-to-all dispatcher (not MoonEP).
#   bench.sh ARM MBS     ARM: plain | mhc4 | ldm_cute_segments | ldm_torch_unwind | ldm_cute_unwind
set -euo pipefail
ARM=$1; MBS=$2; NGPU=${3:-8}
W=/fast/dmartinez/oellm-ldm-work
OUT=$W/bench/${ARM}_m${MBS}
mkdir -p "$OUT"
case $ARM in
  plain)             EXTRA="--recompute-num-layers 2" ;;
  mhc4)              EXTRA="--mhc --recompute-num-layers 2" ;;
  ldm_cute_segments) EXTRA="--depth-memory --depth-memory-impl cute --depth-memory-segments --recompute-num-layers 2" ;;
  ldm_torch_unwind)  EXTRA="--depth-memory --depth-memory-impl torch --depth-memory-unwind --recompute-num-layers 1" ;;
  ldm_cute_unwind2)  EXTRA="--depth-memory --depth-memory-impl cute --depth-memory-unwind --recompute-num-layers 2" ;;
  ldm_cute_unwind1)  EXTRA="--depth-memory --depth-memory-impl cute --depth-memory-unwind --recompute-num-layers 1" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
ARGS="$(tr '\n' ' ' < $W/bench/base_args.txt) $EXTRA \
 --mock-data --tokenizer-type NullTokenizer --vocab-size 262144 \
 --num-experts 32 --expert-model-parallel-size 4 --moe-token-dispatcher-type alltoall \
 --micro-batch-size $MBS --global-batch-size $((NGPU * MBS * 4)) --train-iters ${ITERS:-15} \
 --data-cache-path $OUT/data_cache --tensorboard-dir $OUT/tb"
if [ "${PROFILE:-0}" = "1" ]; then   # a torch-profiler trace of steps 6-7 on rank 0, into $OUT/tb
  ARGS="$ARGS --profile --use-pytorch-profiler --profile-step-start 6 --profile-step-end 8 --profile-ranks 0"
fi
export EXTRA_ENV="RECOMPUTE_OFFLOAD=1,MHC_FUSED_READ=1,MEM_BREAKDOWN=1,LIGER_WGRAD_ADDMM=1,MEGATRON_LAZY_SINGLETON_GROUPS=1,FLA_TENSOR_CACHE_SIZE=64,CUDA_DEVICE_MAX_CONNECTIONS=8,TOKENIZERS_PARALLELISM=false${MEM_TRACE:+,MEM_TRACE_DIR=$OUT/mem_trace,MEM_TRACE_ITER=2}"
echo "ARM $ARM MBS $MBS GPUS $NGPU on $(hostname)"
exec $W/scripts/run_in_env.sh "torchrun --standalone --nproc-per-node $NGPU ./submodules/Megatron-LM/pretrain_hybrid.py $ARGS"
