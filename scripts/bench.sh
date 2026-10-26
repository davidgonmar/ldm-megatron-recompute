#!/bin/bash
# One arm of the speed comparison on one node: OpenEuroLLM's production speed arm (the
# o1_ckptoff e_db flags rendered by their launcher, production_args.txt) on mock data, adapted to
# one node: 32 experts instead of 112, EP 4, the all-to-all dispatcher instead of MoonEP.
# Micro-batch 4 x 16k, 4 accumulation steps.
#   WORK=... OELLM_REPO=... bench.sh ARM [NGPU]
#   ARM: plain | mhc4 | ldm_cute_segments | ldm_torch_unwind | ldm_cute_recompute
# PROFILE=1 adds a torch-profiler trace of steps 6-7 on rank 0 (see tools/).
set -euo pipefail
: "${WORK:?set WORK}" "${OELLM_REPO:?set OELLM_REPO}"
ARM=$1; NGPU=${2:-4}; MBS=4
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$WORK/bench/$ARM
mkdir -p "$OUT"
case $ARM in
  plain)              EXTRA="--recompute-num-layers 2" ;;
  mhc4)               EXTRA="--mhc --recompute-num-layers 2" ;;
  ldm_cute_segments)  EXTRA="--depth-memory --depth-memory-impl cute --depth-memory-segments --recompute-num-layers 2" ;;
  ldm_torch_unwind)   EXTRA="--depth-memory --depth-memory-impl torch --depth-memory-unwind --recompute-num-layers 1" ;;
  ldm_cute_recompute) EXTRA="--depth-memory --depth-memory-impl cute --depth-memory-unwind --recompute-num-layers 2" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
ARGS="$(tr '\n' ' ' < "$HERE/production_args.txt") $EXTRA \
 --mock-data --tokenizer-type NullTokenizer --vocab-size 262144 \
 --num-experts 32 --expert-model-parallel-size 4 --moe-token-dispatcher-type alltoall \
 --micro-batch-size $MBS --global-batch-size $((NGPU * MBS * 4)) --train-iters ${ITERS:-15} \
 --data-cache-path $OUT/data_cache --tensorboard-dir $OUT/tb"
if [ "${PROFILE:-0}" = "1" ]; then
  ARGS="$ARGS --profile --use-pytorch-profiler --profile-step-start 6 --profile-step-end 8 --profile-ranks 0"
fi
export EXTRA_ENV="RECOMPUTE_OFFLOAD=1,MHC_FUSED_READ=1,MEM_BREAKDOWN=1,LIGER_WGRAD_ADDMM=1,MEGATRON_LAZY_SINGLETON_GROUPS=1,FLA_TENSOR_CACHE_SIZE=64,CUDA_DEVICE_MAX_CONNECTIONS=8,TOKENIZERS_PARALLELISM=false"
echo "ARM $ARM MBS $MBS GPUS $NGPU on $(hostname)"
exec "$HERE/run_in_env.sh" "torchrun --standalone --nproc-per-node $NGPU ./submodules/Megatron-LM/pretrain_hybrid.py $ARGS"
