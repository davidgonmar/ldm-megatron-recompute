#!/bin/bash
# One arm of the residual-growth test: OpenEuroLLM's screen5resgqa s1 rematch (their rendered plain
# arm, base_args.txt) on one 8-GPU node, FineWeb-Edu (GPT-NeoX tokenizer) instead of their flag
# mix, no CUDA graphs, no checkpoints, residual size per layer every 50 steps (tensorboard).
#   run_arm.sh plain | mhc4 | ldm
set -euo pipefail
ARM=$1
G=/fast/dmartinez/oellm-ldm-work/growth
OUT=$G/runs/$ARM
mkdir -p $OUT
case $ARM in
  plain) EXTRA="" ;;
  mhc4)  EXTRA="--mhc --residual-mixer-lr-mult 0.25" ;;
  ldm)   EXTRA="--depth-memory --depth-memory-impl cute --residual-mixer-lr-mult 2.0" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
ARGS="$(tr '\n' ' ' < $G/base_args.txt) $EXTRA \
 --data-path $G/data/fwedu_neox_4shards --tokenizer-type NullTokenizer --vocab-size 50279 \
 --data-cache-path $OUT/data_cache --tensorboard-dir $OUT/tb"
export EXTRA_ENV="CUDA_DEVICE_MAX_CONNECTIONS=1,OMP_NUM_THREADS=16,TOKENIZERS_PARALLELISM=false"
echo "ARM $ARM on $(hostname): $ARGS"
exec $G/run_in_snap.sh "torchrun --standalone --nproc-per-node 8 ./submodules/Megatron-LM/pretrain_hybrid.py $ARGS"
