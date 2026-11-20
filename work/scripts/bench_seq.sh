#!/bin/bash
# Several arms back to back on the same node, so step times compare without node variance.
# usage: bench_seq.sh MBS ARM [ARM ...]
MBS=$1; shift
W=/fast/dmartinez/oellm-ldm-work
echo "SEQ on $(hostname): $*"
for ARM in "$@"; do
  ITERS=${ITERS:-10} bash $W/scripts/bench.sh $ARM $MBS 4 > $W/logs/seq-$ARM-m$MBS-$CLUSTER.out 2> $W/logs/seq-$ARM-m$MBS-$CLUSTER.err
  echo "DONE $ARM: $(grep -E 'iteration +[0-9]+/' $W/logs/seq-$ARM-m$MBS-$CLUSTER.out | sed -E 's/.*iteration +([0-9]+)\/.*elapsed time per iteration \(ms\): ([0-9.]+).*/\1:\2/' | awk -F: '$1>=3{printf "%.1f ", $2/1000}')"
done
