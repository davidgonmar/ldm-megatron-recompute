#!/bin/bash
# test_unwind_matches_one_sequence (torch path) 2x on our Megatron branch and 2x on upstream, same node.
set -uo pipefail
T=tests/unit_tests/models/test_hybrid_depth_memory_segments.py::test_unwind_matches_one_sequence
for tree in Megatron-LM Megatron-LM-base; do
  for i in 1 2; do
    cd /fast/pnazari/ldm-work/$tree
    PYTHONPATH=/fast/pnazari/ldm-work/$tree:$PYTHONPATH torchrun --nproc-per-node 1 -m pytest -q -rfE -p no:cacheprovider "$T" 2>&1 \
      | grep -E "^FAILED|^[0-9]+ (passed|failed)|[0-9]+ failed, [0-9]+ passed|Greatest absolute" | sed "s|^|$tree run $i: |"
  done
done
