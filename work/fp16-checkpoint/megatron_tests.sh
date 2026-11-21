#!/bin/bash
# Megatron's LDM tests against our cute_kernels branch (inside run_in_env.sh): the new fp16 /
# checkpoint tests and the existing CuTe ones (regressions).
set -uo pipefail
cd /fast/pnazari/ldm-work/Megatron-LM
torchrun --nproc-per-node 1 -m pytest -q -p no:cacheprovider \
    tests/unit_tests/models/test_hybrid_depth_memory_segments.py \
    tests/unit_tests/transformer/test_depth_memory_cute.py
echo "pytest exit $?"
