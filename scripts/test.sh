#!/bin/bash
# The tests of the mode on one Hopper / Blackwell GPU: the kernels and the read kernel against
# the PyTorch oracle, and the full model under full recompute (1 to 4 layers per recompute group,
# dense and MoE patterns, two micro-batches) against the oracle and the kernels without recompute.
#   WORK=... OELLM_REPO=... test.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec "$HERE/run_in_env.sh" "cd submodules/Megatron-LM && torchrun --nproc-per-node 1 -m pytest -q -p no:cacheprovider \
  tests/unit_tests/transformer/test_depth_memory_cute.py \
  tests/unit_tests/models/test_hybrid_depth_memory_segments.py::test_cute_recompute_matches_one_sequence"
