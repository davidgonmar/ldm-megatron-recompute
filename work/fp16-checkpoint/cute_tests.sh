#!/bin/bash
# cute_ldm: the fp16 / checkpoint tests, the existing numerics tests (BF16 / FP16 regressions),
# then the timing benchmark at the v4b shape (inside run_in_env.sh).
set -uo pipefail
cd /fast/pnazari/ldm-work/cute_kernels
python3 -m pytest -q -p no:cacheprovider tests/test_ldm_fp16_checkpoint.py tests/test_ldm_numerics.py
echo "pytest exit $?"
python3 /fast/pnazari/ldm-work/scripts/ldm_ckpt_bench.py 16384
