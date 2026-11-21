#!/bin/bash
# The fp16 / checkpoint tests, the error table and the timing benchmark against $CUTE_DIR (inside run_in_env.sh).
set -uo pipefail
cd "$CUTE_DIR"
python3 -m pytest -q -p no:cacheprovider tests/test_ldm_fp16_checkpoint.py
echo "pytest exit $?"
python3 tests/test_ldm_fp16_checkpoint.py
python3 /fast/pnazari/ldm-work/scripts/ldm_ckpt_bench.py 16384
