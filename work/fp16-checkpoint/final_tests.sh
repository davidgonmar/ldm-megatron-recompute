#!/bin/bash
# The final patch: every cute_ldm test (new and existing numerics) against $CUTE_DIR (inside run_in_env.sh).
set -uo pipefail
cd "$CUTE_DIR"
python3 -m pytest -q -p no:cacheprovider tests/test_ldm_fp16_checkpoint.py tests/test_ldm_numerics.py 2>&1 | tail -3
