#!/bin/bash
# FP16 + checkpoints at the production shape against FP64 on a token subset (inside run_in_env.sh).
set -uo pipefail
cd "$CUTE_DIR"
python3 /fast/pnazari/ldm-work/scripts/prod_shape_check.py 2>&1 | grep -v "UserWarning\|schedule = \|warnings.warn"
