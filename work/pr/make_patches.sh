#!/bin/bash
# Regenerate the patches of both submodules from their working trees (nothing is committed).
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
out=/fast/dmartinez/oellm-ldm-work/pr
root=/fast/dmartinez/oellm-autoexp/submodules
( cd $root/cute_kernels && git add -N cute_ldm/kernels/read.py && git diff > $out/cute_kernels.patch && git reset -q cute_ldm/kernels/read.py )
( cd $root/Megatron-LM && git diff > $out/megatron.patch )
wc -l $out/*.patch
