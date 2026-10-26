#!/bin/bash
# The environment the benchmark and tests ran in: OpenEuroLLM's base image (NGC PyTorch 26.09)
# plus the few pip extras the hybrid stack imports, installed into a user prefix under $WORK.
# Megatron, their FLA fork and cute_kernels come from the oellm-autoexp checkout on PYTHONPATH
# at run time (run_in_env.sh), as in their launcher.
#   WORK=/path/to/workdir OELLM_REPO=/path/to/oellm-autoexp setup_env.sh
set -euo pipefail
: "${WORK:?set WORK}" "${OELLM_REPO:?set OELLM_REPO}"
SIF=$WORK/images/pytorch_26.09-py3.sif
export APPTAINER_CACHEDIR=$WORK/cache/apptainer
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-/tmp/apptainer_tmp}  # node-local: extraction on a parallel FS is slow
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$WORK/cache/pip" "$WORK/images" "$WORK/home"

if [ ! -f "$SIF" ]; then
    apptainer pull "$SIF.partial" docker://nvcr.io/nvidia/pytorch:26.09-py3
    mv "$SIF.partial" "$SIF"
fi

apptainer exec --cleanenv --bind "$WORK" --bind "$OELLM_REPO" \
    --env HOME=$WORK/home,PYTHONUSERBASE=$WORK/pyuser,PIP_CACHE_DIR=$WORK/cache/pip,REPO=$OELLM_REPO,W=$WORK \
    "$SIF" bash -euo pipefail -c '
# keep every package the image ships at its version
python3 "$REPO/container/gen_constraints.py" --output "$W/constraints.txt" \
    --exclude pip setuptools wheel huggingface_hub apache-tvm-ffi
python3 -c "import cutlass" 2>/dev/null || \
    python3 -m pip install --user -c "$W/constraints.txt" "nvidia-cutlass-dsl[cu13]==4.7.1"
python3 -m pip install --user -c "$W/constraints.txt" \
    compoconf omegaconf hydra-core einops opt-einsum sentencepiece tiktoken liger-kernel cuda-python \
    transformers pytest
python3 -m pip install --user --no-deps "nvidia-resiliency-ext==0.6.0"
python3 -m pip install --user --no-deps -e "$REPO"
python3 -c "import torch, cutlass, liger_kernel; print(\"imports ok\", torch.__version__)"
'
