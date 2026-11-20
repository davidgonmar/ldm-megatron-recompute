#!/bin/bash
# Environment for running OpenEuroLLM's Megatron hybrid stack on our cluster: their base image
# (NGC PyTorch 26.09) plus the few pip extras the CKDA / MoE / LDM speed arms import, installed
# into a user prefix on /fast. Megatron, their FLA fork and cute_kernels come from the checkout
# on PYTHONPATH at run time, as in their launcher. Nothing is written under the home directory.
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
W=/fast/dmartinez/oellm-ldm-work
REPO=/fast/dmartinez/oellm-autoexp
SIF=$W/images/pytorch_26.09-py3.sif
export APPTAINER_CACHEDIR=$W/cache/apptainer
export APPTAINER_TMPDIR=${_CONDOR_SCRATCH_DIR:-/tmp}/apptainer_tmp  # node-local: extraction on Lustre is slow
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$W/cache/pip"

if [ ! -f "$SIF" ]; then
    apptainer pull "$SIF.partial" docker://nvcr.io/nvidia/pytorch:26.09-py3
    mv "$SIF.partial" "$SIF"
fi

apptainer exec --cleanenv --bind /fast \
    --env HOME=$W/home,PYTHONUSERBASE=$W/pyuser,PIP_CACHE_DIR=$W/cache/pip,REPO=$REPO,W=$W \
    "$SIF" bash -euo pipefail -c '
python3 -c "import torch, transformer_engine; print(\"torch\", torch.__version__, \"te\", transformer_engine.__version__)"
python3 -c "import cutlass; print(\"cutlass dsl\", cutlass.__version__)" || echo "no CuTe DSL in the image"
# keep every package the image ships at its version
python3 "$REPO/container/gen_constraints.py" --output "$W/constraints.txt" \
    --exclude pip setuptools wheel huggingface_hub apache-tvm-ffi
python3 -c "import cutlass" 2>/dev/null || \
    python3 -m pip install --user -c "$W/constraints.txt" "nvidia-cutlass-dsl[cu13]==4.7.1"
python3 -m pip install --user -c "$W/constraints.txt" \
    compoconf omegaconf hydra-core einops opt-einsum sentencepiece tiktoken liger-kernel cuda-python
python3 -m pip install --user --no-deps "nvidia-resiliency-ext==0.6.0"
python3 -m pip install --user --no-deps -e "$REPO"
python3 -c "import torch, cutlass, liger_kernel, compoconf, omegaconf; print(\"imports ok\", torch.__version__)"
'
echo SETUP_DONE
