#!/bin/bash
# Run a command in the environment of setup_env.sh, from the oellm-autoexp checkout, with their
# PYTHONPATH order: the checkout, Megatron, the FLA fork, cute_kernels.
#   WORK=... OELLM_REPO=... run_in_env.sh <command ...>
set -euo pipefail
: "${WORK:?set WORK}" "${OELLM_REPO:?set OELLM_REPO}"
SIF=$WORK/images/pytorch_26.09-py3.sif
PYPATH=$OELLM_REPO:$OELLM_REPO/submodules/Megatron-LM:$OELLM_REPO/flash-linear-attention:$OELLM_REPO/submodules/cute_kernels
# compile caches on node-local disk: thousands of small files are slow on a parallel FS
SCR=${SCRATCH_DIR:-/tmp/$USER-oellm}
C=$SCR/cache
mkdir -p "$C"
ENVS=PYTHONUSERBASE=$WORK/pyuser,PIP_CACHE_DIR=$WORK/cache/pip,CUDA_CACHE_PATH=$C/nv,PYTHONPYCACHEPREFIX=$C/pyc
ENVS=$ENVS,TRITON_CACHE_DIR=$C/triton,TORCHINDUCTOR_CACHE_DIR=$C/inductor
ENVS=$ENVS,XDG_CACHE_HOME=$C/xdg,HF_HOME=$WORK/cache/hf,CUTE_DSL_CACHE_DIR=$C/cute
ENVS=$ENVS,TRITON_LIBCUDA_PATH=/.singularity.d/libs
ENVS=$ENVS,PYTHONPATH=$PYPATH,PYTHONUNBUFFERED=1,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENVS=$ENVS${EXTRA_ENV:+,$EXTRA_ENV}${CUDA_VISIBLE_DEVICES:+,CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES}
cd "$OELLM_REPO"
exec apptainer exec --nv --cleanenv --home "$WORK/home" --bind "$WORK" --bind "$OELLM_REPO" --bind "$SCR" \
    --env "$ENVS" "$SIF" bash -c "$*"
