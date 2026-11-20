#!/bin/bash
# run_in_env.sh on the frozen snapshot (growth/snap): their rematch commits plus the residual metric.
# their PYTHONPATH order: the checkout, Megatron, the FLA fork, cute_kernels. Every cache and
# HOME are under /fast/dmartinez/oellm-ldm-work.
#   run_in_env.sh <command ...>
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
W=/fast/dmartinez/oellm-ldm-work
REPO=/fast/dmartinez/oellm-ldm-work/growth/snap
SIF=$W/images/pytorch_26.09-py3.sif
PYPATH=$REPO:$REPO/submodules/Megatron-LM:$REPO/flash-linear-attention:$REPO/submodules/cute_kernels
# compile caches on node-local disk: thousands of small files are slow on Lustre
SCR=${_CONDOR_SCRATCH_DIR:-/tmp/$USER-oellm}
C=$SCR/cache
mkdir -p $C
ENVS=PYTHONUSERBASE=$W/pyuser,PIP_CACHE_DIR=$W/cache/pip,CUDA_CACHE_PATH=$C/nv,PYTHONPYCACHEPREFIX=$C/pyc
ENVS=$ENVS,TRITON_CACHE_DIR=$C/triton,TORCHINDUCTOR_CACHE_DIR=$C/inductor
ENVS=$ENVS,XDG_CACHE_HOME=$C/xdg,HF_HOME=$W/cache/hf,CUTE_DSL_CACHE_DIR=$C/cute
ENVS=$ENVS,TRITON_LIBCUDA_PATH=/.singularity.d/libs
ENVS=$ENVS,PYTHONPATH=$PYPATH,PYTHONUNBUFFERED=1,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENVS=$ENVS${EXTRA_ENV:+,$EXTRA_ENV}${CUDA_VISIBLE_DEVICES:+,CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES}
cd "$REPO"
exec apptainer exec --nv --cleanenv --home $W/home --bind /fast --bind $SCR --env "$ENVS" "$SIF" bash -c '
python3 -c "import megatron.core, cute_ldm; m = megatron.core.__file__; assert m.startswith(\"/fast/dmartinez/oellm-ldm-work/growth/snap/submodules/Megatron-LM/\"), m; assert cute_ldm.__file__.startswith(\"/fast/dmartinez/oellm-ldm-work/growth/snap/\"), cute_ldm.__file__; print(\"megatron from\", m, \"cute_ldm from\", cute_ldm.__file__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
'"$*"
