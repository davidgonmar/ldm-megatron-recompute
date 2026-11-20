#!/bin/bash
# burst.py in the NGC image, with OpenEuroLLM's cute_kernels 4739513 (exported, unmodified) first
# on the path and the pip extras of /fast/dmartinez/oellm-ldm-work/pyuser.
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
W=/fast/dmartinez/oellm-ldm-work
D=$W/flagdiag
SCR=${_CONDOR_SCRATCH_DIR:-/tmp/$USER-oellm}
C=$SCR/cache
mkdir -p $C
ENVS=PYTHONUSERBASE=$W/pyuser,PIP_CACHE_DIR=$W/cache/pip,CUDA_CACHE_PATH=$C/nv,PYTHONPYCACHEPREFIX=$C/pyc
ENVS=$ENVS,TRITON_CACHE_DIR=$C/triton,XDG_CACHE_HOME=$C/xdg,CUTE_DSL_CACHE_DIR=$C/cute
ENVS=$ENVS,PYTHONPATH=$D/cute_kernels_4739513,PYTHONUNBUFFERED=1${CUDA_VISIBLE_DEVICES:+,CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES}
cd $D
exec apptainer exec --nv --cleanenv --home $W/home --bind /fast --bind $SCR --env "$ENVS" $W/images/pytorch_26.09-py3.sif bash -c '
python3 -c "import cute_ldm; print(\"cute_ldm from\", cute_ldm.__file__)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L; python3 -c "import torch; torch.zeros(1, device=\"cuda\"); print(\"cuda ok\", torch.cuda.get_device_name())"
python3 '"$*"
