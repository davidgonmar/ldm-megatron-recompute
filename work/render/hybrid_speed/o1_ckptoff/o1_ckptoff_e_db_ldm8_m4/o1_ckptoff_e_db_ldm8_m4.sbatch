#!/bin/bash
#SBATCH --account=ehpc533
#SBATCH --job-name=o1_ckptoff_e_db_ldm8_m4
#SBATCH --nodes=8
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:40:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=160
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --output=/fast/dmartinez/oellm-ldm-work/render/hybrid_speed/o1_ckptoff/o1_ckptoff_e_db_ldm8_m4/slurm-%A_%a.log
#SBATCH --ntasks=8
#SBATCH --threads-per-core=2
#SBATCH --gpu-bind=none
#SBATCH --wait-all-nodes=1
#SBATCH --exclusive

export HF_HOME=/fast/dmartinez/oellm-ldm-work/hybrid_env/cache/hf
export WANDB_MODE=offline
export MACHINE_NAME=MARENOSTRUM
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MASTER_ADDR=$(nslookup $(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) | grep "Address: " | sed "s/Address: //" | tail) 
export MASTER_PORT=20074
export NUM_NODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=4
export NUM_GPUS_PER_NODE=4
export NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))
export ARCH=$(uname -m)
export SLURM_CPUS_PER_TASK=160
export SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CXX=g++-13
export CC=gcc-13
export NCCL_NCHANNELS_PER_PEER=4
export NCCL_MIN_NCHANNELS=4
export NCCL_NET=IB
export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NUMEXPR_MAX_THREADS=256
export WANDB_DIR=$TMPDIR
export WANDB_CACHE_DIR=$TMPDIR
export WANDB_INIT_TIMEOUT=120
export TORCHINDUCTOR_CACHE_DIR=/fast/dmartinez/oellm-ldm-work/hybrid_env/cache/inductor
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TRITON_CACHE_DIR=/fast/dmartinez/oellm-ldm-work/hybrid_env/cache/triton
export TRITON_LIBCUDA_PATH=/.singularity.d/libs
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=16
export TORCHINDUCTOR_COMPILE_THREADS=1
export WATCHDOG_EXCLUDE_FILE=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/mn5_exclude_nodes.txt
export WATCHDOG_SCRIPT=scripts/korbi/hybrid_speed/job_watchdog.py
export WATCHDOG_PYSPY=/fast/dmartinez/oellm-ldm-work/hybrid_env/py-spy
export WATCHDOG_STARTUP_S=1500
export WATCHDOG_STALL_S=600
export WATCHDOG_INTERVAL_S=15

# WAIT_FOR_JOB_NAME (optional): start only once no job of that name runs any more -- a run that
# takes over another one's checkpoints (e.g. 32 -> 128 nodes) must not load before the other's
# final save is written
if [ -n "$WAIT_FOR_JOB_NAME" ]; then
    while squeue -u "$USER" -h -n "$WAIT_FOR_JOB_NAME" -t R,CG -o %i | grep -q .; do
        echo "waiting for $WAIT_FOR_JOB_NAME to end"; sleep 30
    done
fi

# --- Watchdog: one per node, on the host, next to the training step ---------------
# GPU memory trace + hang kill (no iteration for WATCHDOG_STALL_S, or none within
# WATCHDOG_STARTUP_S): py-spy stacks of every rank, then scancel.
# scripts/korbi/hybrid_speed/job_watchdog.py; both steps use --overlap.
echo "Launching watchdog on $SLURM_NNODES node(s)"
srun --ntasks-per-node=1 --nodes=$SLURM_NNODES --overlap --cpu-bind=none python3 $WATCHDOG_SCRIPT &
WATCHDOG_PID=$!
# -------------------------------------------------------------------------------

echo "Launching job o1_ckptoff_e_db_ldm8_m4"
srun --overlap bash -c 'module load singularity/4.1.5 ; \
export HF_HOME='$HF_HOME' ; \
 export WANDB_MODE='$WANDB_MODE' ; \
 export MACHINE_NAME='$MACHINE_NAME' ; \
 export HF_ALLOW_CODE_EVAL='$HF_ALLOW_CODE_EVAL' ; \
 export HF_DATASETS_OFFLINE='$HF_DATASETS_OFFLINE' ; \
 export TRANSFORMERS_OFFLINE='$TRANSFORMERS_OFFLINE' ; \
 export MASTER_ADDR='$MASTER_ADDR' ; \
 export MASTER_PORT='$MASTER_PORT' ; \
 export NUM_NODES='$NUM_NODES' ; \
 export GPUS_PER_NODE='$GPUS_PER_NODE' ; \
 export NUM_GPUS_PER_NODE='$NUM_GPUS_PER_NODE' ; \
 export NUM_GPUS='$NUM_GPUS' ; \
 export ARCH='$ARCH' ; \
 export SLURM_CPUS_PER_TASK='$SLURM_CPUS_PER_TASK' ; \
 export SLURM_ARRAY_TASK_ID='$SLURM_ARRAY_TASK_ID' ; \
 export CUDA_DEVICE_MAX_CONNECTIONS='$CUDA_DEVICE_MAX_CONNECTIONS' ; \
 export CXX='$CXX' ; \
 export CC='$CC' ; \
 export NCCL_NCHANNELS_PER_PEER='$NCCL_NCHANNELS_PER_PEER' ; \
 export NCCL_MIN_NCHANNELS='$NCCL_MIN_NCHANNELS' ; \
 export NCCL_NET='$NCCL_NET' ; \
 export NCCL_SOCKET_IFNAME='$NCCL_SOCKET_IFNAME' ; \
 export NCCL_IB_HCA='$NCCL_IB_HCA' ; \
 export NCCL_NVLS_ENABLE='$NCCL_NVLS_ENABLE' ; \
 export NCCL_IB_DISABLE='$NCCL_IB_DISABLE' ; \
 export NCCL_DEBUG='$NCCL_DEBUG' ; \
 export NUMEXPR_MAX_THREADS='$NUMEXPR_MAX_THREADS' ; \
 export WANDB_DIR='$WANDB_DIR' ; \
 export WANDB_CACHE_DIR='$WANDB_CACHE_DIR' ; \
 export WANDB_INIT_TIMEOUT='$WANDB_INIT_TIMEOUT' ; \
 export TORCHINDUCTOR_CACHE_DIR='$TORCHINDUCTOR_CACHE_DIR' ; \
 export TORCHINDUCTOR_FX_GRAPH_CACHE='$TORCHINDUCTOR_FX_GRAPH_CACHE' ; \
 export TRITON_CACHE_DIR='$TRITON_CACHE_DIR' ; \
 export TRITON_LIBCUDA_PATH='$TRITON_LIBCUDA_PATH' ; \
 export HF_HUB_OFFLINE='$HF_HUB_OFFLINE' ; \
 export OMP_NUM_THREADS='$OMP_NUM_THREADS' ; \
 export TORCHINDUCTOR_COMPILE_THREADS='$TORCHINDUCTOR_COMPILE_THREADS' ; \
 export WATCHDOG_EXCLUDE_FILE='$WATCHDOG_EXCLUDE_FILE' ; \
 export WATCHDOG_SCRIPT='$WATCHDOG_SCRIPT' ; \
 export WATCHDOG_PYSPY='$WATCHDOG_PYSPY' ; \
 export WATCHDOG_STARTUP_S='$WATCHDOG_STARTUP_S' ; \
 export WATCHDOG_STALL_S='$WATCHDOG_STALL_S' ; \
 export WATCHDOG_INTERVAL_S='$WATCHDOG_INTERVAL_S' ; \
 \
export LOCAL_ADDR=$(nslookup $(hostname) | grep "Address: " | sed "s/Address: //" | tail) ; singularity exec --env HF_HOME=$HF_HOME \
 --env WANDB_MODE=$WANDB_MODE \
 --env MACHINE_NAME=$MACHINE_NAME \
 --env HF_ALLOW_CODE_EVAL=$HF_ALLOW_CODE_EVAL \
 --env HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE \
 --env TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE \
 --env MASTER_ADDR=$MASTER_ADDR \
 --env MASTER_PORT=$MASTER_PORT \
 --env NUM_NODES=$NUM_NODES \
 --env GPUS_PER_NODE=$GPUS_PER_NODE \
 --env NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE \
 --env NUM_GPUS=$NUM_GPUS \
 --env ARCH=$ARCH \
 --env SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK \
 --env SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID \
 --env CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS \
 --env CXX=$CXX \
 --env CC=$CC \
 --env NCCL_NCHANNELS_PER_PEER=$NCCL_NCHANNELS_PER_PEER \
 --env NCCL_MIN_NCHANNELS=$NCCL_MIN_NCHANNELS \
 --env NCCL_NET=$NCCL_NET \
 --env NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
 --env NCCL_IB_HCA=$NCCL_IB_HCA \
 --env NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE \
 --env NCCL_IB_DISABLE=$NCCL_IB_DISABLE \
 --env NCCL_DEBUG=$NCCL_DEBUG \
 --env NUMEXPR_MAX_THREADS=$NUMEXPR_MAX_THREADS \
 --env WANDB_DIR=$WANDB_DIR \
 --env WANDB_CACHE_DIR=$WANDB_CACHE_DIR \
 --env WANDB_INIT_TIMEOUT=$WANDB_INIT_TIMEOUT \
 --env TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR \
 --env TORCHINDUCTOR_FX_GRAPH_CACHE=$TORCHINDUCTOR_FX_GRAPH_CACHE \
 --env TRITON_CACHE_DIR=$TRITON_CACHE_DIR \
 --env TRITON_LIBCUDA_PATH=$TRITON_LIBCUDA_PATH \
 --env HF_HUB_OFFLINE=$HF_HUB_OFFLINE \
 --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
 --env TORCHINDUCTOR_COMPILE_THREADS=$TORCHINDUCTOR_COMPILE_THREADS \
 --env WATCHDOG_EXCLUDE_FILE=$WATCHDOG_EXCLUDE_FILE \
 --env WATCHDOG_SCRIPT=$WATCHDOG_SCRIPT \
 --env WATCHDOG_PYSPY=$WATCHDOG_PYSPY \
 --env WATCHDOG_STARTUP_S=$WATCHDOG_STARTUP_S \
 --env WATCHDOG_STALL_S=$WATCHDOG_STALL_S \
 --env WATCHDOG_INTERVAL_S=$WATCHDOG_INTERVAL_S \
 --env PYTHONPATH=/fast/dmartinez/oellm-ldm-work/hybrid_env/pylibs_moonep2:.:submodules/Megatron-LM:flash-linear-attention:submodules/cute_kernels:/fast/dmartinez/oellm-ldm-work/hybrid_env/retention:/fast/dmartinez/oellm-ldm-work/hybrid_env/pylibs_ep \
 --env RUN_DIR=./submodules/Megatron-LM \
 --env PROJECT_DIR=. \
 --env HF_HOME= \
 --env CUDA_DEVICE_MAX_CONNECTIONS=8 \
 --env NVIDIA_PYTORCH_VERSION=25.12 \
 --env TOKENIZERS_PARALLELISM=false \
 --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
 --env FLA_TENSOR_CACHE_SIZE=64 \
 --env RECOMPUTE_OFFLOAD=1 \
 --env MHC_FUSED_READ=1 \
 --env MEM_BREAKDOWN=1 \
 --env LIGER_WGRAD_ADDMM=1 \
 --env MEGATRON_LAZY_SINGLETON_GROUPS=1 \
 --nv --bind /gpfs \
 /fast/dmartinez/oellm-ldm-work/hybrid_env/container/MegatronTraining-hybrid-ep_x86_64_202610161948.sif bash /fast/dmartinez/oellm-ldm-work/render/hybrid_speed/o1_ckptoff/o1_ckptoff_e_db_ldm8_m4/task_${SLURM_ARRAY_TASK_ID}.sh'
RC=$?

kill $WATCHDOG_PID 2>/dev/null; wait $WATCHDOG_PID 2>/dev/null || true
echo "[SLURM] JOB_ID:   $SLURM_JOB_ID"
echo "[SLURM] NODELIST: $SLURM_NODELIST"
echo "[SLURM] NNODES:   $SLURM_NNODES"
exit $RC
