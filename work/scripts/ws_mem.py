# How much GPU memory the recompute workspace holds at the production shape.
import torch, sys
sys.path.insert(0, "/fast/dmartinez/oellm-autoexp/submodules/cute_kernels")
from cute_ldm.workspace import CuteRecomputeWorkspace
n = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
ws = CuteRecomputeWorkspace(total_edges=96, groups=16, key_dim=8, value_dim=128, span=4,
                            running_covariance_dtype="float32", offload=True)
torch.cuda.reset_peak_memory_stats()
ws.ensure(n, "cuda")
torch.cuda.synchronize()
print(f"n={n} allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB  peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
for name, v in sorted(vars(ws).items(), key=lambda kv: -(kv[1].numel()*kv[1].element_size() if isinstance(kv[1], torch.Tensor) else 0)):
    if isinstance(v, torch.Tensor) and v.is_cuda:
        print(f"  {name:20s} {tuple(v.shape)} {str(v.dtype):14s} {v.numel()*v.element_size()/2**30:.2f} GiB")
