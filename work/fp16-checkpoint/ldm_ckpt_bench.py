"""Time and memory of CuteRecomputeWorkspace's forward + recompute reads + backward (Megatron's
order, BF16 stream) at the v4b LDM shape (105 edges, G16 K8 D128, 2 layers per recompute group),
per storage / checkpoint config. N tokens per micro-batch (copies scale linearly with N)."""
import sys
import time

import torch

import os; sys.path.insert(0, os.path.join(os.environ.get("CUTE_DIR", "/fast/pnazari/ldm-work/cute_kernels"), "tests"))
from test_ldm_fp16_checkpoint import run  # noqa: E402
import test_ldm_fp16_checkpoint as t  # noqa: E402
from cute_ldm import numerics  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
t.L, t.N = 105, N
t.BAND = set()
CONFIGS = {
    "bf16": dict(storage="bfloat16"),
    "fp16 no scaling": dict(storage="float16"),
    "fp16 eager scaling": dict(storage="float16", grad_scaling=True),
    "fp16": dict(storage="float16", grad_scaling=True),
    "fp16+ckpt32": dict(storage="float16", grad_scaling=True, interval=32),
    "fp16+ckpt16": dict(storage="float16", grad_scaling=True, interval=16),
    "fp16+ckpt8": dict(storage="float16", grad_scaling=True, interval=8),
    "fp16+ckpt16 prefetch": dict(storage="float16", grad_scaling=True, interval=16, prefetch=True),
    "fp16+ckpt8 prefetch": dict(storage="float16", grad_scaling=True, interval=8, prefetch=True),
    "fp16+ckpt16 gpu": dict(storage="float16", grad_scaling=True, interval=16, offload=False),
}
steps, cot = numerics.make_steps("nominal", t.L, N, 16, 8, 128, seed=0)
steps = [(k, q, v.to(torch.bfloat16), s, lam) for k, q, v, s, lam in steps]
cot = [(c.float() * 1e-6).to(torch.bfloat16) for c in cot]
print(f"N {N} tokens, {t.L} edges; times per micro-batch (forward + recompute + backward of the workspace)")
import os
for name, config in CONFIGS.items():
    os.environ["CUTE_LDM_EAGER_SCALING"] = "1" if "eager" in name else "0"
    ws = None
    times = []
    for i in range(4):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        got = run(steps, cot, workspace=ws, **({} if ws is not None else config))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        ws = got["workspace"]
        del got
    pinned = 0 if ws.checkpoint_m is None or not ws.checkpoint_offload else ws.checkpoint_m.numel() * 2
    print(f"{name:16s} {min(times[1:]) * 1e3:8.1f} ms  (first {times[0]:.1f} s)  peak GPU "
          f"{torch.cuda.max_memory_allocated() / 2**30:6.2f} GiB  pinned host {pinned / 2**30:5.2f} GiB  "
          f"nonfinite {int(ws.grad_nonfinite)}", flush=True)
    del ws
    torch.cuda.empty_cache()
