"""Error of the CuTe unwind workspace against the taped fp32 oracle, per edge, next to the error of the
standard CuTe workspace (one-edge windows, no recompute) on the same inputs: does running under
recompute add error beyond the kernels' own bf16 rounding?

  python unwind_error.py   (from submodules/Megatron-LM)
"""
import sys

import torch

sys.path.insert(0, ".")
from cute_ldm.workspace import CuteDepthMemoryWorkspace, CuteUnwindWorkspace  # noqa: E402
from megatron.core.transformer.depth_memory_state import TorchDepthMemoryWorkspace  # noqa: E402
from tests.unit_tests.transformer.test_depth_memory_unwind import _inputs  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device("cuda")


def err(a, b):
    a, b = a.float(), b.float()
    return (a - b).abs().max().item() / b.abs().max().clamp_min(1e-12).item()


def run(edges, n=64, g=4, k=8, d=32):
    inputs, grads = _inputs(edges, n, g, k, d, seed=edges)
    ref = TorchDepthMemoryWorkspace(edges, g, k, d, 1.0)
    std = CuteDepthMemoryWorkspace(total_edges=edges, groups=g, key_dim=k, value_dim=d, span=1,
                                   running_covariance_dtype="float32")
    unw = CuteUnwindWorkspace(total_edges=edges, groups=g, key_dim=k, value_dim=d)
    for w in (ref, std, unw):
        w.ensure(n, dev)
    ref_reads = [ref.forward(e, *inputs[e]) for e in range(edges)]
    std_reads = [std.forward(e, *inputs[e]) for e in range(edges)]
    slots, unw_reads = [], []
    for e in range(edges):
        if e > 0:
            unw_reads.append(unw.read(e - 1, slots[e - 1]))
        slots.append(unw.forward(e, *inputs[e]))
    unw_reads.append(unw.read(edges - 1, slots[-1]))
    rows = []
    for e in reversed(range(edges)):
        rg = ref.backward(e, grads[e])
        sg = std.backward(e, grads[e].to(torch.bfloat16))
        rec = None
        if e > 0:
            rec = unw.read(e - 1, slots[e - 1])
            unw.forward(e, *inputs[e])
        ug = unw.backward(e, grads[e].to(torch.bfloat16))
        names = ("dkey", "dquery", "dvalue", "dscale", "dforget")
        rows.append((e, err(std_reads[e], ref_reads[e]), err(unw_reads[e], ref_reads[e]),
                     None if rec is None else err(rec, ref_reads[e - 1]),
                     {nm: (err(s, r), err(u, r)) for nm, r, s, u in zip(names, rg, sg, ug)}))
    print(f"\n=== {edges} edges: max |error| / max |oracle|; std = CuTe one-edge windows without "
          f"recompute, unw = CuTe unwind")
    print(f"{'edge':>4} {'read std':>9} {'read unw':>9} {'recomp rd':>9}  " +
          "  ".join(f"{nm + ' std':>11} {nm + ' unw':>11}" for nm in rows[0][4]))
    for e, rs, ru, rr, gs in sorted(rows):
        print(f"{e:>4} {rs:9.2e} {ru:9.2e} {'-' if rr is None else f'{rr:9.2e}':>9}  " +
              "  ".join(f"{s:11.2e} {u:11.2e}" for s, u in gs.values()))


for edges in (7, 33, 97):
    run(edges)
