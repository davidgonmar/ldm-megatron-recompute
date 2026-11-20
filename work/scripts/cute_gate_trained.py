"""Gate of the cute_ldm kernels on trained statistics: one forward and backward over the per-edge
inputs dumped from a trained checkpoint (dump_step_inputs.py), the CuTe workspace against the
FP32 PyTorch oracle (TorchDepthMemoryWorkspace). Reports, per edge, the relative error of the
read and of each input gradient (max |error| / max |oracle|), and the worst over all edges.

    python cute_gate_trained.py INPUTS.pt [--edges N]     (from submodules/Megatron-LM)
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")
from cute_ldm.workspace import CuteDepthMemoryWorkspace  # noqa: E402
from megatron.core.transformer.depth_memory_state import TorchDepthMemoryWorkspace  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("inputs")
parser.add_argument("--edges", type=int, default=None)
args = parser.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
d = torch.load(args.inputs)
L = args.edges or d["edges"]
N, G, K, D = d["tokens"], d["groups"], d["key_dim"], d["value_dim"]
print(f"{d['checkpoint']}: {L} edges, {N} tokens, G {G}, K {K}, D {D}", flush=True)
oracle = TorchDepthMemoryWorkspace(L, G, K, D, d["alpha"])
cute = CuteDepthMemoryWorkspace(total_edges=L, groups=G, key_dim=K, value_dim=D, alpha=d["alpha"], span=4)
oracle.ensure(N, torch.device("cuda"))
cute.ensure(N, "cuda")
step = lambda e: (d["khat"][e].cuda(), d["query"][e].cuda(), d["value"][e].cuda(),  # noqa: E731
                  d["scale"][e].cuda(), d["forgetting"][e].cuda())
reads_o = [oracle.forward(e, *step(e)) for e in range(L)]
reads_c = [cute.forward(e, *step(e)) for e in range(L)]
back_o = [oracle.backward(e, d["read_grad"][e].cuda()) for e in reversed(range(L))]
back_c = [cute.backward(e, d["read_grad"][e].cuda()) for e in reversed(range(L))]


def err(a, b):
    """(max |error| / max |oracle|, ||error|| / ||oracle||)"""
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item(), \
           ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


names = ("read", "dkhat", "dquery", "dvalue", "dscale", "dforget")
worst_max = {n: 0.0 for n in names}
worst_l2 = {n: 0.0 for n in names}
print("per edge: max-relative / L2-relative error against the FP32 oracle")
print(f"{'edge':>5} " + " ".join(f"{n:>17}" for n in names))
for e in range(L):
    pairs = [err(reads_c[e], reads_o[e])] + [err(c, o) for o, c in zip(back_o[L - 1 - e], back_c[L - 1 - e])]
    for n, (m, l) in zip(names, pairs):
        worst_max[n] = max(worst_max[n], m)
        worst_l2[n] = max(worst_l2[n], l)
    if e % 8 == 0 or e == L - 1:
        print(f"{e:>5} " + " ".join(f"{m:8.1e}/{l:8.1e}" for m, l in pairs))
print("worst max-relative: " + " ".join(f"{k} {v:.2e}" for k, v in worst_max.items()))
print("worst L2-relative:  " + " ".join(f"{k} {v:.2e}" for k, v in worst_l2.items()))
