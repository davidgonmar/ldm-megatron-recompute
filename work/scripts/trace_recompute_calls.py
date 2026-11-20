"""Trace the order of workspace calls (forward, recomputed forward, backward) of one forward and
backward of the small hybrid model under depth_memory_unwind with the CuTe kernels, to see how
Megatron's chunked recompute drives the state.  Run from submodules/Megatron-LM."""
import sys

import torch

sys.path.insert(0, ".")
from tests.unit_tests.models.test_hybrid_depth_memory_segments import B, S, VOCAB, Utils, _model  # noqa: E402
from cute_ldm import workspace as W  # noqa: E402

pattern = sys.argv[1] if len(sys.argv) > 1 else "*-*-*-"
chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 1
orig_forward, orig_backward = W.CuteRecomputeWorkspace.forward, W.CuteRecomputeWorkspace.backward


def forward(self, edge, *a):
    kind = "forward" if (edge == 0 or self._phase == "forward") else "RECOMPUTE"
    print(f"{kind:9s} edge {edge:2d}  phase {self._phase:8s} next {self._next:2d} boundary {self._boundary:2d}", flush=True)
    try:
        return orig_forward(self, edge, *a)
    except Exception as e:
        print("   ->", e, flush=True)
        raise


def backward(self, edge, g):
    print(f"backward  edge {edge:2d}  phase {self._phase:8s} boundary {self._boundary:2d}", flush=True)
    return orig_backward(self, edge, g)


W.CuteRecomputeWorkspace.forward, W.CuteRecomputeWorkspace.backward = forward, backward
Utils.initialize_model_parallel(1, 1)
kw = dict(num_moe_experts=4, moe_router_topk=2, moe_ffn_hidden_size=64) if "E" in pattern else {}
model = _model(pattern, "cute", depth_memory_unwind=True, depth_memory_tape_offload=False,
               recompute_granularity="full", recompute_method="uniform", recompute_num_layers=chunk, **kw)
ws = model.decoder.ldm_workspace if hasattr(model, "decoder") else None
print("pattern", pattern, "chunk", chunk, "edges", getattr(ws, "total_edges", "?"), "windows", getattr(ws, "windows", "?"))
ids = torch.randint(0, VOCAB, (B, S), device="cuda")
pos = torch.arange(S, device="cuda").unsqueeze(0).expand(B, -1)
out = model(ids, position_ids=pos, attention_mask=None)
print("--- backward", flush=True)
out.float().square().mean().backward()
print("done")
