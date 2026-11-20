"""The real CuTe kernels (CuteDepthMemoryWorkspace, BF16 M) with the state checkpointed every
SEG edges: one workspace per segment, the forward state carried with export_state / load_state
(the checkpoint: the forward's own BF16 M and its FP32 P), the backward's dL/dM, dL/dP with
export_grad / load_grad. Inside a segment the backward unwinds as usual. SEG = 16 edges is N = 8
recompute groups of 2 layers (edge 0, the embedding write, rides in the first segment).

Against the FP64 chain, the band burst B of stress.py; per B, for no checkpoints (one workspace)
and for SEG = 16: the unwound M at window starts before the band (the read it implies) and every
gradient, normwise over the edges before the band.
    python3 segtest.py
"""
import sys

import torch

from cute_ldm import numerics, reference
from cute_ldm.workspace import CuteDepthMemoryWorkspace

BAND = list(range(55, 72, 2))
NAMES = numerics.GRAD_NAMES
SPAN = 4


def steps_for(b, edges=97, n=512, g=16, k=8, d=128):
    steps, cot = numerics.make_steps("nominal", edges, n, g, k, d, seed=0)
    for e in BAND:
        khat, query, value, scale, lam = steps[e]
        steps[e] = (khat, query, (value.float() * b).to(torch.bfloat16), scale, lam)
    return steps, cot


def segments(L, seg):
    if seg is None:
        return [(0, L)]
    out, lo = [], 0
    hi = 1 + seg  # edge 0 plus the first N groups
    while lo < L:
        out.append((lo, min(hi, L)))
        lo, hi = hi, hi + seg
    return out


def run_segmented(steps, cot, seg):
    """Reads, gradients and the unwound M at every window start (global edge index of the window
    start s: the state after edge s - 1)."""
    L = len(steps)
    n, g, k = steps[0][0].shape
    d = steps[0][2].shape[-1]
    parts = segments(L, seg)
    wss, state, saved = [], None, {}
    for lo, hi in parts:
        ws = CuteDepthMemoryWorkspace(total_edges=hi - lo, groups=g, key_dim=k, value_dim=d,
                                      alpha=1.0, span=SPAN, running_covariance_dtype="float32")
        ws.ensure(n, "cuda")
        if state is not None:
            ws.load_state(*state)
        for e in range(lo, hi):
            ws.forward(e - lo, *steps[e])
        m, p = ws.export_state()
        state = (m.clone(), p.clone())
        saved[hi] = state[0][..., :d].double()
        wss.append(ws)
    grads, unwound, carried = [None] * L, {}, None
    for (lo, hi), ws in reversed(list(zip(parts, wss))):
        if carried is not None:
            ws.load_grad(*carried)
        for e in reversed(range(lo, hi)):
            local = e - lo
            grads[e] = [t.float().clone() for t in ws.backward(local, cot[e])]
            start = local // SPAN * SPAN
            if local == min(start + SPAN, hi - lo) - 1 and start > 0:
                unwound[lo + start] = ws.m[..., :d].double().clone()
        if lo > 0:
            dm, dp = ws.export_grad()
            carried = (dm.to(torch.bfloat16).clone(), dp.clone())
            unwound[lo] = saved[lo]  # the checkpoint: the previous segment's forward end state
    return grads, unwound


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    print("errors normwise over the edges before the band (< 55); unwound M worst over window "
          "starts < 55 (the read it implies is the same within 1 %, see stress.py)")
    print("| B | checkpoints | unwound M | " + " | ".join(NAMES) + " |")
    for b in (1, 10, 30, 100, 300, 1000):
        steps, cot = steps_for(b)
        L = len(steps)
        # the window starts of both layouts (segment windows count from each segment's start)
        ref_reads, ref_grads, ref_states = reference.chain(steps, 1.0, torch.float64, cot,
                                                           range(L))
        for name, seg in (("none", None), ("every 16 edges (N=8)", 16)):
            grads, unwound = run_segmented(steps, cot, seg)
            um = [float((m - ref_states[s - 1][0]).norm() / ref_states[s - 1][0].norm())
                  for s, m in unwound.items() if s < 55]
            errs = []
            for j in range(5):
                num = sum((grads[e][j].double() - ref_grads[e][j].double()).square().sum()
                          for e in range(1, 55)) ** 0.5
                den = sum(ref_grads[e][j].double().square().sum() for e in range(1, 55)) ** 0.5
                errs.append(float(num / den))
            print(f"| {b:g} | {name} | {max(um):.1e} | " + " | ".join(f"{x:.1e}" for x in errs)
                  + " |", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
