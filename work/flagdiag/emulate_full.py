"""The kernels' state schedule emulated per edge, for a storage dtype (bf16 / fp16) and with or
without the state checkpointed every 16 edges (N = 8 recompute groups of 2 layers).

Forward: M in FP32 inside a window of 4 edges (windows counted from each segment's start),
committed in the storage dtype at the window end; each edge's record u rounded to the storage
dtype. Backward: from the committed end state, window by window: inside a window M is unwound
in FP32 by subtracting kappa u' of the records, the window start committed in the storage dtype;
at a segment start the checkpoint (the forward's own committed M) replaces the unwound state.
Backward windows: 4 edges without recompute, 2 edges (one recompute group) with recompute.
P is exact (FP64): the kernels' FP32 P unwind is at 1e-6.

Per edge e, against FP64: the unwound state M_e (after e's write), the query gradient M_e dr_e,
and (recompute) the read a recomputed layer takes, M_e' q_e. Worst and mean over the edges
before the band (1 .. 54), and worst over all edges.
    python3 emulate_full.py
"""
import sys

import torch

sys.path.insert(0, "/fast/dmartinez/oellm-ldm-work/flagdiag/cute_kernels_4739513")
from cute_ldm import numerics  # noqa: E402

BAND = list(range(55, 72, 2))
SPAN = 4
SEG = 16


def steps_for(b, n):
    steps, cot = numerics.make_steps("nominal", 97, n, 16, 8, 128, seed=0, device="cpu")
    out = []
    for e, (khat, query, value, scale, lam) in enumerate(steps):
        if e in BAND:
            value = (value.float() * b).to(torch.bfloat16)
        out.append((khat.double(), query.double(), value.double(), scale.double(), lam.double()))
    return out, [c.double() for c in cot]


def bounds(L, seg):
    if seg is None:
        return [(0, L)]
    return [(0, 1 + seg)] + [(lo, min(lo + seg, L)) for lo in range(1 + seg, L, seg)]


def run(steps, cot, store, seg, bwd_span):
    n, g, k = steps[0][0].shape
    d = steps[0][2].shape[-1]
    L = len(steps)
    parts = bounds(L, seg)
    fwd_ends = {min(s + SPAN, hi) - 1 for lo, hi in parts for s in range(lo, hi, SPAN)}
    seg_ends = {hi - 1 for lo, hi in parts[:-1]}
    m64 = torch.zeros(n, g, k, d, dtype=torch.float64)
    p = torch.eye(k, dtype=torch.float64).expand(n, g, k, k).clone()
    m32 = torch.zeros(n, g, k, d, dtype=torch.float32)
    exact, kappas, records, saved = [], [], [], {}
    for e, (khat, query, value, scale, lam) in enumerate(steps):
        kbar = khat * scale.unsqueeze(-1)
        vbar = value * scale.unsqueeze(-1)
        pk = (p * kbar.unsqueeze(-2)).sum(-1)
        den = lam + (kbar * pk).sum(-1)
        kappa = pk / den.unsqueeze(-1)
        p = (p - kappa.unsqueeze(-1) * pk.unsqueeze(-2)) / lam.view(1, -1, 1, 1)
        m64 = m64 + kappa.unsqueeze(-1) * (vbar - (m64 * kbar.unsqueeze(-1)).sum(-2)).unsqueeze(-2)
        u = vbar.float() - (m32 * kbar.float().unsqueeze(-1)).sum(-2)
        m32 = m32 + kappa.float().unsqueeze(-1) * u.unsqueeze(-2)
        kappas.append(kappa.float())
        records.append(u.to(store))
        exact.append(m64.float())
        if e in fwd_ends:
            m32 = m32.to(store).float()
            if e in seg_ends:
                saved[e] = m32.clone()
    # backward windows: bwd_span edges, never crossing a segment start
    starts = sorted({s for lo, hi in parts for s in range(lo, hi, bwd_span)})
    windows = [(s, min(nxt, s + bwd_span)) for s, nxt in zip(starts, starts[1:] + [L])]
    state = m32  # the committed end state
    err_m, err_q, err_r = [0.0] * L, [0.0] * L, [0.0] * L
    for lo, hi in reversed(windows):
        cur = state.clone()  # M after edge hi - 1, FP32 inside the window
        for e in range(hi - 1, lo - 1, -1):
            true = exact[e].double()
            got = cur.double()
            err_m[e] = float((got - true).norm() / true.norm())
            dr = cot[e]
            dq_t = (true * dr.unsqueeze(-2)).sum(-1)
            dq_g = (got * dr.unsqueeze(-2)).sum(-1)
            err_q[e] = float((dq_g - dq_t).norm() / dq_t.norm())
            q = steps[e][1]
            r_t = (true * q.unsqueeze(-1)).sum(-2)
            r_g = (got * q.unsqueeze(-1)).sum(-2)
            err_r[e] = float((r_g - r_t).norm() / r_t.norm())
            cur = cur - kappas[e].unsqueeze(-1) * records[e].float().unsqueeze(-2)
        # the state after edge lo - 1: the checkpoint at a segment start, else committed
        state = saved[lo - 1].clone() if (lo - 1) in saved else cur.to(store).float()
    return err_m, err_q, err_r


def stats(errs, edges):
    vals = [errs[e] for e in edges]
    return max(vals), sum(vals) / len(vals)


def main():
    torch.set_num_threads(32)
    n = 128
    early = range(1, 55)
    every = range(1, 97)
    print("per-edge normwise relative error against FP64; worst / mean over edges 1-54 (before "
          "the band); worst over all edges")
    print("| B | state | checkpoints | backward | unwound M worst / mean | query grad worst / mean | "
          "recomputed read worst / mean | worst any edge (M, dq, read) |")
    # python3 emulate_full.py [B ...] [--recompute-only]
    bs = [float(a) for a in sys.argv[1:] if not a.startswith("--")] or [1, 10, 30, 85, 100, 300, 1000]
    modes = (("recompute", 2),) if "--recompute-only" in sys.argv else (("no recompute", 4), ("recompute", 2))
    for b in bs:
        steps, cot = steps_for(b, n)
        for name, store in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            for seg in (None, SEG):
                for mode, span in modes:
                    em, eq, er = run(steps, cot, store, seg, span)
                    wm, mm = stats(em, early)
                    wq, mq = stats(eq, early)
                    wr, mr = stats(er, early)
                    read = f"{wr:.1e} / {mr:.1e}" if mode == "recompute" else "-"
                    print(f"| {b:g} | {name} | {'none' if seg is None else 'N=8'} | {mode} | "
                          f"{wm:.1e} / {mm:.1e} | {wq:.1e} / {mq:.1e} | {read} | "
                          f"{max(em[e] for e in every):.1e}, {max(eq[e] for e in every):.1e}, "
                          f"{max(er[e] for e in every):.1e} |", flush=True)


if __name__ == "__main__":
    main()
