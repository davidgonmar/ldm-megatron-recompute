"""The kernels' state schedule emulated in PyTorch for a storage dtype X (bfloat16 / float16):
M committed in X at every window end (FP32 inside a window), each edge's record u rounded to X,
and the backward's unwind by subtraction (FP32 inside a window, committed in X at each window
start). P is exact (FP64): its FP32 unwind is at 1e-6. Against the FP64 chain, the unwound M at
every window start (relative error before the band) and the read it implies, for the band burst
B of stress.py.
    python3 emulate_dtype.py
"""
import sys

import torch

sys.path.insert(0, "/fast/dmartinez/oellm-ldm-work/flagdiag/cute_kernels_4739513")
from cute_ldm import numerics  # noqa: E402

BAND = list(range(55, 72, 2))
SPAN = 4


def steps_for(b, n):
    steps, _ = numerics.make_steps("nominal", 97, n, 16, 8, 128, seed=0, device="cpu")
    out = []
    for e, (khat, query, value, scale, lam) in enumerate(steps):
        if e in BAND:
            value = (value.float() * b).to(torch.bfloat16)
        out.append((khat.double(), query.double(), value.double(), scale.double(), lam.double()))
    return out


def window_ends(L, seg):
    """Window ends of span SPAN counted from each segment's first edge (the segmented
    workspaces' windows); seg None: one segment."""
    bounds = [(0, L)] if seg is None else [(0, 1 + seg)] + [
        (lo, min(lo + seg, L)) for lo in range(1 + seg, L, seg)]
    ends = []
    for lo, hi in bounds:
        ends += [min(s + SPAN, hi) - 1 for s in range(lo, hi, SPAN)]
    return ends, {hi - 1 for lo, hi in bounds[:-1]}


def run(steps, store, seg=None):
    n, g, k = steps[0][0].shape
    d = steps[0][2].shape[-1]
    m64 = torch.zeros(n, g, k, d, dtype=torch.float64)
    p = torch.eye(k, dtype=torch.float64).expand(n, g, k, k).clone()
    mc = torch.zeros(n, g, k, d, dtype=store)
    m32 = mc.float()
    exact, kappas, records, overflow, saved = {}, [], [], False, {}
    L = len(steps)
    commits, restores = window_ends(L, seg)
    for e, (khat, query, value, scale, lam) in enumerate(steps):
        kbar = khat * scale.unsqueeze(-1)
        vbar = value * scale.unsqueeze(-1)
        pk = (p * kbar.unsqueeze(-2)).sum(-1)
        den = lam + (kbar * pk).sum(-1)
        kappa = pk / den.unsqueeze(-1)
        p = (p - kappa.unsqueeze(-1) * pk.unsqueeze(-2)) / lam.view(1, -1, 1, 1)
        m64 = m64 + kappa.unsqueeze(-1) * (vbar - (m64 * kbar.unsqueeze(-1)).sum(-2)).unsqueeze(-2)
        u = vbar.float() - (m32 * kbar.float().unsqueeze(-1)).sum(-2)
        rec = u.to(store)
        overflow |= not bool(torch.isfinite(rec).all())
        m32 = m32 + kappa.float().unsqueeze(-1) * u.unsqueeze(-2)
        kappas.append(kappa.float())
        records.append(rec)
        if e in commits:
            mc = m32.to(store)
            overflow |= not bool(torch.isfinite(mc).all())
            m32 = mc.float()
            exact[e] = m64.clone()
            if e in restores:
                saved[e] = mc.clone()  # the checkpoint: the forward's own committed M
    # the backward: unwind window by window from the committed end state
    unwound = {}
    ends = sorted(exact)
    for i in range(len(ends) - 1, 0, -1):
        hi, lo = ends[i], ends[i - 1]
        m32 = mc.float()
        for e in range(hi, lo, -1):
            m32 = m32 - kappas[e].unsqueeze(-1) * records[e].float().unsqueeze(-2)
        mc = saved[lo] if lo in saved else m32.to(store)
        unwound[lo] = mc.double()
    rel = {e: float((unwound[e] - exact[e]).norm() / exact[e].norm()) for e in unwound}
    implied = {}
    for e in unwound:
        q = steps[e][1]
        r_true = (exact[e] * q.unsqueeze(-1)).sum(-2)
        r_unw = (unwound[e] * q.unsqueeze(-1)).sum(-2)
        implied[e] = float((r_unw - r_true).norm() / r_true.norm())
    mmax = float(max(x.abs().max() for x in exact.values()))
    umax = float(max(r.float().abs().max() for r in records))
    return rel, implied, overflow, mmax, umax


def main():
    torch.set_num_threads(16)
    n = 128
    print("worst over window starts before the band (edges < 55); max |M|, max |u| over the chain")
    print("| B | dtype | checkpoints | unwound M, edges < 48 | unwound M, edges 48-54 (the band's segment) | max abs M | max abs u | overflow |")
    for b in (1, 10, 30, 100, 300, 1000):
        steps = steps_for(b, n)
        for name, store, seg in (("bf16", torch.bfloat16, None), ("bf16", torch.bfloat16, 16),
                                 ("fp16", torch.float16, None), ("fp16", torch.float16, 16)):
            rel, implied, overflow, mmax, umax = run(steps, store, seg)
            before = [e for e in rel if e < 48]
            near = [e for e in rel if 48 <= e < 55]
            print(f"| {b:g} | {name} | {'none' if seg is None else 'N=8'} | {max(rel[e] for e in before):.1e} | "
                  f"{max(rel[e] for e in near):.1e} | {mmax:.3g} | {umax:.3g} | {overflow} |",
                  flush=True)


if __name__ == "__main__":
    main()
