"""Which part of the CuTe LDM kernels fails under the ambitious run's value burst, and how badly.

Inputs: cute_ldm.numerics' nominal production regime (97 edges, G16 K8 D128, window 4, 2 edges
per recompute group, 512 tokens), with the values of the band's edges (55, 57, ..., 71: the MoE
sublayers whose expert outputs grew in ambitious_ldm_main) multiplied by B. Per B, against the
FP64 chain (cute_ldm.reference):

  true      the FP64 gradients' norms over all edges (does the math itself blow up?)
  cute      CuteDepthMemoryWorkspace (no recompute): gradient errors; the backward's unwound M
  recompute CuteRecomputeWorkspace in Megatron's order: gradient errors; the recomputed reads
  implied   the read the unwound M of `cute` gives at each window start, M_unwound' q: if the
            recomputed reads are off by the same amount, the unwind is the failing stage and
            the read kernel only passes its error on

Errors are normwise over the edges before the band (< 55), where the unwind has to subtract the
band's writes, and over all edges (what the global gradient norm sees).
    python3 stress.py [B ...]
"""
import json
import sys

import torch

from cute_ldm import numerics, reference

BAND = list(range(55, 72, 2))
NAMES = numerics.GRAD_NAMES


def steps_for(b, edges=97, n=512, g=16, k=8, d=128):
    steps, cot = numerics.make_steps("nominal", edges, n, g, k, d, seed=0)
    for e in BAND:
        khat, query, value, scale, lam = steps[e]
        steps[e] = (khat, query, (value.float() * b).to(torch.bfloat16), scale, lam)
    return steps, cot


def norm(ts):
    return float(sum(t.double().square().sum() for t in ts) ** 0.5)


def err(got, ref, edges):
    return norm([got[e] - ref[e] for e in edges]) / max(norm([ref[e] for e in edges]), 1e-300)


def finite(ts):
    return all(bool(torch.isfinite(t).all()) for t in ts)


def main():
    bs = [float(x) for x in sys.argv[1:]] or [1, 10, 30, 100, 300, 1000]
    torch.backends.cuda.matmul.allow_tf32 = False
    out = []
    for b in bs:
        steps, cot = steps_for(b)
        L = len(steps)
        early, every = range(1, 55), range(L)
        starts = list(range(4, L, 4))
        ref_reads, ref_grads, ref_states = reference.chain(steps, 1.0, torch.float64, cot,
                                                           [s - 1 for s in starts])
        cute = numerics.run_cute(steps, cot, alpha=1.0, span=4)
        rec = numerics.run_cute_recompute(steps, cot, alpha=1.0, span=4, per_group=2)
        res = dict(b=b, finite=dict(cute=finite([t for gr in cute["grads"] for t in gr]),
                                    recompute=finite([t for gr in rec["grads"] for t in gr])))
        res["true_norm"] = {n: norm([ref_grads[e][j] for e in every]) for j, n in enumerate(NAMES)}
        for tag, got in (("cute", cute), ("recompute", rec)):
            for j, n in enumerate(NAMES):
                g = [gr[j] for gr in got["grads"]]
                r = [gr[j] for gr in ref_grads]
                res[f"{tag}/{n}/early"] = err(g, r, early)
                res[f"{tag}/{n}/all"] = err(g, r, every)
                res[f"{tag}/{n}/abs_all"] = norm([g[e] - r[e] for e in every])
        # the unwound M against FP64 at each window start (state after edge s - 1)
        um_abs, um_rel, implied = {}, {}, {}
        for s, (m, _) in cute["unwound"].items():
            true_m = ref_states[s - 1][0]
            um_abs[s] = float((m.double() - true_m).abs().max())
            um_rel[s] = float((m.double() - true_m).norm() / true_m.norm())
            q = steps[s - 1][1].double()
            r_unw = (m.double() * q.unsqueeze(-1)).sum(-2)
            implied[s - 1] = float((r_unw - ref_reads[s - 1]).norm() / ref_reads[s - 1].norm())
        res["unwound_m_rel"] = um_rel
        res["unwound_m_abs"] = um_abs
        res["implied_read"] = implied
        res["recomputed_read"] = {e: float((r.double() - ref_reads[e]).norm() / ref_reads[e].norm())
                                  for e, r in rec["recomputed"].items()}
        res["forward_read"] = max(float((cute["reads"][e].double() - ref_reads[e]).norm()
                                        / ref_reads[e].norm()) for e in every)
        res["m_true_max"] = float(max(ref_states[s - 1][0].abs().max() for s in starts))
        out.append(res)

        print(f"\n=== B = {b:g}   finite: {res['finite']}   forward reads worst {res['forward_read']:.1e}"
              f"   max |M| (FP64) {res['m_true_max']:.3g}", flush=True)
        print("true FP64 gradient norms: " + "  ".join(f"{n} {res['true_norm'][n]:.3g}" for n in NAMES))
        print("| gradient | cute early | cute all | recompute early | recompute all | recompute abs err |")
        for n in NAMES:
            print(f"| {n} | {res[f'cute/{n}/early']:.1e} | {res[f'cute/{n}/all']:.1e} | "
                  f"{res[f'recompute/{n}/early']:.1e} | {res[f'recompute/{n}/all']:.1e} | "
                  f"{res[f'recompute/{n}/abs_all']:.3g} |")
        pick = [s for s in (4, 8, 16, 32, 48, 52, 56, 64, 72, 80, 92) if s in um_rel]
        print("window start:        " + " ".join(f"{s:>8d}" for s in pick))
        print("unwound M rel err:   " + " ".join(f"{um_rel[s]:8.1e}" for s in pick))
        print("unwound M abs err:   " + " ".join(f"{um_abs[s]:8.1e}" for s in pick))
        print("implied read err:    " + " ".join(f"{implied[s - 1]:8.1e}" for s in pick))
        print("recomputed read err: " + " ".join(
            f"{res['recomputed_read'].get(s - 1, float('nan')):8.1e}" for s in pick), flush=True)
        del cute, rec
        torch.cuda.empty_cache()
    with open("/fast/dmartinez/oellm-ldm-work/flagdiag/stress.json", "w") as f:
        json.dump(out, f, default=str)


if __name__ == "__main__":
    main()
