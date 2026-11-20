"""cute_ldm.numerics (OpenEuroLLM cute_kernels 4739513, unmodified) with value bursts shaped like
the ambitious LDM run's routed-expert outputs: 1x everywhere, except the edges of the MoE layers
55-71 whose output/input RMS grew from ~0.05 (W&B viability-layers, ambitious_ldm_main).
    burst_650: the profile at step 650 (just before the first blow-up)
    burst_950: the profile at step 950 (during the sustained grad-norm growth)
Prints, per regime and implementation, the worst normwise relative error against FP64 over the
edges before the burst (< 55), inside it and after it.
"""
import json
import sys

import torch

from cute_ldm import numerics

BURSTS = {
    "nominal": {},
    "burst_650": {55: 14, 57: 14, 59: 14, 61: 30, 63: 42, 65: 20, 67: 17, 69: 30, 71: 41, 73: 6},
    "burst_950": {1: 29, 55: 1.2, 57: 2, 59: 2.6, 61: 40, 63: 85, 65: 4, 67: 5, 69: 30, 71: 60,
                  73: 4, 75: 4, 83: 6},
}
_make = numerics.make_steps


def make_steps(regime, edges, n, g, k, d, seed=0, device="cuda"):
    steps, cot = _make("nominal", edges, n, g, k, d, seed=seed, device=device)
    for e, mult in BURSTS[regime].items():
        khat, query, value, scale, forgetting = steps[e]
        steps[e] = (khat, query, (value.float() * mult).to(torch.bfloat16), scale, forgetting)
    return steps, cot


numerics.make_steps = make_steps
numerics.REGIMES.update({r: {} for r in BURSTS})


def worst(vals, lo, hi):
    if isinstance(vals, dict):
        vals = {int(e): v for e, v in vals.items()}
        sel = [v for e, v in vals.items() if lo <= e < hi]
    else:
        sel = vals[lo:hi]
    return max(sel) if sel else float("nan")


def main():
    regimes = sys.argv[1:] or list(BURSTS)
    out = []
    cols = ("read", "recomputed_read", "recomputed_vs_forward") + numerics.GRAD_NAMES + ("unwound_m",)
    for regime in regimes:
        res = numerics.analyse(regime)
        out.append(res)
        print(f"\n## {regime}", flush=True)
        for span_name, lo, hi in (("edges <55", 0, 55), ("edges 55-73", 55, 74), ("edges >73", 74, 97)):
            print(f"\n{span_name}\n| impl | " + " | ".join(cols) + " |")
            for name, impl in res["impl"].items():
                cells = [f"{worst(impl[c], lo, hi):.1e}" if c in impl and len(impl[c]) else "" for c in cols]
                print(f"| {name} | " + " | ".join(cells) + " |")
        dg = res["diagnostics"]
        print("M rms (last)", dg["m_rms"][-1], "read rms (last)", dg["read_rms"][-1], flush=True)
        torch.cuda.empty_cache()
    with open("/fast/dmartinez/oellm-ldm-work/flagdiag/burst.json", "w") as f:
        json.dump(out, f, default=str)


if __name__ == "__main__":
    main()
