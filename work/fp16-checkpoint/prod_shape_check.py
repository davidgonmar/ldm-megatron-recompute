"""CuteRecomputeWorkspace at the production shape -- N = 65536 tokens (v4b: one 64k sequence per
micro-batch), 97 edges, G16 K8 D128, the tuned sm90 schedules -- with a BF16 stream, driven in
Megatron's order. Tokens are independent recurrences, so the kernels' outputs on a subset of 1024
tokens are compared with the FP64 chain of those tokens alone. Streams live on the host (the driver
keeps every group's); only the subset's reads and gradients are kept."""
import os
import sys
import time
import warnings

import torch

sys.path.insert(0, os.path.join(os.environ.get("CUTE_DIR", "/fast/pnazari/ldm-work/cute_kernels"), "tests"))
from cute_ldm import numerics, reference  # noqa: E402
from cute_ldm.workspace import CuteRecomputeWorkspace  # noqa: E402

L, N, G, K, D, SPAN, PER_GROUP = 97, 65536, 16, 8, 128, 4, 2
BAND = set(range(55, 72, 2))
BEFORE = [e for e in range(L) if e < min(BAND)]
gen = torch.Generator().manual_seed(1)
IDX = torch.cat([torch.arange(512), 512 + torch.randperm(N - 512, generator=gen)[:512]]).cuda()


def drive(steps, cot, **config):
    ws = CuteRecomputeWorkspace(total_edges=L, groups=G, key_dim=K, value_dim=D, span=SPAN,
                                running_covariance_dtype="float32", **config)
    groups = numerics.recompute_groups(L, PER_GROUP)
    ws.layout(groups, G * D)
    ws.ensure(N, "cuda")
    slots = ws.stream_slots(torch.bfloat16)
    prev = torch.zeros(slots, N, G * D, dtype=torch.bfloat16, pin_memory=True)
    streams = []
    for first, count in groups:
        stream = prev.clone().pin_memory()
        for e in range(first, first + count):
            if e > 0:
                ws.ingest(e, prev)
                ws.read(e - 1)
            ws.forward(e, *steps[e])
            ws.put_record(e, stream[1:])
        streams.append(stream)
        prev = stream
    ws.ingest(L, streams[-1])
    ws.read(L - 1)
    grads, recomputed = [None] * L, {}

    def keep(t):
        return t.reshape(N, *t.shape[1:])[IDX].float().clone() if t.dim() > 1 else t.float().clone()

    grads[L - 1] = [keep(t) for t in ws.backward(L - 1, cot[L - 1].cuda())]
    for index in range(len(groups) - 1, 0, -1):
        first, count = groups[index]
        for e in range(first, first + count):
            ws.ingest(e, streams[index - 1])
            recomputed[e - 1] = ws.read(e - 1)[IDX].float().clone()
            ws.forward(e, *steps[e])
        for e in reversed(range(first, first + count)):
            grads[e - 1] = [keep(t) for t in ws.backward(e - 1, cot[e - 1].cuda())]
    torch.cuda.synchronize()
    return recomputed, grads, ws


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    for burst in (1.0, 85.0):
        steps, cot = numerics.make_steps("nominal", L, N, G, K, D, seed=0)
        cot = [(c.float() * 1e-5).to(torch.bfloat16).cpu().pin_memory() for c in cot]  # ~1e-6, on the host
        torch.cuda.empty_cache()
        for e in BAND:
            kh, q, v, s, lam = steps[e]
            steps[e] = (kh, q, (v.float() * burst).to(torch.bfloat16), s, lam)
        sub = [(kh[IDX], q[IDX], v[IDX], s[IDX], lam) for kh, q, v, s, lam in steps]
        ref_reads, ref_grads, _ = reference.chain(sub, 1.0, torch.float64, [c.cuda()[IDX] for c in cot])
        for name, config in (("bf16", dict(storage_dtype="bfloat16")),
                             ("fp16+ckpt16", dict(storage_dtype="float16", grad_scaling=True,
                                                  checkpoint_interval=16))):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                start = time.perf_counter()
                recomputed, grads, ws = drive(steps, cot, **config)
                seconds = time.perf_counter() - start
            untuned = sorted({str(w.message).split("'")[1] for w in caught if "no tuned schedule" in str(w.message)})
            read_err = max(numerics.rel(recomputed[e], ref_reads[e]) for e in BEFORE if e in recomputed)
            out = {"recomputed_read": read_err}
            for j, gname in enumerate(numerics.GRAD_NAMES[:4]):
                out[gname] = max(numerics.rel(grads[e][j], ref_grads[e][j]) for e in BEFORE)
            finite = all(torch.isfinite(t).all() for g in grads for t in g)
            print(f"burst {burst:g} {name:12s} " + " ".join(f"{k} {v:.1e}" for k, v in out.items())
                  + f"  finite {finite}  nonfinite {int(ws.grad_nonfinite)}  scale 2^{torch.log2(ws.grad_scale).item():.0f}"
                  + f"  {seconds:.1f} s  untuned launch keys: {untuned if untuned else 'none'}", flush=True)
            del ws, recomputed, grads
            torch.cuda.empty_cache()
        del steps, cot, sub


if __name__ == "__main__":
    main()
