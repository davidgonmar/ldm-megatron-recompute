# Summarise a torch-profiler trace (rank-0.json.gz): GPU time per kernel family inside the
# profiled steps, memcpy traffic, and GPU idle time.
import gzip, json, sys, re, collections
path = sys.argv[1]
with gzip.open(path, "rt") as f:
    ev = json.load(f)["traceEvents"]
steps = [e for e in ev if e.get("name", "").startswith("ProfilerStep#") and e.get("ph") == "X"]
steps.sort(key=lambda e: e["ts"])
print(f"{len(steps)} profiled steps:", [(s["name"], round(s["dur"]/1e6, 2)) for s in steps])
def family(name):
    n = name.lower()
    for pat, fam in [("memcpy htod|memcpy dtoh", "memcpy host<->device"), ("memcpy dtod|memset", "memcpy/memset device"),
                     ("cute_ldm|ldm_|depth_memory|window_end|interior|recompute_read", "cute_ldm kernels"),
                     ("gemm|cutlass|nvjet|matmul|cublas", "gemm"), ("flash|attn|fmha|attention", "attention"),
                     ("fla|chunk_|gated_delta|kda|cross_domain", "fla / kda kernels"), ("nccl", "nccl"),
                     ("moe|permute|unpermute|router|topk|sort|index|scatter|gather", "moe routing"),
                     ("norm|rms", "norms"), ("triton", "triton other"), ("elementwise|vectorized|reduce|fill|copy_|cat", "elementwise")]:
        if re.search(pat, n): return fam
    return "other"
gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
for s in steps[1:2]:   # the second profiled step (the first carries profiler warm-up)
    t0, t1 = s["ts"], s["ts"] + s["dur"]
    inside = [e for e in gpu if t0 <= e["ts"] < t1]
    by = collections.defaultdict(float); byname = collections.defaultdict(float)
    for e in inside:
        by[family(e["name"])] += e["dur"]; byname[e["name"]] += e["dur"]
    print(f"\n{s['name']}: {s['dur']/1e6:.2f} s wall")
    # union of busy intervals per stream for idle time
    streams = collections.defaultdict(list)
    for e in inside: streams[e.get("tid")].append((e["ts"], e["ts"] + e["dur"]))
    allint = sorted(i for v in streams.values() for i in v)
    busy, cur = 0.0, None
    for a, b in allint:
        if cur is None or a > cur[1]:
            if cur: busy += cur[1] - cur[0]
            cur = [a, b]
        else: cur[1] = max(cur[1], b)
    if cur: busy += cur[1] - cur[0]
    print(f"  GPU busy (any stream): {busy/1e6:.2f} s, idle {(s['dur']-busy)/1e6:.2f} s")
    # the same without the memcpy streams: time the compute kernels are running
    cint = sorted((e["ts"], e["ts"] + e["dur"]) for e in inside if e.get("cat") == "kernel")
    cbusy, cur = 0.0, None
    for a, b in cint:
        if cur is None or a > cur[1]:
            if cur: cbusy += cur[1] - cur[0]
            cur = [a, b]
        else: cur[1] = max(cur[1], b)
    if cur: cbusy += cur[1] - cur[0]
    print(f"  compute kernels busy: {cbusy/1e6:.2f} s, no compute kernel running {(s['dur']-cbusy)/1e6:.2f} s")
    for fam, d in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"  {fam:24s} {d/1e6:7.2f} s")
    print("  top kernels:")
    for name, d in sorted(byname.items(), key=lambda kv: -kv[1])[:18]:
        print(f"    {d/1e6:6.2f} s  {name[:110]}")
    mem = [e for e in inside if e.get("cat") == "gpu_memcpy"]
    bytes_by = collections.defaultdict(float)
    for e in mem:
        bytes_by[e["name"]] += e.get("args", {}).get("bytes", 0)
    for name, b in bytes_by.items(): print(f"  {name}: {b/2**30:.1f} GiB")
