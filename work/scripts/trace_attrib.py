# Attribute the GPU time of elementwise / copy kernels in one profiled step to the CPU op that
# launched them (innermost aten op, plus the enclosing autograd node or Python-level op), with
# input shapes, from a torch-profiler trace.
import gzip, json, sys, re, bisect, collections
path = sys.argv[1]
with gzip.open(path, "rt") as f:
    ev = json.load(f)["traceEvents"]
steps = sorted([e for e in ev if e.get("name", "").startswith("ProfilerStep#") and e.get("ph") == "X"], key=lambda e: e["ts"])
step = steps[1]; t0, t1 = step["ts"], step["ts"] + step["dur"]
runtime = {}
for e in ev:
    if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {}):
        runtime[e["args"]["correlation"]] = e
# the enclosing CPU ops of every runtime launch: one sweep per thread with a stack of open ops
by_thread = collections.defaultdict(list)
for e in ev:
    if e.get("cat") == "cpu_op" and e.get("ph") == "X":
        by_thread[(e["pid"], e["tid"])].append((e["ts"], 0, e))
for r in runtime.values():
    by_thread[(r["pid"], r["tid"])].append((r["ts"], 1, r))
stack_of = {}
for items in by_thread.values():
    items.sort(key=lambda x: (x[0], x[1]))
    open_ops = []
    for ts, kind, e in items:
        while open_ops and open_ops[-1]["ts"] + open_ops[-1]["dur"] < ts:
            open_ops.pop()
        if kind == 0:
            open_ops.append(e)
        else:
            stack_of[e["args"]["correlation"]] = list(open_ops)
want = re.compile(r"elementwise|copy|Cat|Fill|reduce|Memset|Memcpy DtoD", re.I)
agg = collections.defaultdict(lambda: [0.0, 0])
for e in ev:
    if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"): continue
    if not (t0 <= e["ts"] < t1) or not want.search(e["name"]): continue
    st = stack_of.get(e.get("args", {}).get("correlation"))
    if st is None:
        key = ("?", e["name"][:60])
    else:
        names = [o["name"] for o in st]
        inner = st[-1] if st else None
        dims = str(inner.get("args", {}).get("Input Dims", ""))[:90] if inner else ""
        outer = [n for n in names if not n.startswith("aten::")][-2:]
        key = (" > ".join(outer)[:80], (names[-1] if names else "?") + " " + dims)
    agg[key][0] += e["dur"]; agg[key][1] += 1
tot = sum(v[0] for v in agg.values())
print(f"step {step['name']} {step['dur']/1e6:.2f} s; elementwise/copy total {tot/1e6:.2f} s")
for (outer, inner), (d, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:40]:
    print(f"{d/1e6:6.3f} s {c:6d}x  {outer:80s} | {inner}")
