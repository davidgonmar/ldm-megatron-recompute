# What is live at the peak of a traced step (torch.cuda.memory snapshot with history): replay the
# allocations and frees of the trace, find the peak, and group the allocations live there by the
# first frame in our code (Megatron, cute_ldm, FLA, TE) and its caller.
import pickle, sys, collections
path = sys.argv[1]
snap = pickle.load(open(path, "rb"))
trace = snap["device_traces"][0]
OURS = ("megatron/", "cute_ldm/", "flash-linear-attention/", "fla/", "transformer_engine/", "liger_kernel/")

def owner(frames):
    picked = []
    for f in frames:
        name = f.get("filename", "")
        if any(o in name for o in OURS):
            picked.append(f"{name.split('/')[-1]}:{f.get('line')} {f.get('name')}")
            if len(picked) == 2:
                break
    return " < ".join(picked) if picked else (frames[0].get("filename", "?").split("/")[-1] + ":" + str(frames[0].get("line")) if frames else "?")

live, cur, peak, peak_i, peak_live = {}, 0, 0, -1, None
for i, e in enumerate(trace):
    a = e["action"]
    if a == "alloc":
        live[e["addr"]] = e
        cur += e["size"]
        if cur > peak:
            peak, peak_i = cur, i
            peak_live = dict(live)
    elif a in ("free_completed",):
        old = live.pop(e["addr"], None)
        if old is not None:
            cur -= old["size"]
print(f"{path}: {len(trace)} events; peak of the traced allocations {peak / 2**30:.2f} GiB at event {peak_i}")
groups = collections.defaultdict(lambda: [0, 0])
for e in peak_live.values():
    k = owner(e.get("frames", []))
    groups[k][0] += e["size"]; groups[k][1] += 1
for k, (b, c) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:40]:
    print(f"{b / 2**30:7.2f} GiB {c:5d}x  {k}")
# the stack at the peak event itself
f = trace[peak_i].get("frames", [])
print("peak allocation:", trace[peak_i]["size"] / 2**30, "GiB at", owner(f))
