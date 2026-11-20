"""Residual size per layer over training from the runs' tensorboard logs.
    python3 residual_table.py [STEP ...]
Prints, per arm and step: RMS of the residual after layers 1, 6, 12, 18, 24 (main stream; mHC: also
the mean of the other streams; LDM: also the pending read), the input of the final norm, and the
training loss.
"""
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = "/fast/dmartinez/oellm-ldm-work/growth/runs"
LAYERS = (1, 6, 12, 18, 24)


def load(arm):
    acc = EventAccumulator(os.path.join(RUNS, arm, "tb"), size_guidance={"scalars": 0})
    acc.Reload()
    return {tag: {e.step: e.value for e in acc.Scalars(tag)} for tag in acc.Tags()["scalars"]}


def main():
    want = [int(s) for s in sys.argv[1:]]
    for arm in ("plain", "mhc4", "ldm"):
        if not glob.glob(os.path.join(RUNS, arm, "tb", "events.*")):
            continue
        s = load(arm)
        main_tag = "arch/residual/stream_rms/main/layer_{}"
        steps = sorted(s.get(main_tag.format(24), {}))
        if want:
            steps = [t for t in steps if t in want]
        loss = s.get("lm loss", {})
        print(f"\n{arm}")
        head = "step " + " ".join(f"L{l:>2} main" for l in LAYERS)
        if arm != "plain":
            head += "  " + " ".join(f"L{l:>2} {'others' if arm == 'mhc4' else 'read'}" for l in LAYERS)
        print(head + "   exit   loss")
        for t in steps:
            row = f"{t:>4} " + " ".join(f"{s[main_tag.format(l)][t]:>8.3g}" for l in LAYERS)
            if arm != "plain":
                row += "  " + " ".join(
                    f"{s['arch/residual/stream_rms/others/layer_%d' % l][t]:>9.3g}" for l in LAYERS
                )
            ex = s.get("arch/residual_exit/stream_rms/main/layer_1", {}).get(t)
            ls = loss.get(t)
            row += f" {ex:>7.3g}" if ex is not None else "       -"
            row += f" {ls:>6.3f}" if ls is not None else "      -"
            print(row)


if __name__ == "__main__":
    main()
