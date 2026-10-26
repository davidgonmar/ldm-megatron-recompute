# Linear Depth Memory under full activation recompute (Megatron, CuTe kernels)

Patches for OpenEuroLLM's Megatron hybrid stack that run Linear Depth Memory (LDM,
[Depth Memory](https://arxiv.org/abs/2610.18809), kernels from
[davidgonmar/depth-memory](https://github.com/davidgonmar/depth-memory)) with the paper's CuTe
kernels under full activation recompute with host checkpoint offload, the production layout.
Nothing of the LDM state is checkpointed and nothing is inverted: the forward is the one without
recompute, bit for bit, and the backward unwinds the state with the kernels' stable updates.

| Patch | Applies to | Base |
|---|---|---|
| `patches/cute_kernels.patch` | [OpenEuroLLM/cute_kernels](https://github.com/OpenEuroLLM/cute_kernels) | `10e0b51` (`ep-fp8-speed`) |
| `patches/megatron.patch` | [OpenEuroLLM/NVIDIA-Megatron-LM](https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM) | `6863266d6` (`oellm/v0.19-hybrid`) |

```bash
cd oellm-autoexp/submodules/cute_kernels && git checkout 10e0b51 && git apply /path/to/patches/cute_kernels.patch
cd ../Megatron-LM && git checkout 6863266d6 && git apply /path/to/patches/megatron.patch
```

Then `--depth-memory --depth-memory-impl cute --depth-memory-unwind` with
`--recompute-granularity full --recompute-method uniform` and any `--recompute-num-layers`.

## Results

One node of 4x B200, OpenEuroLLM's production speed arm (`production_args.txt`: 98 layers of
CKDA / attention / MoE, hidden 2048, 16k context, top-5 routing, Megatron-FSDP, full recompute
with checkpoint offload to the host) on mock data, with 32 experts instead of 112, EP 4 and the
all-to-all dispatcher instead of MoonEP. Micro-batch 4 x 16k (65k tokens per GPU), 4 accumulation
steps, 1.05M tokens per step. Steady steps from 3 on; memory is rank 0 after step 2.

| arm | layers per recompute group | s / step | GPU allocated / peak |
|---|---|---|---|
| plain | 2 | 12.9–13.0 | 44 / 60 GB |
| mHC-4 | 2 | 16.9–17.1 | 44 / 64 GB |
| LDM, PyTorch unwind (`impl torch`, closed-form inverse, FP32 M) | 1 | 18.8 | 55 / 69 GB |
| LDM, CuTe with state checkpoints (`depth_memory_segments`) | 2 | 35–39 | 59 / 90 GB, > 360 GB host |
| **LDM, CuTe under recompute (these patches)** | 2 | **17.0** | **53.5 / 75.2 GB** |

Per-step numbers and nodes: `results/runs.md`.

- **Speed:** at mHC-4's step time, 10 % faster than the PyTorch unwind. Of the 17.0 s, the LDM
  kernels take about 1 s and the host checkpoint offload of the 4-slot stream about 2 s of
  waiting, the same as for mHC-4's four streams.
- **GPU memory over mHC-4:** +9.4 GB allocated, +11 GB at peak, at 65k tokens per GPU (it grows
  with tokens per GPU, not with the number of experts): the state M and its gradient (2 GB
  each, BF16 [tokens, 16, 8, 128]), one backward window of work buffers (about 3 GB) and the
  window slots (2.3 GB).
- **Host:** as mHC-4. The per-edge records travel in Megatron's own checkpoints.
- Not measured: the production layout on 64 GB H100s with 112 experts.

## How it works

**What the backward needs.** For an exact, stable backward, each LDM write (edge) has to leave
behind the vector it overwrites in M, u = v̄ − Mᵀk, plus its gain κ, denominator λ + kᵀPk,
forgetting and scale, and its key and query: about 5.8 KB per token per edge at hidden 2048.
With that, M is unwound by subtraction, M ← M − κuᵀ, and P by addition from the FP32 running
covariance, P ← λP + den κκᵀ. Recovering u from M instead (u = den/λ · (v̄ − Mᵀk)) divides by
λ/den at every edge and amplifies the rounding of M across depth; that is what the PyTorch
unwind does, with M in FP32.

**Where it lives: in the stream, like mHC's streams.** The residual stream is
`[x, u of each edge of the recompute group, the other per-edge values packed bit for bit]`,
slot first (`[slots, s, b, h]`, so x and every record are contiguous blocks). At 2 layers per
group that is 4 slots, as mHC-4. Megatron checkpoints and offloads it like any stream, so a
recomputed group's input carries the records of the group before it.

- **Forward:** the kernels unchanged, with their windows of 4 edges; after each write the edge's
  record goes into the stream (`put_record`).
- **Backward:** the collapse before the LM head hands the last group's records to the workspace.
  Megatron then recomputes the groups last first; the first read of a recomputed group takes the
  previous group's records from the group's input (`ingest`). A recomputed write writes nothing;
  a recomputed read comes from the running state, which sits at a window boundary, through a new
  read kernel: from the window's end, rⱼ = Mᵀqⱼ − Σ_{i>j} (κᵢ·qⱼ) uᵢ; from its start,
  rⱼ = Bᵀqⱼ + Σ_{i≤j} (κᵢ·qⱼ) uᵢ. Then the window-end and interior kernels run as without
  recompute.
- **Backward windows:** the backward of edge e runs in the backward of the layer after it, so a
  window's window-end launch runs once the group after its last edge has been recomputed, and all
  the window's edges must lie in the group before that one or later. With 2 layers per group the
  backward windows are 2 edges (the backward never uses the forward's intermediate M, so its
  windows need not match the forward's). Measured: the LDM kernels take 0.97 s per step with
  windows of 2, against 1.17 s with 4.
- **On the GPU:** M, P, their gradients, one backward window of work buffers, and 3 window slots
  of per-edge values (2 edges each), in one buffer shared with the forward's window.

**P unwound in the kernel (also without recompute).** The window-end kernel unwinds P from the
FP32 running covariance by addition instead of reading a saved BF16 P per edge, so the forward
no longer writes that tape (1.15 KB per token per edge).

## Numerics

Measured on the inputs of a trained 1.6B, 54-layer LDM model (109 edges, 4096 held-out tokens)
against an FP64 recursion:

| quantity | error |
|---|---|
| pre-write P from the saved BF16 tape (before) | 1.6e-3 mean / 3.5e-3 worst; gain κ up to 6.1e-2 |
| pre-write P unwound by addition from the FP32 state (now) | 2.4e-7 mean / 1.1e-6 worst; gain κ up to 7.1e-6 |
| M unwound by subtraction (BF16) | 0.58 % mean, 4 % worst |
| M recovered in closed form per window (the inverse) | amplifies 1.8–5.4x per window, 97 % at the first blocks |

Gradients of both kernel versions on the same trained inputs against the FP32 oracle: reads,
dvalue and dquery unchanged; dkey, dscale and dforget closer (dforget worst 1.3e-2 to 5.1e-3).

## Requirements and limits

- Full, uniform activation recompute; any number of layers per group.
- FP32 running covariance (`depth_memory_covariance_dtype float32`). A BF16 P drifts
  indefinite over deep stacks.
- No pipeline parallelism, `depth_memory_sites all`, no `RECOMPUTE_SKIP_CHUNKS`: each raises.
- A BF16 or FP32 stream; key dimension 4, 8 or 16; Hopper or Blackwell.

## Tests

`scripts/test.sh` (in the Megatron patch): the kernels and the read kernel against the PyTorch
oracle, and the whole model under full recompute (1 to 4 layers per group, dense and MoE
patterns, two micro-batches) against the oracle and the kernels without recompute.

## Changes

`cute_kernels` (`cute_ldm/`):
- `kernels/forward.py`, `kernels/window_end.py`, `kernels/interior.py`: P unwound in the
  window-end kernel from the FP32 running covariance (no saved-P tape).
- `kernels/read.py` (new): the reads of recomputed layers from the state at a window boundary.
- `schedules.py`: the read kernel's schedule axes.
- `workspace.py`: `CuteRecomputeWorkspace` (new); the base workspace requires an FP32 covariance.

`NVIDIA-Megatron-LM`:
- `transformer/depth_memory.py`: the mixer, the stream boundary and the collapse on the record
  stream.
- `transformer/depth_memory_state.py`: `make_workspace` for `impl cute` with
  `depth_memory_unwind`; the autograd nodes of the step, the read and the stream.
- `models/hybrid/hybrid_block.py`: the recompute groups handed to the workspace.
- `transformer/transformer_config.py`: the configuration checks; `recompute.py`: the
  `RECOMPUTE_SKIP_CHUNKS` check.
- Tests: `tests/unit_tests/transformer/test_depth_memory_cute.py`,
  `tests/unit_tests/models/test_hybrid_depth_memory_segments.py`.

## Reproducing

```bash
export WORK=/path/to/workdir OELLM_REPO=/path/to/oellm-autoexp   # patched submodules
scripts/setup_env.sh              # NGC PyTorch 26.09 image + pip extras under $WORK
scripts/test.sh                   # one GPU
scripts/bench.sh ldm_cute_recompute 4    # or plain, mhc4, ldm_torch_unwind, ldm_cute_segments
PROFILE=1 scripts/bench.sh ldm_cute_recompute 4 && python3 tools/trace_summary.py $WORK/bench/ldm_cute_recompute/torch_profile/rank-0.json.gz
```

`tools/trace_summary.py` splits a profiled step into kernel families, host copies and GPU idle
time; `tools/trace_attrib.py` attributes the elementwise and copy kernels to the ops that
launched them.
