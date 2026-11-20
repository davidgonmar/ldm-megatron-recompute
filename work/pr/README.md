# LDM under full activation recompute: PR preparation

Two pull requests, one per OpenEuroLLM repository. Nothing is committed yet; the changes are on
local branches of the submodules of `/fast/dmartinez/oellm-autoexp`:

| Repo | Branch | Base |
|---|---|---|
| `OpenEuroLLM/cute_kernels` | `ldm-ustream-unwind` (local, in `submodules/cute_kernels`) | `10e0b51` (ep-fp8-speed) |
| `OpenEuroLLM/NVIDIA-Megatron-LM` | `ldm-cute-recompute` (local, in `submodules/Megatron-LM`) | `6863266d6` (oellm/v0.19-hybrid) |

The patches in this folder are regenerated from the working trees by `make_patches.sh`.

## 1. cute_kernels: `cute_ldm` — P unwound in the backward, recompute workspace

**Kernels** (`cute_ldm/kernels/`)
- `forward.py`: no saved-covariance tape. The forward no longer writes a BF16 copy of the
  pre-write P per edge (−K(K+1)/2 BF16 per group per edge of tape, 1.15 KB per token per edge at
  G 16, K 8).
- `window_end.py`: the running FP32 covariance (after the window's last edge) is staged, and the
  window's pre-write covariances are recovered edge by edge as
  `P_{j-1} = lam_j P_j + den_j kappa_j kappa_j^T` from the FP32 gain, denominator and forgetting
  tapes. The last edge's P feeds the covariance VJP; every edge's goes to an FP32 scratch for the
  interior launches; the window's start P is written back as the running P. No inverse, a
  contraction (`lam < 1`) on the carried state.
- `interior.py`: reads its edge's pre-write P from the FP32 scratch instead of the BF16 tape.
- `read.py` (new): the reads of a window's edges from the running state at a window boundary,
  `r_j = T^T q_j - sum_{i>j} (kappa_i . q_j) u_i` (from the end) or
  `r_j = B^T q_j + sum_{i<=j} (kappa_i . q_j) u_i` (from the start); one pass over M.

**Workspace** (`cute_ldm/workspace.py`)
- `CuteDepthMemoryWorkspace`: the running covariance must be FP32 (the backward unwinds from it;
  BF16 P drifts indefinite over deep stacks, which is what crashed a 1.6B/54-layer run at step
  11k); `p_scratch` replaces `saved_covariance_tape`; `recompute_reads()`.
- `CuteRecomputeWorkspace` (new): Megatron's `depth_memory_unwind` with `impl cute`. The
  per-edge tapes travel in the residual stream as records, the way mHC's streams do: the stream
  is `[x, u of each edge of a recompute group, the small tapes packed bit for bit]` (4 slots of
  h for 2 layers per group, as mHC-4), so Megatron checkpoints and offloads them with the stream.
  A recomputed group's input carries the previous group's records, which the workspace takes
  at the group's first read (`ingest`). A window's window-end launch runs in the backward of the
  group after its last edge, so its edges must lie in the group before that one or later: the
  backward windows are 2 edges at 2 layers per group (the forward keeps its 4, so it is the
  forward without recompute bit for bit). On the GPU only the windows the kernels work on, in
  one buffer shared by the forward window and the backward's 3 slots (2.3 GB at 65k tokens per
  rank); the stream is slot first ([slots, s, b, h]), so x and the records are contiguous
  blocks. A recomputed layer's read comes from the read kernel at its window's boundary; the
  backward is the unchanged window-end / interior path (M by subtraction, P by addition). Any
  chunk size.

**Tests** (in Megatron, `tests/unit_tests/transformer/test_depth_memory_cute.py`)
- `test_kernels_match_the_torch_oracle`: passes (edges 1/5/11/9, widths 48–128), FP32 covariance.
- `test_recompute_reads_match_the_torch_oracle` (new): passes.

**Numerics, measured** (scripts in `/fast/dmartinez/oellm-ldm-work/scripts/` and
`/lustre/home/dmartinez/DepthBench-DepthMixer/scripts/david/depth_theory/2026-10-25_cb841b/`,
results in `/fast/dmartinez/analysis/depth_theory/`)
- `p_unwind_accuracy.py` on the final 1.6B L54 checkpoint (109 edges, FP64 reference): the
  pre-write P from the BF16 tape has relative error 1.6e-3 (gain error up to 6 %); unwound from
  the FP32 running state by addition: 2.4e-7 at every edge (gain 7e-6).
- `m_unwind_analysis.py`: the closed-form window recovery of the removed recompute backward
  amplifies by 1.8–5.4x per window (97 % error at the first blocks); the subtraction unwind does
  not (0.58 % mean).
- `cute_gate_trained.py` (both kernel versions, same trained inputs, vs the FP32 oracle): reads,
  dvalue and dquery identical; dkhat, dscale and dforget more accurate with the unwound P
  (dforget worst 1.3e-2 -> 5.1e-3).

## 2. NVIDIA-Megatron-LM: `depth_memory_unwind` with `depth_memory_impl: cute`

- `depth_memory_state.make_workspace`: returns `CuteRecomputeWorkspace` for `impl cute` +
  `depth_memory_unwind`.
- `transformer_config.py`: `depth_memory_unwind` accepts `cute` (any `recompute_num_layers`,
  FP32 covariance required); docstrings.
- `hybrid_block.py`: the recompute groups' edges handed to the workspace (`layout`).
- `recompute.py`: `RECOMPUTE_SKIP_CHUNKS` raises with this mode (a chunk that is never
  recomputed never hands its records over).
- `hybrid_block.py`: the one-site-per-layer restriction applies to the torch unwind only.
- `tests/unit_tests/models/test_hybrid_depth_memory_segments.py::test_cute_recompute_matches_one_sequence`
  (new): dense and MoE patterns, 1 to 4 layers per chunk, two micro-batches,
  against the torch oracle and the CuTe one-sequence form.

## 3. Speed

4x B200, their production arm (`ep_ckptoff_mn5` e_db family rendered by their launcher, 32
experts, EP 4, all-to-all dispatcher, mock data, 16k, mbs 4, 4 accumulation steps = 1.05M
tokens/step; steady steps 3+; resident = `allocated` after a step on rank 0, peak = `max allocated`):

| arm | s/step | tok/s/GPU | resident / peak GB |
|---|---|---|---|
| plain (2 layers/chunk) | 12.9–13.3 | ~20,000 | 44 / 60 |
| mHC-4 (2 layers/chunk) | 17.1–17.5 | ~15,100 | 44 / 64 |
| LDM, their torch unwind (1 layer/chunk) | 18.8 | ~13,900 | 55 / 69 |
| LDM, cute, their segments path (2 layers/chunk) | 35–39 | ~7,000 | > 360 GB host |
| LDM, cute recompute, records in the stream, 2 layers/chunk | 17.04–17.08 | ~15,400 | 53.5 / 75.2 |
