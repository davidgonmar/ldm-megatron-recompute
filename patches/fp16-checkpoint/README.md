# FP16 M and M checkpoints for LDM under full recompute

Two additions to the production path (`depth_memory_impl cute` + `depth_memory_unwind`) against
the BF16 unwinding error of section 2.9 of `HANDOFF-megatron.md`: the backward recovers M by
subtraction from a later, larger state, so a burst of large writes deep in the stack degrades
every state before it.

1. **FP16 M under BF16 training.** M, the u records, dM and the DY / DV tapes in FP16: three
   more mantissa bits for the state the backward unwinds. The kernels' FP16 path already exists
   (`cute_kernels` `a4e4da1`, made for FP16 training); here it runs under BF16 training.
2. **M checkpoints every N edges.** The forward copies its committed M every N edges; the
   backward replaces its unwound M by that copy when it reaches the boundary. The unwinding then
   never runs across more than N edges.

| Patch | Applies to | Base |
|---|---|---|
| `cute_kernels.patch` | [OpenEuroLLM/cute_kernels](https://github.com/OpenEuroLLM/cute_kernels) | `4bbf960` (`ldm-recompute`, the commit `oellm-autoexp` `feat/hybrid-v0.19` pins) |
| `megatron.patch` | [OpenEuroLLM/NVIDIA-Megatron-LM](https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM) | `abb31d2aa` (`oellm/v0.19-hybrid`, 2026-11-21) |

These bases are newer than those of `patches/` and `patches/memory-wip/`: both already contain
`patches/` (as `c756eb0` / `a6ac93843`), not the memory work.

```bash
cd cute_kernels && git checkout 4bbf960 && git apply /path/to/patches/fp16-checkpoint/cute_kernels.patch
cd ../Megatron-LM && git checkout abb31d2aa && git apply /path/to/patches/fp16-checkpoint/megatron.patch
```

Then, with the flags of the production LDM runs:

```
--depth-memory-storage-dtype float16 --depth-memory-checkpoint-interval 16
```

## Results

**Accuracy** (H100). 97 edges, G16 K8 D128, 2 layers per recompute group, BF16 stream, driven in
Megatron's order; cotangents of activation-gradient size (~1e-7, BF16). The values of edges 55,
57, ..., 71 (the production run's MoE sublayers) are scaled by the burst B. Worst edge before the
band (< 55), normwise relative error against the FP64 chain of `cute_ldm/reference.py`
(checked against the closed-form weighted ridge solution and an independent naive RLS to 1e-15):

| B | config | recomputed read | khat grad | query grad | value grad | scale grad |
|---|---|---|---|---|---|---|
| 1 | BF16 (production) | 3.3e-02 | 7.1e-03 | 2.0e-02 | 5.4e-03 | 5.6e-03 |
| 1 | BF16 + ckpt 16 | 1.3e-02 | 5.9e-03 | 7.7e-03 | 5.4e-03 | 5.6e-03 |
| 1 | FP16 | 4.1e-03 | 8.9e-04 | 2.4e-03 | 6.7e-04 | 6.9e-04 |
| 1 | **FP16 + ckpt 16** | **1.6e-03** | 7.3e-04 | **9.7e-04** | 6.7e-04 | 6.9e-04 |
| 1 | FP16 + ckpt 8 | 9.3e-04 | 7.2e-04 | 6.8e-04 | 6.7e-04 | 6.9e-04 |
| 30 | BF16 (production) | 2.2e-01 | 2.4e-02 | 1.3e-01 | 5.4e-03 | 5.3e-03 |
| 30 | BF16 + ckpt 16 | 3.7e-02 | 5.6e-03 | 3.7e-02 | 5.4e-03 | 5.3e-03 |
| 30 | FP16 | 2.8e-02 | 3.0e-03 | 1.7e-02 | 6.7e-04 | 6.5e-04 |
| 30 | **FP16 + ckpt 16** | **4.6e-03** | 7.0e-04 | **4.6e-03** | 6.7e-04 | 6.5e-04 |
| 30 | FP16 + ckpt 8 | 1.3e-03 | 6.5e-04 | 1.3e-03 | 6.7e-04 | 6.5e-04 |
| 85 | BF16 (production) | 6.3e-01 | 2.9e-02 | 3.8e-01 | 5.4e-03 | 5.2e-03 |
| 85 | BF16 + ckpt 16 | 1.0e-01 | 5.7e-03 | 1.0e-01 | 5.4e-03 | 5.2e-03 |
| 85 | FP16 | 7.9e-02 | 3.7e-03 | 4.7e-02 | 6.7e-04 | 6.5e-04 |
| 85 | **FP16 + ckpt 16** | **1.3e-02** | 7.0e-04 | **1.3e-02** | 6.7e-04 | 6.5e-04 |
| 85 | FP16 + ckpt 8 | 3.4e-03 | 6.4e-04 | 3.4e-03 | 6.7e-04 | 6.5e-04 |
| 300 | BF16 (production) | 2.2e+00 | 3.1e-02 | 1.3e+00 | 5.4e-03 | 5.2e-03 |
| 300 | **FP16 + ckpt 16** | **4.6e-02** | 7.0e-04 | **4.6e-02** | 6.7e-04 | 6.4e-04 |
| 300 | FP16 + ckpt 8 | 1.2e-02 | 6.4e-04 | 1.2e-02 | 6.7e-04 | 6.4e-04 |

Checkpoints every 32 edges equal every 16 for B >= 30 here (the worst edges sit below the
boundary at 64 in both). Not comparable one to one with section 2.9: there, the error over all
edges before the band without recompute; here, the worst single edge under recompute.

**Production shape** (H100, `work/fp16-checkpoint/prod_shape_check.py`): N = 65536 tokens,
97 edges, every launch on a tuned `sm90.json` schedule; 1024 of the tokens against their FP64
chain (tokens are independent recurrences). The same errors as above to two digits: B = 1,
recomputed read 3.3e-2 (BF16) -> 1.6e-3 (FP16 + ckpt 16); B = 85, 6.4e-1 -> 1.3e-2. Gradient
scale 2^22 for cotangents ~1e-6; no non-finite gradients.

**Time** of the workspace alone (forward, recomputed reads, backward; N = 16384, 105 edges, no
layers in between, so nothing hides the copies):

| config | ms / micro-batch |
|---|---|
| BF16 (production) | 238 |
| FP16, no gradient scaling | 263 |
| FP16 + gradient scaling (compiled) | 260 |
| FP16 + gradient scaling (eager) | 408 |
| FP16 + ckpt 32, host | 313 |
| FP16 + ckpt 16, host | 363 |
| FP16 + ckpt 8, host | 474 |
| FP16 + ckpt 16, host + prefetch | 311 |
| FP16 + ckpt 16, on the GPU | 265 |

**Memory.** FP16 has BF16's width. Host checkpoints take no GPU memory: one copy of M each,
N x G x K x D x 2 bytes (2 GiB at 65536 tokens); 6 copies at 16 edges (12 GiB pinned host per
GPU), 13 at 8. The gradient scaling returns FP32 input gradients for one edge at a time (~+1 GB
transient at 65536 tokens; Megatron casts them to BF16 right after). Prefetch adds one 2 GiB
staging buffer on the GPU.

## How it works

**FP16 under BF16 training.** Two things break when FP16 storage meets a BF16 model:

- *The records.* u travels in the BF16 stream; `put_record` cast it, rounding the FP16 u back to
  BF16. The backward would then subtract a u that differs from the one the forward committed into
  M by the BF16 rounding of the write, which cancels the gain exactly when the writes are large.
  Now a record holds u's bits (same width: a view, as the small tapes already are), and `ingest`
  reads them back as FP16.
- *The gradient tapes.* One storage type covers M and dM, DY, DV and the raw value gradient (they
  share the kernels' staging panel and MMA operand type). Unscaled activation gradients
  (~1e-7 and below) underflow FP16; FP16 training avoids that with its loss scale. Here the
  workspace scales its own backward (`grad_scaling`): the backward is linear in the read
  cotangents, so every cotangent is multiplied by a power of two and every input gradient
  divided by it (FP32 out) -- exact up to rounding. The scale maps the larger of the previous
  backward's largest |cotangent| and this backward's first one to 2^5, which leaves 2^11 of FP16's
  range for cotangents that outgrow it and for dM's sum over the edges. Scaled cotangents
  saturate instead of overflowing; non-finite input gradients are zeroed, counted
  (`grad_nonfinite`, logged as `ldm/grad_nonfinite` with `ldm/grad_scale_log2` in the
  architecture metrics) and make the next backward's scale 16x smaller. The scaling runs as two
  `torch.compile`d functions (eager: +150 ms per micro-batch above, compiled: none). Megatron
  turns it on whenever the storage is FP16 and the training is not.

**Checkpoints** (`CuteRecomputeWorkspace(checkpoint_interval=N)`). At every boundary b, a multiple
of N (and of the window, so the forward has just committed M_b, and a backward window start):

- *Forward:* after the launch that commits M_b, a copy of M on a side stream into pinned host
  memory (or a device buffer); the next launch that writes M waits for it.
- *Backward:* right after the window-end launch that brings the running state to b, M is
  overwritten by the copy (host to device on the compute stream, or a device copy from the
  prefetch buffer, which the next checkpoint is copied into an interval ahead). Recomputed reads
  at or below b then see the forward's M.

The windows, records, read kernel and P unwinding are those of `patches/` unchanged; the one
workspace covers the whole stack (no segments, no dM folding). P is not checkpointed: its FP32
unwinding is at ~1e-6. Not CUDA-graph capturable (side stream, host copies): it raises.

## Tests

H100 (MPI cluster), NGC PyTorch 26.09, CuTe DSL from the image:

- `cute_kernels` `tests/test_ldm_fp16_checkpoint.py` (new), 9 passed: the records carry FP16
  bits (a BF16-stream run equals an FP16-stream run bit for bit); the scaling is exact
  (cotangents 2^-20 smaller give gradients exactly 2^-20 smaller); unscaled FP16 loses the
  gradients; host checkpoints, with and without prefetch, equal device checkpoints bit for bit
  over two micro-batches; cotangents 2^16 above the first saturate and the next scale adapts;
  FP16 + ckpt 16 beats the production path by > 8x on the recomputed reads and the query
  gradient at B = 1 and 85; layout checks. `tests/test_ldm_numerics.py` (the existing BF16 / FP16
  bounds): all 39 pass on the patched workspace (48 passed together, job 17995736).
- Megatron: `test_cute_recompute_fp16_checkpoints_match_oracle` (new; FP16 with and without
  checkpoints, prefetch, BF16 with checkpoints; 1 and 2 layers per group; two micro-batches)
  within the CuTe tolerance of the torch oracle, the config checks, and the existing
  `test_hybrid_depth_memory_segments.py` / `test_depth_memory_cute.py`: 32 of 33 pass. The one
  failure, `test_unwind_matches_one_sequence[*-*-*--0]` / `[*E*E-*-0]` (the PyTorch unwind,
  `impl torch`, which the patch does not touch), fails the same way on the unpatched base on the
  same node, intermittently (`work/fp16-checkpoint/flaky.sh`).

## Not verified

- Training: nothing has trained with it. Whether removing the unwinding error stops the
  production divergence (section 2.9: amplifier, not trigger) needs a run, e.g. `ambrepro`.
- Inputs: synthetic (nominal regime, a scaled band), not the dumped tapes of a trained model.
- The production layout: step time, GPU memory at 65536 tokens on 64 GB H100s, and the host RAM
  of 12 GiB per GPU next to the stream offload and the saves.
- The FP16 kernel path (`a4e4da1`) was read (a storage-type parametrization of the BF16 kernels;
  vi's three-part split is exact down to FP16's 2^-24 floor, fine at the scaled magnitudes) and
  tested as above, not otherwise reviewed.

## Changes

`cute_kernels` (`cute_ldm/workspace.py`):
- `CuteDepthMemoryWorkspace(grad_scaling=)`: the scale, `_cotangent`, `_input_grads`,
  `numerics_stats`; FP16 values saturate instead of overflowing.
- `CuteRecomputeWorkspace(checkpoint_interval=, checkpoint_offload=, checkpoint_prefetch=)`:
  boundaries in `layout`, buffers in `ensure`, copy in `forward`, restore in `backward`; u stored
  bit for bit in `put_record` / `ingest`.
- `tests/test_ldm_fp16_checkpoint.py` (new); `python tests/test_ldm_fp16_checkpoint.py` prints
  the accuracy table.

`NVIDIA-Megatron-LM`:
- `transformer_config.py`: `depth_memory_storage_dtype`, `depth_memory_checkpoint_interval`,
  `depth_memory_checkpoint_offload`, `depth_memory_checkpoint_prefetch` and their checks.
- `depth_memory_state.py`: `make_workspace` passes them (grad scaling for FP16 storage without
  FP16 training).
- `depth_memory.py`: `ldm/grad_scale_log2`, `ldm/grad_nonfinite` metrics.
- `tests/unit_tests/models/test_hybrid_depth_memory_segments.py`: the new tests.

## Reproducing

`work/fp16-checkpoint/`: `run_in_env.sh` (your image and pip prefix, read-only, our checkouts
first on `PYTHONPATH`), the HTCondor submit files and the scripts they run: `cute_tests.sh`,
`bench.sh` (tests, accuracy table, timing via `ldm_ckpt_bench.py`), `megatron_tests.sh`,
`prod.sh` (`prod_shape_check.py`), `flaky.sh`. Logs of the runs: `/fast/pnazari/ldm-work/logs`
(jobs 17995109, 17995170, 17995171, 17995206, 17995620, 17995736).
