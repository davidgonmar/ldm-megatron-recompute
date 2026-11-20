# Handoff: LDM in the OpenEuroLLM Megatron stack

This note covers the work from 2026-10-26 to 2026-10-29 on running Linear Depth Memory (LDM)
inside OpenEuroLLM's Megatron hybrid stack. Nothing has been done on it since 2026-10-29.
Paths under `/fast/dmartinez` are on our cluster; ask David for copies of anything you need.

## 1. Goal and environment

### Goal

Your production layout (16k context, MoE hybrid stack, Megatron-FSDP) only fits with full
activation recompute and host checkpoint offload. Our CuTe LDM kernels came from OLMo-core, where we
never trained with whole-block recompute. Under your recompute, LDM ran either through your
segments path (our kernels, with the LDM state checkpointed at every recompute boundary) or
through your PyTorch unwind (`depth_memory_impl torch`, closed-form inverse). Both were slower than
mHC-4. The goal was to make our kernels run under your production recompute setting, as close to
mHC-4 in speed and memory as possible, and to test that on our cluster against mHC-4.

On 2026-10-29 a second question came up: why your production LDM run (`ambitious_ldm_64n`,
W&B project `openeurollm-project/flag_hybrid`) diverged.

### Environment

| | |
|---|---|
| Cluster | Our HTCondor cluster. Work and data live on the Lustre `/fast` filesystem; compile caches go on node-local disk (see 2.4). Jobs are submitted with `condor_submit_bid 100 <file>.sub`. B200 nodes have a minimum running price of 55, so a bid of 25 never starts. |
| Nodes used | 4x B200 nodes (i3xx and i4xx; runs landed on i301, i302, i305, i307, i308, i401, i402, i404, i405, i407) for every benchmark. 8x H100 80GB HBM3 nodes (i2xx) for the first unit tests and for single-GPU numerics jobs. |
| Nodes to avoid | i203, i205, i207 (H100: one GPU unreachable, see 2.4); i306, i307 (B200: i307 hangs silently). |
| Container | Apptainer 1.1.6. Image: NGC PyTorch 26.09 (`docker://nvcr.io/nvidia/pytorch:26.09-py3`), pulled as a 10.4 GB SIF at `/fast/dmartinez/oellm-ldm-work/images/pytorch_26.09-py3.sif`. Extra pip packages go into a user prefix (`/fast/dmartinez/oellm-ldm-work/pyuser`), constrained to the image's versions. |
| Python | 3.12, the image's `/usr/local/lib/python3.12`. |
| PyTorch | `2.14.0a0+b2c75dd062.nv26.09`, printed by the setup job. |
| Transformer Engine | 2.19 according to your container definition (`container/megatron/MegatronTraining-hybrid.def.in`: "NGC PyTorch (26.09: torch 2.14, TE 2.19, FA4)"). We did not print the version from our image. |
| CUDA | The 13.x line that ships with the 26.09 image; the CuTe DSL wheel is the `cu13` build. The exact CUDA and driver versions were not recorded. |
| CuTe DSL | `nvidia-cutlass-dsl[cu13]==4.7.1`, installed by `setup_env.sh` only if `import cutlass` fails in the image. Which of the two was used was not recorded. |
| Megatron | `OpenEuroLLM/NVIDIA-Megatron-LM` at `6863266d6` (branch `oellm/v0.19-hybrid`), the pin of `oellm-autoexp` branch `feat/hybrid-v0.19`. |
| cute_kernels | `OpenEuroLLM/cute_kernels` at `10e0b51` (branch `ep-fp8-speed`). |
| FLA | Your fork `OpenEuroLLM/ComplexKDA` at `ecd3df87` (the `flash-linear-attention` submodule). |
| Other pip | `transformers`, `pytest`, `compoconf`, `omegaconf`, `hydra-core`, `einops`, `opt-einsum`, `sentencepiece`, `tiktoken`, `liger-kernel`, `cuda-python`, `nvidia-resiliency-ext==0.6.0` (no deps), and `oellm-autoexp` 0.2.1 (editable). |

Your production run `ambitious_ldm_64n` used Megatron `a6ac93843` and cute_kernels `c756eb0`.
`cute_ldm` in `c756eb0` is file for file our published `cute_kernels.patch` applied on `10e0b51`.
On the Megatron side, the run's commit differs from ours only by the nGPT/LERP option, which is
off in that run.

## 2. What was tried, in order

### 2.1 Getting the code (2026-10-26)

```bash
cd /fast/dmartinez
git clone -b feat/hybrid-v0.19 git@github.com:OpenEuroLLM/oellm-autoexp.git oellm-autoexp
cd oellm-autoexp && git submodule update --init
```

All seven submodules checked out at the pinned commits. Your LDM material is in
`docs/hybrid_moe_3b_campaign.md` sections 12.3-12.5 (branch `feat/hybrid-v0.19`) and
`docs/ep_fp8_speed.md` section 6d (branch `feat/ep-fp8-speed`).

Reading your configs confirmed that every LDM quality run (loss screens and retrieval
evaluations) used our CuTe kernels (`depth_memory_impl: cute`). Your PyTorch unwind was used only
in the three speed arms of `config/experiments/korbi/hybrid_speed/ep_ckptoff_mn5.yaml`.

### 2.2 Building the environment

The image build runs as a Condor job (`/fast/dmartinez/oellm-ldm-work/scripts/setup_env.sub`:
8 CPUs, 32 GB, `request_disk = 100GB`). The script as later published:

```bash
WORK=/fast/dmartinez/oellm-ldm-work OELLM_REPO=/fast/dmartinez/oellm-autoexp scripts/setup_env.sh
```

It runs `apptainer pull` of the NGC image, generates a constraints file with your
`container/gen_constraints.py`, and installs the extras listed above with `pip install --user`.

- **First attempt (job 17764924): stalled.** `APPTAINER_TMPDIR` was on Lustre, and unpacking the
  roughly 45 GB image (millions of small files) sat for over 30 minutes without finishing. We
  killed it.
- **Second attempt (job 17765097): worked.** With `APPTAINER_TMPDIR` on the node's local disk,
  unpacking took 4 minutes. Packing the SIF took about 20-30 minutes, and copying it to `/fast`
  ran at about 15 MB/s. The job ended with `imports ok 2.14.0a0+b2c75dd062.nv26.09`.

### 2.3 First unit tests (one H100)

Tests are run from inside the Megatron checkout, because they import `tests.unit_tests...`:

```bash
cd submodules/Megatron-LM && torchrun --nproc-per-node 1 -m pytest -q -p no:cacheprovider <tests>
```

- **Job 17765210** failed before any test ran:
  ```
  WARNING: Overriding HOME environment variable with APPTAINERENV_HOME is not permitted
  AttributeError: 'NoneType' object has no attribute 'startswith'
  /usr/bin/python3: No module named pytest
  ```
  Fixes:
  - pass `--home` to `apptainer exec` instead of setting `HOME` in `--env`;
  - our import check used `megatron.__file__`, which is `None` for the namespace package, so it
    now uses `megatron.core.__file__`;
  - install `pytest` into the user prefix.
- **Job 17765214** failed on Triton setup:
  ```
  E       AssertionError: libcuda.so cannot found!
  /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/driver.py:45: AssertionError
  ```
  Under `apptainer --nv` the driver library is mounted at `/.singularity.d/libs`, which Triton's
  `ldconfig` lookup does not see. The fix is the one your launcher uses:
  `TRITON_LIBCUDA_PATH=/.singularity.d/libs`.
- **Job 17765224**: the first version of the new mode failed its oracle test with
  `AssertionError: Tensor-likes are not close!` (`tests/unit_tests/transformer/test_depth_memory_unwind.py:109`).
  A diagnostic job (17765230) showed worst-case normalized errors of 2.5e-2 and 2.8e-2 at 97
  edges, the same as the standard kernels. `assert_close`'s elementwise tolerance was too
  strict for that. The test now bounds the normalized max error at 3e-2, and 7 of 7 cases pass
  (job 17765259).

### 2.4 First benchmarks

`bench.sh` runs your production speed arm (the `o1_ckptoff` `e_db` flags rendered by your
launcher's dry run, saved as `scripts/production_args.txt`) on one node, with these changes:
- mock data, `NullTokenizer`, vocabulary 262144;
- 32 experts instead of 112, EP 4;
- the all-to-all dispatcher instead of MoonEP;
- micro-batch 4 x 16k with 4 accumulation steps, so 1.05M tokens per step, 15 iterations.

```bash
WORK=/fast/dmartinez/oellm-ldm-work OELLM_REPO=/fast/dmartinez/oellm-autoexp scripts/bench.sh <arm> 4
# arm: plain | mhc4 | ldm_cute_segments | ldm_torch_unwind | ldm_cute_recompute
```

- **Jobs 17765254-17765258 (8x H100)** died at CUDA initialization on i205 and i207:
  ```
  Unable to determine the device handle for GPU6: 0000:C3:00.0: Unknown Error
  RuntimeError: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero.
  ```
  Each of these nodes has one unreachable GPU. We excluded them and moved the arms to 4x B200
  nodes. The same error came back on 2026-10-29 on i203, i205 and i207.
- **Jobs 17765266-17765270** all died at model build, because your FLA fork imports
  `transformers` and the NGC image does not ship it:
  ```
  File "/fast/dmartinez/oellm-autoexp/flash-linear-attention/fla/layers/attn.py", line 16, in <module>
    from transformers.utils import logging
  ModuleNotFoundError: No module named 'transformers'
  ```
  Fix: `pip install --user -c constraints.txt transformers`. After that, the training entry point
  and every module it pulls in (including the CKDA layer) imported cleanly.
- **Jobs 17765391-17765395** hung in the first step for about an hour, with the GPUs at 0%. Every
  rank was compiling kernels into caches on `/fast` and sat in Lustre I/O waits. Fix:
  `run_in_env.sh` puts the Triton, Inductor, CUDA, CuTe and pycache caches on node-local disk
  (`SCRATCH_DIR`, default `/tmp/$USER-oellm`). The relaunch (jobs 17765419-17765423) compiled
  normally. The first step takes about 5-7 minutes and the second about 3.
- **The segments arm was held by Condor:**
  ```
  Job has gone over memory limit of 368640 megabytes. Peak usage: 17031 megabytes.
  ```
  That design offloads the full LDM state at every recompute boundary and needs more than 360 GB
  of host memory per node. With 1 TB requested (job 17765450) it ran.

Baselines on 4x B200, 2 layers per recompute group, steady state from step 3:

| arm | s/step | GPU memory, rank 0 (allocated / peak / peak reserved, GB) |
|---|---|---|
| plain | 12.9-13.0 | 44.1 / 60.2 / 62.6 |
| mHC-4 | 16.9-17.1 | 44.2 / 64.1 / 66.9 |
| LDM, your PyTorch unwind (1 layer per group) | 18.8 | 55.4 / 69.3 / 71.7 |
| LDM, our kernels with your segments | 35-39 | 58.7 / 90.3 / - |

### 2.5 Making our kernels work under full recompute

The basic constraint: to unwind the memory M exactly, the backward needs one d-wide vector per
edge (u, the part of M that the write overwrites), plus small per-edge values. We do not use your
closed-form inverse. It divides by lambda/den at every edge, and on our trained 1.6B checkpoint it
amplified errors by 1.8-5.4x per window. The variants we tried, in order:

1. **u carried in the checkpointed stream slot instead of the read r, window size 1.** Works only
   with 1 layer per recompute group. It had a PyTorch read (`einsum` on `M.float()`) and a PyTorch
   P unwind on the hot path. 20.3-21.2 s/step at 1 layer per group, 18.5-18.7 at 2. Dropped.
2. **Kernel changes, kept in every later version.** P is unwound by addition from the FP32 running
   covariance inside the window-end kernel, which removes the saved-P tape, and a new read kernel
   serves the reads of recomputed layers. On the inputs of our trained 1.6B model (109 edges),
   against FP64, the pre-write P error dropped from 1.6e-3 to 2.4e-7 (mean).
3. **Per-edge tapes in our own store, offloaded to the host.** 17.4 s/step, but about 180 GB of
   extra host copies per step on top of Megatron's own offload.
4. **Tapes resident on the GPU.** 14.9 s/step but 91 GB allocated and 110 GB peak, which does not
   fit.
5. **Records in the stream (the published design).** The residual stream becomes
   `[x, u of each edge in the recompute group, the other per-edge values]`, so Megatron
   checkpoints and offloads the records like mHC's streams. At 2 layers per group that is 4
   slots, the same width as mHC-4.
   - First version: 18.9-19.4 s/step at 59 / 81 GB. The records were interleaved per token, so
     every copy was strided, about 1.75 s/step of elementwise work.
   - Slot-major layout `[slots, s, b, h]`: 17.07 s/step.
   - Backward work buffers sized to the backward window, and the forward window and backward
     slots sharing one buffer: 53.5 / 75.2 GB.

Errors hit while building variant 5. All are fixed and covered by the model test:

```
RuntimeError: cute_ldm recompute: edge 5 recomputed with the state at edge boundary 6, not at its window 4-7
```
In the test pattern every sublayer is its own recompute group, so a window's last edge is re-run
and backpropagated before the earlier edges of that window.

```
RuntimeError: cute_ldm recompute: edge 95 recomputed with the state at edge boundary 97, not at its window 92-96
```
A 2-layer group whose edges straddle a window boundary, re-run while the state is past both
windows.

```
RuntimeError: cute_ldm recompute: read of edge 7 before its step
```
Megatron's recompute of a group starts with the group's first read, before any write step.

### 2.6 The published version (this repository, 2026-10-26)

The patches in `patches/` were generated from the working trees at that point (see 2.7). At 2
layers per recompute group:
- **Speed:** 17.0 s/step (16.99-17.08), against 16.9-17.1 for mHC-4.
- **GPU memory:** 53.5 / 75.2 / 77.5 GB.
- **Host traffic:** 188 GB of host copies per step, the same as mHC-4.
- **Tests:** 14 of 14 pass (`scripts/test.sh`: the kernel tests and 8 full-model cases, dense and
  MoE, 1 to 4 layers per group, two micro-batches).
- **Forward:** bit for bit the forward without recompute.

### 2.7 Memory work after publishing (uncommitted, not in this repository)

The target was mHC-4's peak. At hidden 2048 with G16 K8 D128, M is 8x hidden per token; M and its
gradient together are about 4.2-4.3 GB at 65k tokens per GPU, and no design can drop them.

- **Per-phase buffers**, with the backward reading its tapes in place from the checkpointed
  stream. Allocated memory between steps came down to 44.4 GB, the same as mHC-4.
  - 2 layers per group: 17.0 s/step, 74.4 / 77.0 GB peak allocated / reserved.
  - 1 layer per group: 21.2 s/step, 65.6 / 68.5 GB.
- **Two leak fixes**, 14 of 14 tests: 17.07 s/step and 73.7 GB peak allocated (job 17768764). That
  is still 9.5 GB above mHC-4, of which only about 4.2 GB is M and its gradient.
- **Allocator-trace runs** to locate the rest (LDM job 17768806, mHC-4 job 17768807) both died
  immediately:
  ```
  [rank0]:     assert self.lr_warmup_steps < self.lr_decay_steps
  [rank0]: AssertionError
  ```
  They ran with `ITERS=4`, while the production flags carry `--lr-warmup-iters 5`. They were not
  rerun; `ITERS=6` fixes it.

This code is now in `patches/memory-wip/` in this repository, exported from the uncommitted
working trees in our checkout:
- `patches/memory-wip/megatron.patch`: Megatron-LM branch `ldm-cute-recompute`, base `6863266d6`;
- `patches/memory-wip/cute_kernels.patch`: cute_kernels branch `ldm-ustream-unwind`, base
  `10e0b51`, including the new `cute_ldm/kernels/read.py`.

Both apply with `git apply` on the same bases as `patches/`, and they supersede `patches/`. This
is the code state that benchmark job 17768764 ran (17.07 s/step, 73.7 GB peak allocated). The
working trees themselves are still uncommitted in `/fast/dmartinez/oellm-autoexp/submodules/`.

### 2.8 Residual growth under MuOwn (2026-10-27)

This was your question about whether LDM shows the large residual growth that mHC-4 showed with
MuOwn. We set up your `screen5resgqa_s1` rematch recipe (rendered by your launcher):
- model: hidden 768, 24 layers, 144 experts;
- optimizer: `muown_classical`, lr 4e-3, 64 x 8k batch, 4000 steps;
- data: FineWeb-Edu converted to Megatron format;
- added: a per-layer residual RMS metric.

The plain and mHC-4 runs (jobs 17772399, 17772400) died at optimizer setup:

```
ImportError: emerging-optimizers package is required for optimizer='muown_classical'. Install it with: pip install emerging-optimizers
```

Your container installs `git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.2.0` with
`--no-deps`; our environment does not have it. The test was cancelled before it was installed, so
it never ran.

From the code: LDM's read enters only the sublayer input, never the residual, so LDM's residual is
structurally the same as plain's.

### 2.9 Why `ambitious_ldm_64n` diverged (2026-10-29)

We used only what your W&B project exposes through its public API, plus our own GPU jobs. Nothing
of yours was modified.

**What the logs show:**
- **Steps 0-650:** LDM tracked mHC and was slightly ahead (loss 1.739 vs 1.755 at step 650).
- **From step ~350:** in MoE sublayers 55-71 only, the routed experts' output/input RMS grew from
  about 0.05 to 0.7-2.4 by step 650. The expert weight row norms moved less than 5%, and the
  routers stayed healthy.
- **Steps 655-684:** grad-norm went 0.3 -> 7,114 and loss went 1.74 -> 2.78. It then partly
  recovered.
- **Steps ~750-1038:** sublayers 63 and 71 regrew, grad-norm climbed with spikes up to 22,233, and
  loss stalled at 1.73-1.78 while mHC reached 1.63-1.66.

**No exactly matched run exists.** The mHC-4 run on the same recipe differs in several ways:
- MHA instead of GQA;
- 128 nodes instead of 64;
- mixer LR x0.25 instead of x1;
- an older Megatron commit.

That run stayed stable. The production mHC run's streams also grew, from 1.1 to 12 by step 2500,
while the small screens stayed flat. So the activation growth comes with the production setting
and is not specific to LDM.

**A real defect in our recompute path.** Under recompute, the backward rebuilds M by subtraction
in BF16. The absolute error is about 3 * 2^-9 * max|M|. When mid-depth writes burst, max|M| grows,
and every layer before the burst loses precision in proportion. On our kernels at your production
shape (97 edges, G16 K8 D128, 2 layers per group), against FP64, worst error over layers 1-54:

| profile | forward reads | recomputed reads | query gradient | key gradient | value / scale / forgetting gradients |
|---|---|---|---|---|---|
| no burst | 0.5% | 3.3% | 2.0% | 0.7% | <= 0.9% |
| burst as at step 650 | 0.5% | 20% | 12% | 2.3% | <= 0.5% |
| burst as at step 950 | 0.7% | 33% | 4.6% | 1.4% | <= 0.6% |

A sweep over burst size B (the writes of edges 55, 57, ..., 71 scaled by B) shows the error
growing linearly:
- unwound M: 0.9% at B=1, 15% at B=100;
- recomputed reads: 39% at B=100.

The forward, the P unwind and the key, value, scale and forgetting gradients stay at the floor.
Nothing went NaN or Inf. Separately from the kernels, the exact FP64 gradients for keys, queries,
write scales and forgetting also grow in proportion to B, because they scale with the write
residual u. Under a global clip at 1, they then shrink every other parameter's update.

We do not know whether this defect caused the divergence. The expert growth started at step ~350,
before any gradient anomaly, so the trigger is upstream of it.

**Fixes evaluated, not implemented.** These were emulated per edge, with recompute, against FP64.
Each cell is the error over the edges before the burst, worst / mean:

| burst B | BF16 (now) | BF16 + M checkpoint every 8 groups | FP16 M | FP16 M + checkpoint every 8 groups |
|---|---|---|---|---|
| 1 | 2.5% / 1.1% | 1.0% / 0.52% | 0.31% / 0.14% | 0.13% / 0.07% |
| 85 (about the run's burst) | 47% / 26% | 12% / 1.7% | 5.9% / 3.3% | 1.5% / 0.22% |
| 1000 | 560% / 310% | 140% / 15% | 70% / 39% | 17% / 1.9% |

The emulation runs about 1.5-2x worse than the real kernels in absolute terms, but the ratios
between columns held on the kernels. The checkpoint-every-8-groups column was also checked on the
real BF16 kernels (job 17790205).

## 3. What works now and what is still broken

**Works:**
- Our cluster can build and run your stack: the SIF plus a user prefix, `run_in_env.sh` with
  node-local caches and `TRITON_LIBCUDA_PATH`.
- The published patch: our CuTe kernels under full uniform recompute, at any number of layers per
  group, 17.0 s/step against mHC-4's 16.9-17.1 on the 4x B200 proxy. Tests pass. Your
  `a6ac93843` and `c756eb0` already contain it.
- The P unwind in the kernel (FP32, by addition) is accurate.

**Still broken or unverified:**
- **Numerics:** the BF16 M unwind under recompute loses precision when mid-depth writes burst.
  That reaches 20-33% read error at your run's burst. Not fixed.
- **Peak memory:** about 9.5 GB above mHC-4 at 2 layers per group (73.7 vs 64.1 GB). About 5 GB of
  that is not accounted for. The allocator traces never ran.
- **Fit on your GPUs:** never measured on 64 GB H100s with 112 experts.
- **Memory work:** the post-publish memory work (2.7) is in `patches/memory-wip/` (bases
  `6863266d6` and `10e0b51`). It supersedes `patches/` and is the state benchmarked as job
  17768764. It is not committed to any branch of your repositories.
- **Limits:** no pipeline parallelism, `depth_memory_sites all` only, FP32 running covariance
  required, no `RECOMPUTE_SKIP_CHUNKS`. Each raises an error.
- **Residual growth:** the MuOwn question (2.8) is unanswered.

## 4. Open hypotheses and next steps

**Hypotheses:**
1. **Amplifier:** the BF16 M unwind error under recompute amplifies the instability once
   activations burst, but does not start it.
2. **Trigger:** the expert activation growth in production (mHC showed it too) is the trigger, and
   is independent of the residual scheme.
3. **Confounds:** GQA and the mixer LR (x1 for LDM, x0.25 for mHC) confound the comparison
   between your LDM and mHC production runs.
4. **Memory:** the unexplained ~5 GB of peak memory is implementation overhead in our workspace,
   not inherent. Going all the way to mHC-4's memory would need K=4, which is a model change.

**Next steps:**
1. **Matched pair.** Run your reproducer `ambrepro` (8 nodes, gbs 512, the production LDM config)
   with only the residual swapped: LDM vs mHC-4. Then add LDM at mixer LR x0.25 and LDM with MHA.
2. **Spike location.** Run your `ldm_ckpt_stats.py` on the rolling checkpoints of
   `ambitious_ldm_64n` to see which parameters carry the grad-norm spikes.
3. **Kernel fix in `cute_ldm`**, in increasing cost:
   - FP16 M state: about 8x lower error, no memory or speed cost. It needs an overflow check that
     raises; FP16 overflows near a 10,000x burst.
   - Checkpoint M every 8 recompute groups: about 11.5 GB per micro-batch per GPU to the host and
     back; estimated +0.45 to +2 s/step, not measured.
   - FP32 M: about +2.1 GB per GPU at 65k tokens.

   Re-check with `flagdiag/stress.py` and `flagdiag/segtest.py`.
4. **Amplifier test.** Run LDM without recompute, or with FP32 M, on the reproducer.
5. **Memory.**
   - Rerun the allocator traces with `ITERS=6` (`MEM_TRACE` in
     `/fast/dmartinez/oellm-ldm-work/scripts/bench.sh`) and diff LDM against mHC-4 at the peak.
   - Decide whether to publish the post-publish memory work from 2.7.
   - Measure on a 64 GB H100 with 112 experts.
6. **Residual growth.** Install Emerging-Optimizers v0.2.0 (`--no-deps`, as your container does)
   and run the three arms in `/fast/dmartinez/oellm-ldm-work/growth/` (`run_arm.sh`,
   `condor/arm.sub`, `residual_table.py`).
7. **Model-side option.** RMS-normalize each group of the value before it is written, which bounds
   M. This changes the model. We started a 400M test of it in our OLMo-core codebase on
   2026-10-29; its result is not part of this note.

## 5. Pointers

In this repository:
- `README.md`: the published design, results, numerics and limits.
- `patches/cute_kernels.patch` (base `10e0b51`) and `patches/megatron.patch` (base `6863266d6`).
- `patches/memory-wip/cute_kernels.patch` and `patches/memory-wip/megatron.patch`: the same bases,
  with the memory work of 2.7. They supersede `patches/`.
- `work/`: a copy of the parts of `/fast/dmartinez/oellm-ldm-work` you need to continue, at the
  same relative paths, so `/fast/dmartinez/oellm-ldm-work/<path>` below is `work/<path>` here:
  - `work/scripts/`: every working script and Condor submit file, including `bench.sh` with
    `MEM_TRACE` and `mem_peak.py`;
  - `work/flagdiag/`: the 2026-10-29 numerics scripts, `run.sh`, `burst.json`, `stress.json`,
    `emulate_full.out` and the Condor logs in `condor/`;
  - `work/growth/`: the residual-growth scripts, `condor/` (submit files and logs) and `runs/`;
  - `work/render/`: your launcher's rendered speed arms;
  - `work/pr/`: `make_patches.sh` and the PR description `README.md`.

  The scripts use absolute `/fast/dmartinez/...` paths and our Condor submit format; change both
  on your side. Not copied:
  - `logs/`, `bench/`, `traces/` (outputs);
  - `images/`, `pyuser/`, `cache/`, `home/` and `constraints.txt` (the environment, which
    `scripts/setup_env.sh` rebuilds);
  - `cute_kernels_base/`, a copy of your cute_kernels at the base commit;
  - `growth/snap/`, `growth/render/` (your sources and launcher) and `growth/data/` (tokens);
  - `flagdiag/cute_kernels_4739513/`, which is your cute_kernels at commit `4739513`.
- `scripts/setup_env.sh`, `scripts/run_in_env.sh`, `scripts/test.sh`, `scripts/bench.sh`,
  `scripts/production_args.txt`.
- `tools/trace_summary.py`, `tools/trace_attrib.py`: split a profiled step into kernel families
  and attribute elementwise and copy kernels to the ops that launched them.
- `results/runs.md`: per-step times and memory of the published runs.

On our cluster:
- `/fast/dmartinez/oellm-autoexp`: your repository at `feat/hybrid-v0.19`, with the uncommitted
  branches `ldm-cute-recompute` (Megatron-LM) and `ldm-ustream-unwind` (cute_kernels). Changed
  files:
  - Megatron-LM: `hybrid_block.py`, `recompute.py`, `depth_memory.py`, `depth_memory_state.py`,
    `transformer_config.py` and the two test files;
  - cute_kernels: `forward.py`, `interior.py`, `window_end.py`, `schedules.py`, `workspace.py` and
    the new `read.py`.
- `/fast/dmartinez/oellm-ldm`: a partial early copy of your LDM files (`megatron-src` at
  `2971ff0e`), plus `ep_fp8_speed.feat-ep-fp8-speed.md`.
- `/fast/dmartinez/oellm-ldm-work`, the working directory:
  - `images/pytorch_26.09-py3.sif`, `pyuser/`, `constraints.txt`: the environment;
  - `scripts/`: the working versions of the scripts and every Condor submit file (`setup_env.sub`,
    `bench4b*.sub`, `cute_*.sub`, `repo_tests.sub`, `trace_*.sub`), plus `mem_peak.py`;
  - `pr/`: the patches and a PR description, identical to this repository's patches;
    `make_patches.sh` regenerates them from the working trees;
  - `render/`: your launcher's rendered speed arms;
  - `bench/`, `traces/`: benchmark outputs and profiler traces;
  - `logs/`: Condor logs named `<kind>-<arm>-<job>.{out,err,log}`, for example
    `setup_env-17765097.*`, `bench4b-plain-m4-17765419.*`,
    `bench4b-ldm_cute_unwind2-m4-17767779.*` (published version),
    `bench4b-ldm_cute_unwind2-m4-17768764.*` (leak fixes) and
    `bench4bmem-*-1776880[67].*` (the failed allocator traces);
  - `flagdiag/`: the 2026-10-29 numerics. `burst.py` with `burst.json`, `stress.py` with
    `stress.json`, `segtest.py`, `emulate_dtype.py`, `emulate_full.py` with `emulate_full.out`,
    the Condor logs in `condor/` (`burst-17789700`, `stress-17789868`, `segtest-17790205`), and a
    copy of your cute_kernels at `4739513`. `emulate_full.py` was edited after
    `emulate_full.out` was produced, to run more burst sizes; that extra run was stopped.
  - `growth/`: the residual-growth setup from 2.8. `snap/` holds frozen sources (Megatron
    `a6ac93843`, cute_kernels `c756eb0`, FLA `32f30367`). `data/` holds 3.0B FineWeb-Edu tokens in
    Megatron format, and `condor/` the logs of jobs 17772399 and 17772400.
