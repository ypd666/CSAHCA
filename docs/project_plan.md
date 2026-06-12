# Project Plan

## Goal

Build a small but serious H100 performance project around DeepSeek-V4-inspired
hybrid compressed attention. The project should demonstrate CUDA kernel work,
profiling literacy, and measurable before/after speedups.

## Phase 0: Repo and H100 Bring-up

- Scaffold the repository.
- Transfer it to the H100 host.
- Confirm GPU, CUDA, PyTorch, compiler, and Nsight availability.
- Build the CUDA extension.
- Run correctness and a small benchmark.

Deliverable: a reproducible `python3 -m pip install -e .` setup and one CSV row
from `scripts/run_bench.sh`.

## Phase 1: Reference Workload

- Implement full decode attention in PyTorch.
- Implement simplified CSA:
  - split KV into chunks;
  - compute compressed chunk keys;
  - select top-k chunks;
  - attend selected chunk tokens.
- Implement simplified HCA:
  - use larger compressed chunk summaries;
  - attend summaries densely.
- Blend CSA/HCA outputs as a mini hybrid block.

Deliverable: correctness checks, benchmark harness, and baseline latency table.

## Phase 2: CUDA CSA v1

- Implement one-block-per-query-head naive CSA decode kernel.
- Use float accumulation for numerical stability.
- Compare with PyTorch CSA reference.
- Add NVTX ranges for profiling.

Deliverable: `cuda-csa` mode passing correctness and benchmarked against
`torch-csa`.

## Phase 3: Nsight Profiling Story

- Nsight Systems:
  - kernel launch timeline;
  - CPU/GPU overlap;
  - unwanted synchronization;
  - H2D/D2H transfers.
- Nsight Compute:
  - DRAM throughput;
  - L2 hit rate;
  - achieved occupancy;
  - warp stall reasons;
  - shared memory bank conflicts;
  - roofline position.

Deliverable: profiler screenshots or exported summaries in `profiling/` and
notes in `docs/optimization_report.md`.

## Phase 4: Kernel Optimization

Prioritize changes that Nsight proves are relevant:

- coalesced and vectorized loads;
- shared-memory tiling for K/V;
- warp-level dot-product and softmax reductions;
- reduced global-memory traffic;
- chunk index layout cleanup;
- optional persistent query/head mapping;
- optional CUDA Graphs for decode-loop launch overhead.

Deliverable: before/after table with latency, effective GB/s, and speedup.

## Phase 5: H100-Specific Experiments

Pick one or two instead of trying all at once:

- FP8 KV cache storage;
- CUTLASS/CuTe path for Tensor Core-friendly subproblems;
- TMA or async copy pipeline;
- thread-block clusters / distributed shared memory;
- Nsight Compute roofline comparison before and after FP8.

Deliverable: one focused H100 section explaining what changed, what improved,
and what remained bottlenecked.

## Phase 6: Optional Real-Inference Integration

- Wrap the kernel in a PyTorch extension API shaped like a future vLLM/SGLang
  backend:

```python
hybrid_decode_attention(q, k_cache, v_cache, chunk_indices, metadata)
```

- First integrate into a tiny V4-style attention block.
- Only later evaluate replacing a real V4-Flash attention path.

Deliverable: a small model/block forward pass calling the extension.

Current mini-block entry point:

```bash
python3 -m hybrid_attention.model_inference --backend cuda-csa-tiled --selection precomputed
```

## Initial Commands

```bash
python3 -m pip install -e .
python3 -m hybrid_attention.correctness --device cuda --require-extension
python3 -m hybrid_attention.benchmark --mode cuda-csa --device cuda --seq-len 16384 --out results/results.csv
bash scripts/profile_nsys.sh
bash scripts/profile_ncu.sh
```
