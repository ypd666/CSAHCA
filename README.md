# H100 Hybrid Compressed Attention Profiling

This repository is a portfolio-style CUDA and profiling project inspired by
DeepSeek-V4-style CSA/HCA long-context decode attention. The first target is
not to reproduce a production model end to end. The target is to build a small,
measurable workload that shows how to:

- implement PyTorch reference attention paths;
- write a naive CUDA CSA decode kernel;
- profile with `torch.profiler`, Nsight Systems, and Nsight Compute;
- use the profiler output to drive concrete kernel optimizations on H100.

## Project Scope

Initial fixed assumptions:

- GPU: 1x H100.
- Workload: decode-time single-token attention.
- Shapes: `q [batch, heads, head_dim]`, KV cache `[batch, heads, seq, head_dim]`.
- Default `head_dim`: 128.
- CSA: chunk KV cache, select top-k chunks, attend selected tokens.
- HCA: attend dense compressed chunk summaries.
- CUDA v1: a correctness-first CSA kernel.

Later phases can add FP8 KV cache, vectorized loads, shared-memory tiling,
warp-level reductions, CUDA Graphs, and optional multi-GPU KV sharding.

## Repository Layout

```text
csrc/
  bindings.cpp
  csa_attention.cu
docs/
  project_plan.md
  optimization_report.md
hybrid_attention/
  benchmark.py
  correctness.py
  extension.py
  reference.py
scripts/
  run_bench.sh
  profile_nsys.sh
  profile_ncu.sh
results/
  README.md
```

## Quickstart on H100

Use an existing CUDA-enabled PyTorch environment, then install the extension:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export CSAHCA_VENV=/mnt/Data/yangpd/envs/csahca
export PATH="${CSAHCA_VENV}/bin:${CUDA_HOME}/bin:${PATH}"
PY=${CSAHCA_VENV}/bin/python

${PY} setup.py build_ext --inplace
${PY} -m hybrid_attention.correctness --device cuda --require-extension
${PY} -m hybrid_attention.benchmark --mode torch-csa --device cuda --seq-len 16384
${PY} -m hybrid_attention.benchmark --mode cuda-csa --device cuda --seq-len 16384
${PY} -m hybrid_attention.benchmark --mode cuda-csa-tiled --device cuda --seq-len 16384
```

Run the default sweep:

```bash
bash scripts/run_bench.sh
```

Create or refresh the dedicated H100 uv environment:

```bash
bash scripts/setup_h100_uv_env.sh
```

Profile with Nsight:

```bash
bash scripts/profile_nsys.sh
bash scripts/profile_nsys_tiled.sh
bash scripts/profile_ncu.sh
bash scripts/profile_ncu_tiled.sh
```

## Portfolio Story

The intended write-up is:

1. Establish PyTorch full attention, CSA, and HCA references.
2. Measure baseline latency, effective memory bandwidth, and scaling by
   sequence length, chunk size, top-k chunks, and dtype.
3. Implement CUDA CSA v1 and verify numerical agreement with the PyTorch
   reference.
4. Use Nsight Systems to inspect launch gaps and CPU/GPU overlap.
5. Use Nsight Compute to inspect DRAM throughput, L2 hit rate, occupancy,
   warp stalls, and memory-bound vs compute-bound behavior.
6. Apply kernel optimizations and report before/after speedups.

See [docs/project_plan.md](docs/project_plan.md) for the staged plan.
