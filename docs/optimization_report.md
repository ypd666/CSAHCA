# Optimization Report

This file is intentionally a living report. Fill it in after each profiling run.

## Environment

- GPU: NVIDIA H100 PCIe, 81559 MiB each, 5 visible devices on `h100`
- Driver: 580.82.09
- CUDA toolkit: `/usr/local/cuda-12.9`
- PyTorch env: `/mnt/Data/yangpd/envs/csahca/bin/python`
- PyTorch: 2.11.0+cu128
- Commit: record the exact `git rev-parse --short HEAD` value for each formal run

## Baseline Results

| Version | Change | Latency ms | Effective GB/s | Speedup | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| v0 | PyTorch CSA reference | 0.1646 | 12.74 | 1.00x | B=1, H=8, S=4096, D=128, chunk=64, top_k=8, BF16 |
| v1 | Naive CUDA CSA | 1.2223 | 1.72 | 0.13x | Correctness-first kernel; expected to be slower before optimization |

## Nsight Systems Findings

- The timed NVTX range `cuda-csa` is 48.027 ms for 100 iterations, matching
  the benchmark latency of roughly 0.480 ms per iteration.
- Within the `cuda-csa` range, 99.2% of GPU kernel time is spent in
  `csa_decode_forward_kernel`.
- Kernel launches are dense inside the timed region, so the immediate bottleneck
  is the custom CUDA kernel rather than CPU launch gaps or PyTorch overhead.

## Nsight Compute Findings

- `csa_decode_forward_kernel` launches with grid size `(8, 1, 1)` for the
  default `B=1, H=8` case, while the H100 has 114 SMs. Nsight Compute flags this
  as a small-grid launch.
- Compute throughput and memory throughput are both below 1%, which means the
  kernel is not saturating either math or HBM bandwidth. It is primarily limited
  by insufficient parallelism.
- Achieved occupancy is reported around 12.5%.
- Barrier stalls are visible, matching the v1 implementation's repeated
  block-wide `__syncthreads()` reductions.

## Optimization Log

### v1: Naive CUDA CSA

- One CUDA block per `(batch, head)` query.
- Online softmax in float.
- Global-memory KV reads.
- No vectorized loads yet.
- Current bottleneck hypothesis: excessive block-wide synchronization and
  scalar global-memory access dominate runtime.

### v2: Planned

- Split one `(batch, head)` workload across selected KV tiles so the grid size
  becomes `B * H * selected_tiles` instead of only `B * H`.
- Use a two-stage online-softmax design: tile kernel writes partial max/sum/value
  statistics, then a merge kernel combines tile partials into the final output.
- Compare grid size, achieved occupancy, kernel duration, and end-to-end latency
  against v1.

## Lessons

- TBD
