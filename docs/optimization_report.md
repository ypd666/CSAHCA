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

## v2 Tiled Results

Default long-context synthetic case:

- `B=1, H=8, S=32768, D=128`
- `chunk_size=64, top_k=8, tile_size=8`
- BF16 KV cache and Q

| Version | Change | Latency ms | Effective GB/s | Speedup vs v1 | Speedup vs PyTorch CSA |
| --- | --- | ---: | ---: | ---: | ---: |
| PyTorch CSA | Reference gather + attention | 0.3016 | 6.95 | 1.60x | 1.00x |
| v1 | One block per `(batch, head)` | 0.4822 | 4.35 | 1.00x | 0.63x |
| v2 | Tile selected KV across CTAs + merge partials | 0.0232 | 90.24 | 20.78x | 13.00x |

Tile-size sweep on the same shape:

| Tile size | Latency ms | Effective GB/s |
| ---: | ---: | ---: |
| 8 | 0.0237 | 88.43 |
| 16 | 0.0262 | 79.94 |
| 32 | 0.0390 | 53.83 |
| 64 | 0.0692 | 30.31 |

## Nsight Systems Findings

- The timed NVTX range `cuda-csa` is 48.027 ms for 100 iterations, matching
  the benchmark latency of roughly 0.480 ms per iteration.
- Within the `cuda-csa` range, 99.2% of GPU kernel time is spent in
  `csa_decode_forward_kernel`.
- Kernel launches are dense inside the timed region, so the immediate bottleneck
  is the custom CUDA kernel rather than CPU launch gaps or PyTorch overhead.
- For v2, the timed NVTX range `cuda-csa-tiled` is 2.748656 ms for 100
  iterations, or roughly 0.0275 ms per iteration in Nsight Systems.
- The v2 report shows two dominant kernels: `csa_decode_tile_kernel` and
  `csa_decode_merge_kernel`, each around 10.5 us per launch in the profiled run.

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

### v2: Optimization Target

- Split one `(batch, head)` workload across selected KV tiles so the grid size
  becomes `B * H * selected_tiles` instead of only `B * H`.
- Use a two-stage online-softmax design: tile kernel writes partial max/sum/value
  statistics, then a merge kernel combines tile partials into the final output.
- Compare grid size, achieved occupancy, kernel duration, and end-to-end latency
  against v1.

### v2: Tiled CTA Parallelism

- `csa_decode_tile_kernel`: one CTA handles one selected KV tile for one
  `(batch, head)` pair.
- `csa_decode_merge_kernel`: combines per-tile online-softmax statistics into
  the final decode attention output.
- For the default case, grid size increases from `B * H = 8` CTAs to
  `B * H * ceil(top_k * chunk_size / tile_size) = 512` CTAs in the tile kernel.
- This directly targets the Nsight Compute "Small Grid" finding on H100.

## Lessons

- TBD
