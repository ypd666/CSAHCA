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

- TBD

## Nsight Compute Findings

- TBD

## Optimization Log

### v1: Naive CUDA CSA

- One CUDA block per `(batch, head)` query.
- Online softmax in float.
- Global-memory KV reads.
- No vectorized loads yet.
- Current bottleneck hypothesis: excessive block-wide synchronization and
  scalar global-memory access dominate runtime.

### v2: Planned

- Use warp-level reductions instead of block-wide shared-memory reductions.
- Improve memory access and reduce synchronization.
- Compare stall reasons and achieved bandwidth.

## Lessons

- TBD
