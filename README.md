# CSAHCA

CSAHCA is a compact H100 CUDA profiling project for compressed sparse
attention. It starts with PyTorch reference implementations, moves into custom
CUDA decode kernels, and includes an experimental SGLang DeepSeek-V4/FlashMLA
ABI bridge for live tensor comparison.

The repository is meant to show the full performance workflow: build a
reproducible workload, profile it with Nsight Systems and Nsight Compute, make a
targeted kernel change, then validate both correctness and serving-level impact.

## Highlights

- PyTorch references for full attention, CSA, HCA, and a mini hybrid decode
  block.
- CUDA CSA decode kernels:
  - `cuda-csa`: correctness-first one-CTA-per-query/head kernel.
  - `cuda-csa-tiled`: tiled CTA parallelism that addresses H100 small-grid
    underutilization.
- Nsight Systems and Nsight Compute scripts for timeline and kernel analysis.
- Model-level A/B harness that inserts the custom kernel inside a small decode
  block instead of timing only a naked kernel.
- Experimental DSV4 ABI prototype for SGLang DeepSeek-V4 attention:
  - `compress_ratio=0`: SWA paged FP8 cache.
  - `compress_ratio=4`: SWA plus C4 compressed cache pages.
  - `compress_ratio=128`: SWA plus C128 compressed cache pages.

## Status

This is a research and portfolio prototype, not a production FlashMLA
replacement.

- The CSA tiled kernel has documented H100 speedups in the synthetic and
  mini-block setting. See [docs/model_kernel_benchmark.md](docs/model_kernel_benchmark.md).
- The DSV4 path is correctness-first. It has passed synthetic checks and live
  tensor comparison against SGLang FlashMLA on H100, including C4 and C128
  paths, but it is not optimized enough to claim end-to-end serving speedup.
- The SGLang integration is behind environment flags and is designed for A/B
  testing, live comparison, smoke replacement, and an experimental native
  decode CUDA graph path. Production use would require further kernel
  optimization and upstream-quality SGLang integration.

No model weights, datasets, profiler reports, or generated benchmark artifacts
are included in the repository.

## Current H100 Benchmark Snapshot

### One-GPU Model-Level Kernel A/B

Shape: `batch=1, heads=8, head_dim=128, seq_len=32768, chunk_size=64,
top_k=8, dtype=bfloat16`.

| Workload | Baseline | CSAHCA kernel path | Result |
| --- | ---: | ---: | ---: |
| Decode block, MLP ratio 2.0 | torch-csa 0.3311 ms/token | cuda-csa-tiled 0.1221 ms/token | 2.71x faster |
| Attention-dominated, MLP ratio 0.0 | torch-csa 0.2807 ms/token | cuda-csa-tiled 0.0742 ms/token | 3.78x faster |
| Dynamic chunk selection | torch-csa 0.3858 ms/token | cuda-csa-tiled 0.3001 ms/token | 1.29x faster |

See [docs/model_kernel_benchmark.md](docs/model_kernel_benchmark.md) for the
full command, CSV paths, and interpretation.

### SGLang DeepSeek-V4 Serving A/B

Workload: H100x4, `32` requests, concurrency `8`, generated prompt length
about `1024` words, `max_tokens=64`, OpenAI-compatible chat endpoint.

| Decode CUDA graph | FlashMLA baseline | CSAHCA path | Result |
| --- | ---: | ---: | ---: |
| Disabled | 22.65 output tok/s | 26.59 output tok/s | 1.17x faster |
| Enabled, guarded hook replacement | 57.06 output tok/s | 57.54 output tok/s | essentially parity |
| Enabled, native CSAHCA branch, clean-tag full-output run | 290.02 output tok/s | 132.22 output tok/s | 0.46x baseline |

The graph-disabled row measures the replacement path without decode CUDA graph
capture. The guarded-hook graph-enabled row is closer to normal SGLang serving
mode, but it does not prove that CSAHCA itself was captured because the hook can
delegate during CUDA stream capture. The native row patches SGLang's
`DeepseekV4AttnBackend.forward()` so the CSAHCA DSV4 op is selected before the
FlashMLA call and is captured by SGLang's decode CUDA graph. That row is the
current apples-to-apples graph-mode result for the same prompt tag and full
`max_tokens=64` outputs. It shows that the integration works, but the current
DSV4 CSAHCA kernel is still much slower than FlashMLA in graph-mode serving.

## Repository Layout

```text
csrc/                         CUDA kernels and pybind bindings
hybrid_attention/             PyTorch references, benchmarks, extension wrappers
integrations/sglang_dsv4/     Env-gated SGLang DeepSeek-V4 hook prototype
scripts/                      H100 setup, benchmark, and profiling helpers
docs/                         Project plan, profiling notes, benchmark reports
results/                      Placeholder for generated local CSV/JSON outputs
```

## Quickstart On H100

Create the uv environment and build the extension:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export CSAHCA_VENV="${HOME}/envs/csahca"
bash scripts/setup_h100_uv_env.sh
```

Or build inside an existing CUDA-enabled PyTorch environment:

```bash
python setup.py build_ext --inplace
python -m hybrid_attention.correctness --device cuda --require-extension
```

Run the synthetic CSA benchmark sweep:

```bash
bash scripts/run_bench.sh
```

Run the mini decode-block A/B benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_model_kernel_ab.sh
```

Profile with Nsight:

```bash
bash scripts/profile_nsys.sh
bash scripts/profile_nsys_tiled.sh
bash scripts/profile_ncu.sh
bash scripts/profile_ncu_tiled.sh
```

## DSV4 Prototype Checks

Build the CUDA extension first, then run synthetic checks for the three DSV4
cache modes:

```bash
python -m hybrid_attention.dsv4_correctness --device cuda --compress-ratio 0
python -m hybrid_attention.dsv4_correctness --device cuda --compress-ratio 4
python -m hybrid_attention.dsv4_correctness --device cuda --compress-ratio 128
```

Microbenchmark one DSV4 mode:

```bash
python -m hybrid_attention.dsv4_benchmark \
  --device cuda \
  --compress-ratio 128 \
  --num-queries 1 \
  --heads 8 \
  --top-k 64 \
  --extra-top-k 64
```

The current DSV4 implementation details and caveats are in
[docs/dsv4_abi_status.md](docs/dsv4_abi_status.md).

## SGLang Integration

The SGLang hook and native-branch helpers live in
[integrations/sglang_dsv4](integrations/sglang_dsv4). They are disabled by
default and are controlled by environment variables:

```bash
export CSAHCA_SGLANG_DSV4_PATCH=1
export CSAHCA_DSV4_MODE=trace       # trace, csahca, or require-kernel
export CSAHCA_DSV4_LIVE_COMPARE=1   # compare CSAHCA output with FlashMLA
```

For replacement smoke tests, use:

```bash
export CSAHCA_DSV4_MODE=csahca
export CSAHCA_DSV4_REPLACE_OUTPUT=1
export CSAHCA_DSV4_REPLACE_FORWARD_MODES=DECODE
```

Read [integrations/sglang_dsv4/README.md](integrations/sglang_dsv4/README.md)
before restarting a live service.

## Performance Story

The main optimization lesson is deliberately simple: the first CUDA CSA kernel
launched only `batch * heads` CTAs, so an H100 with 114 SMs was mostly idle.
The tiled kernel splits selected KV tokens across many CTAs and merges online
softmax partials, turning a small-grid kernel into a much more parallel decode
workload.

That optimization is useful when the selected sparse attention work dominates
or when chunk selection is already available. When dynamic selection is still
performed in Python/PyTorch every step, selector overhead can erase most of the
kernel win. The next real serving target is therefore the indexer/scheduler
path, not only the attention math kernel.
