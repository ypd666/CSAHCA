# SGLang DSV4 Integration Status

Date: 2026-06-12

## Current Capability

The hook can patch SGLang's `DeepseekV4AttnBackend.forward()` at process
startup and route DSV4 decode tensors into the CSAHCA CUDA extension.

Implemented CSAHCA bridge paths:

- `compress_ratio=0`: SWA paged FP8 cache through `dsv4_swa_decode_forward`.
- `compress_ratio=4`: SWA + C4 extra pages through `dsv4_sparse_decode_forward`.
- `compress_ratio=128`: SWA + C128 extra pages through `dsv4_sparse_decode_forward`.

The hook supports:

- ABI tracing with `[CSAHCA][DSV4] ABI` log lines;
- NVTX ranges named `CSAHCA_DSV4/...` when `CSAHCA_DSV4_NVTX=1`;
- live tensor comparison against FlashMLA;
- decode-output replacement for controlled smoke tests.

## H100 Verification

Synthetic correctness passed for `compress_ratio=0`, `4`, and `128`.

Live SGLang tensor comparison passed on H100 for:

- short decode cases with C4 extra-cache tensors present;
- long prompt decode cases with C128 extra-cache tensors present;
- multiple tensor-parallel ranks.

An all-ratio decode replacement smoke test completed HTTP generation with
decode CUDA graph disabled. A small ShareGPT-style benchmark completed without
runtime errors.

## Caveats

This is not yet a production FlashMLA replacement.

- The CUDA kernel is scalar/correctness-first and is expected to be slower than
  FlashMLA in real serving.
- The Python hook is useful for profiling and A/B validation, but a durable
  serving integration should live inside SGLang's backend path.
- Replacement currently requires care around decode CUDA graph capture.
- Some early short-context live calls showed non-finite FlashMLA output for
  certain C4/C128 cases; compare logs should be reviewed before trusting any
  benchmark as apples-to-apples.

## Recommended Next Step

Run ABAB tests with:

1. baseline SGLang service;
2. CSAHCA hook in `trace` mode;
3. CSAHCA hook in `csahca` live-compare mode;
4. CSAHCA hook in decode replacement mode.

Report output throughput, TPOT, TTFT, request latency, and live-compare
correctness side by side.
