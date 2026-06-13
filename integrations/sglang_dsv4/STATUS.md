# SGLang DSV4 Integration Status

Date: 2026-06-13

## Current Capability

The hook can patch SGLang's `DeepseekV4AttnBackend.forward()` at process
startup and route DSV4 decode tensors into the CSAHCA CUDA extension.

There is also an experimental native SGLang source patch. It inserts a small
env-gated branch inside `DeepseekV4AttnBackend.forward()` before the FlashMLA
call, so SGLang's normal decode CUDA graph capture records the CSAHCA DSV4 op
instead of relying on a monkey-patched replacement wrapper.

Implemented CSAHCA bridge paths:

- `compress_ratio=0`: SWA paged FP8 cache through `dsv4_swa_decode_forward`.
- `compress_ratio=4`: SWA + C4 extra pages through `dsv4_sparse_decode_forward`.
- `compress_ratio=128`: SWA + C128 extra pages through `dsv4_sparse_decode_forward`.

The hook supports:

- ABI tracing with `[CSAHCA][DSV4] ABI` log lines;
- NVTX ranges named `CSAHCA_DSV4/...` when `CSAHCA_DSV4_NVTX=1`;
- live tensor comparison against FlashMLA;
- decode-output replacement for controlled smoke tests.

The native path supports:

- `CSAHCA_DSV4_NATIVE=1`: enable the in-backend CSAHCA branch;
- `CSAHCA_DSV4_NATIVE_STRICT=1`: fail instead of silently delegating when the
  native path is miswired;
- `CSAHCA_SGLANG_DSV4_PATCH=0`: keep the hook disabled so native graph evidence
  is not mixed with monkey-patch behavior.

## H100 Verification

Synthetic correctness passed for `compress_ratio=0`, `4`, and `128`.

Live SGLang tensor comparison passed on H100 for:

- short decode cases with C4 extra-cache tensors present;
- long prompt decode cases with C128 extra-cache tensors present;
- multiple tensor-parallel ranks.

An all-ratio decode replacement smoke test completed HTTP generation with
decode CUDA graph disabled. A small ShareGPT-style benchmark completed without
runtime errors.

Native decode CUDA graph verification on H100x4:

- `native=1 hook_patch=0` service completed decode CUDA graph capture on all TP
  ranks.
- Decode logs showed `cuda graph: True`.
- A strict native miswire failed inside the CSAHCA DSV4 op with
  `paged_k_cache_u8 must be [num_pages, bytes_per_page]`, confirming the
  capture path was reaching CSAHCA rather than delegating to FlashMLA. After
  flattening SGLang's FlashMLA cache view back to the CSAHCA 2D cache ABI,
  capture completed successfully.

Clean-tag graph-mode A/B, H100x4, `32` requests, concurrency `8`,
`input_words=1024`, `max_tokens=64`, full `2048` output tokens:

| Service | Output tok/s | Mean latency | p90 latency |
| --- | ---: | ---: | ---: |
| FlashMLA baseline | 290.02 | 1.76 s | 1.80 s |
| Native CSAHCA | 132.22 | 3.86 s | 3.96 s |

## Caveats

This is not yet a production FlashMLA replacement.

- The CUDA kernel is scalar/correctness-first and is expected to be slower than
  FlashMLA in real serving.
- The Python hook is useful for profiling and A/B validation, but graph-mode
  evidence should use the native branch with the hook disabled.
- Replacement through the hook requires care around decode CUDA graph capture;
  the guarded hook can delegate during capture and therefore cannot prove
  CSAHCA was recorded.
- Some early short-context live calls showed non-finite FlashMLA output for
  certain C4/C128 cases; compare logs should be reviewed before trusting any
  benchmark as apples-to-apples.

## Recommended Next Step

Optimize the DSV4 CSAHCA kernel and rerun clean-tag ABAB tests with:

1. baseline SGLang service;
2. native CSAHCA graph service;
3. CSAHCA hook in `csahca` live-compare mode for correctness sampling;
4. graph-disabled replacement mode only as a diagnostic.

Report output throughput, TPOT, TTFT, request latency, and live-compare
correctness side by side.
