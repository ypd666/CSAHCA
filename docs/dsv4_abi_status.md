# DSV4 ABI Status

Date: 2026-06-12

## Implemented Runtime Slices

The extension exposes two DSV4-compatible prototype entries:

```python
hybrid_attention.extension.dsv4_swa_decode_forward(...)
hybrid_attention.extension.dsv4_sparse_decode_forward(...)
```

They target SGLang DeepSeek-V4-style decode tensors:

- packed paged key cache bytes;
- 448 FP8 nope dims, 64 BF16 rope dims, and 8 scale bytes per token;
- `int32` token/page indices;
- per-query top-k lengths;
- per-head attention sinks;
- BF16/FP16 query and BF16/FP16 output.

Supported cache modes:

| Mode | Runtime Path | Extra Cache |
| --- | --- | --- |
| `compress_ratio=0` | SWA only | none |
| `compress_ratio=4` | SWA + sparse extra cache | C4 pages from `c4_sparse_page_indices` and `c4_sparse_topk_lengths` |
| `compress_ratio=128` | SWA + sparse extra cache | C128 pages from `c128_page_indices` and `c128_topk_lengths_clamp1` |

The kernels use an online softmax over the SWA tokens and optional extra-cache
tokens. This is correctness-first CUDA; it is not a tuned FlashMLA equivalent.

## Synthetic H100 Checks

Representative correctness commands:

```bash
python -m hybrid_attention.dsv4_correctness \
  --device cuda --dtype bfloat16 --num-queries 7 --heads 8 \
  --num-tokens 1024 --top-k 128 --page-size 256 --compress-ratio 0

python -m hybrid_attention.dsv4_correctness \
  --device cuda --dtype bfloat16 --num-queries 7 --heads 8 \
  --num-tokens 1024 --top-k 128 --page-size 256 --compress-ratio 4

python -m hybrid_attention.dsv4_correctness \
  --device cuda --dtype bfloat16 --num-queries 7 --heads 8 \
  --num-tokens 1024 --top-k 128 --page-size 256 --compress-ratio 128
```

Observed H100 results:

| Mode | Max Abs | Max Rel | Allclose |
| --- | ---: | ---: | --- |
| `compress_ratio=0` | 0.000031 | 0.007752 | true |
| `compress_ratio=4` | 0.000244 | 0.006849 | true |
| `compress_ratio=128` | 0.000000 | 0.013605 | true |

Representative isolated benchmark cases:

| Mode | Shape | Latency |
| --- | --- | ---: |
| C4 | `num_queries=1, heads=8, top_k=64, extra_top_k=128` | 0.3208 ms |
| C128 | `num_queries=1, heads=8, top_k=64, extra_top_k=64` | 0.2209 ms |

These numbers are useful as a sanity check only. The implementation is scalar
and low-parallelism compared with production FlashMLA.

## SGLang Live Tensor Checks

The SGLang hook can run in compare mode:

```bash
export CSAHCA_SGLANG_DSV4_PATCH=1
export CSAHCA_DSV4_MODE=csahca
export CSAHCA_DSV4_LIVE_COMPARE=1
export CSAHCA_DSV4_REPLACE_OUTPUT=0
```

H100 live comparisons against SGLang FlashMLA passed for:

- short ShareGPT-style decode with C4 layers active;
- long prompt decode with C128 active and non-empty extra indices;
- multiple tensor-parallel ranks.

Typical successful compare lines had `allclose=True`, `bad_finite=0`, and
`max_abs` in the BF16-scale range of roughly `0.0078125` to `0.03125`.

Some early short-context C4/C128 calls showed FlashMLA non-finite output while
the CSAHCA reference path remained finite. Those cases are logged as semantic
mismatches and should be investigated before treating the kernel as a production
replacement.

## Replacement Smoke

The hook also supports replacing SGLang's FlashMLA output for decode calls:

```bash
export CSAHCA_DSV4_MODE=csahca
export CSAHCA_DSV4_REPLACE_OUTPUT=1
export CSAHCA_DSV4_REPLACE_FORWARD_MODES=DECODE
```

With decode CUDA graph disabled, an all-ratio replacement smoke test completed
OpenAI-compatible HTTP generation without runtime errors. A small ShareGPT-style
benchmark completed with output throughput around `4.15 tok/s`.

This is a functional smoke result, not a speedup claim.

## Remaining Work

- Move the replacement path from a Python monkey patch into a stable SGLang
  backend integration.
- Make decode CUDA graph capture compatible.
- Replace scalar cache loads and per-query loops with a tuned tiled/warp design.
- Add explicit DSV4 scheduler/indexer benchmarks.
- Expand A/B serving tests beyond smoke workloads and report TTFT, TPOT, output
  throughput, and correctness side by side.
