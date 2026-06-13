# SGLang DeepSeek-V4 Integration

This folder contains isolated, env-gated integration helpers for testing CSAHCA
inside the real SGLang DeepSeek-V4-Flash serving path.

The original CSAHCA CUDA kernel is not ABI-compatible with SGLang's production
DeepSeek-V4 attention path. The production path calls
`DeepseekV4AttnBackend.forward()` and then `flash_mla_with_kvcache()` with a
packed FP8 SWA cache, optional C4/C128 compressed pages, page indices, top-k
lengths, attention sinks, and FlashMLA scheduler metadata. The repo now has
experimental DSV4-specific CSAHCA entry points for that packed cache ABI.

This integration provides two paths:

- a safe Python hook for tracing, shadow comparison, and graph-disabled smoke
  replacement;
- a small native SGLang source patch that selects the CSAHCA DSV4 op inside
  `DeepseekV4AttnBackend.forward()` before the FlashMLA call, allowing SGLang's
  normal decode CUDA graph capture to record CSAHCA.

The hook supports:

- `CSAHCA_DSV4_MODE=trace`: enter the real DSV4 backend, log call shapes, and
  delegate to the original FlashMLA implementation.
- `CSAHCA_DSV4_MODE=csahca`: try the CSAHCA bridge and delegate if the current
  kernel is not compatible.
- `CSAHCA_DSV4_MODE=require-kernel`: fail fast unless a DSV4-compatible CSAHCA
  kernel is available.

Useful proof knobs:

- `CSAHCA_DSV4_TRACE_ABI=1`: log DSV4 q/cache/index tensor shapes.
- `CSAHCA_DSV4_NVTX=1`: add `CSAHCA_DSV4/...` NVTX ranges for Nsight Systems.
- `CSAHCA_DSV4_REPLACE_REQUIRE_FINITE_Q=1`: keep FlashMLA output when `q`
  contains NaN/Inf during replacement runs.
- `CSAHCA_DSV4_REPLACE_MAX_ABS_Q=1000000`: keep FlashMLA output when `q`
  has finite but suspiciously huge inactive-lane values.

## Smoke Test

```bash
bash /mnt/Data/yangpd/CSAHCA/integrations/sglang_dsv4/scripts/smoke_import_hook.sh
```

Expected output includes `"patched": true`.

## Launch Patched Service

Do not run this on the same GPUs while another 4-GPU service is already using
them.

```bash
PORT=30001 \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
CSAHCA_DSV4_MODE=trace \
bash /mnt/Data/yangpd/CSAHCA/integrations/sglang_dsv4/scripts/launch_patched_dsv4.sh
```

Use `PORT=30000` only if the existing service has been stopped.

To replace the service on a port in one command:

```bash
CONFIRM_RESTART=1 PORT=30000 CSAHCA_DSV4_MODE=trace \
bash /mnt/Data/yangpd/CSAHCA/integrations/sglang_dsv4/scripts/restart_patched_on_port.sh
```

Detached background variant:

```bash
CONFIRM_RESTART=1 PORT=30000 CSAHCA_DSV4_MODE=trace \
bash /mnt/Data/yangpd/CSAHCA/integrations/sglang_dsv4/scripts/restart_patched_detached.sh
```

## Native Decode CUDA Graph Path

Install the native branch into an SGLang source checkout:

```bash
/mnt/Data/yangpd/envs/dsv4_flash/bin/python \
  /mnt/Data/yangpd/CSAHCA/integrations/sglang_dsv4/scripts/install_native_sglang_patch.py \
  --sglang-src /mnt/Data/yangpd/src/sglang-main
```

Launch with the hook disabled and the native branch enabled:

```bash
CONFIRM_RESTART=1 \
PORT=30000 \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
CSAHCA_DSV4_NATIVE=1 \
CSAHCA_DSV4_NATIVE_STRICT=1 \
CSAHCA_SGLANG_DSV4_PATCH=0 \
bash /mnt/Data/yangpd/CSAHCA/integrations/sglang_dsv4/scripts/restart_patched_detached.sh
```

Evidence for a real graph-mode run should include all of the following:

- launcher log shows `native=1 hook_patch=0`;
- SGLang log shows `Capture cuda graph end` on all tensor-parallel ranks;
- decode logs show `cuda graph: True`;
- an earlier strict native run fails inside `dsv4_*decode*` if the CSAHCA op is
  miswired, rather than silently delegating to FlashMLA.

## Benchmark

Lightweight HTTP smoke benchmark:

```bash
PORT=30000 NUM_PROMPTS=4 CONCURRENCY=1 INPUT_WORDS=64 MAX_TOKENS=16 \
bash /mnt/Data/yangpd/CSAHCA/integrations/sglang_dsv4/scripts/bench_http_chat.sh
```

For prefix-cache-safe repeated A/B runs on one service, set a unique
`PROMPT_TAG`:

```bash
PORT=30000 NUM_PROMPTS=32 CONCURRENCY=8 INPUT_WORDS=1024 MAX_TOKENS=64 \
PROMPT_TAG=ab_clean_graph_20260613 \
OUT=/mnt/Data/yangpd/CSAHCA/results/native_graph_20260613_csahca_clean_tag_http_32c8_1024x64.json \
bash /mnt/Data/yangpd/CSAHCA/integrations/sglang_dsv4/scripts/bench_http_chat.sh
```

SGLang's built-in serving benchmark can also be used:

```bash
PORT=30000 NUM_PROMPTS=16 RANDOM_INPUT_LEN=1024 RANDOM_OUTPUT_LEN=64 \
bash /mnt/Data/yangpd/CSAHCA/integrations/sglang_dsv4/scripts/bench_openai_random.sh
```

For A/B testing, run the same command against the baseline port and the patched
port, then compare output throughput, TPOT, TTFT, and request latency.

Current H100x4 clean-tag graph-mode result, `32` requests, concurrency `8`,
`input_words=1024`, `max_tokens=64`, full `2048` completion tokens:

| Service | Key env | Output tok/s | Mean latency | p90 latency |
| --- | --- | ---: | ---: | ---: |
| FlashMLA baseline | `CSAHCA_DSV4_NATIVE=0 CSAHCA_SGLANG_DSV4_PATCH=0` | 290.02 | 1.76 s | 1.80 s |
| Native CSAHCA | `CSAHCA_DSV4_NATIVE=1 CSAHCA_SGLANG_DSV4_PATCH=0` | 132.22 | 3.86 s | 3.96 s |

This proves the native CSAHCA op can be captured by decode CUDA graph, but the
current DSV4 CSAHCA kernel is still slower than FlashMLA in graph-mode serving.


## Closure Workflow

Run shadow compare without replacing model outputs:

```bash
PORT=30001 CUDA_VISIBLE_DEVICES=1,2,3,4 \
CSAHCA_DSV4_COMPARE_LAYER_IDS=all \
CSAHCA_DSV4_COMPARE_MAX_Q=8 \
CSAHCA_DSV4_COMPARE_MAX_CALLS=512 \
bash /mnt/Data/yangpd/CSAHCA/integrations/sglang_dsv4/scripts/run_shadow_compare.sh
```

Shadow compare intentionally keeps `CSAHCA_DSV4_COMPARE_REQUIRE_FINITE_Q=0`
and `CSAHCA_DSV4_COMPARE_MAX_ABS_Q=0` by default. This preserves evidence for
SGLang/FlashMLA native NaN or inactive-head lanes while still returning the
native output to clients. Replacement runs are stricter: by default they only
return CSAHCA output when `q` is finite and its absolute maximum is below
`CSAHCA_DSV4_REPLACE_MAX_ABS_Q`.

Optionally capture a small number of replayable calls. Captures include the
selected indices, attention sink, FlashMLA reference output, CSAHCA candidate
output, and packed cache tensors by default:

```bash
CSAHCA_DSV4_CAPTURE_DIR=/mnt/Data/yangpd/CSAHCA/results/dsv4_captures \
CSAHCA_DSV4_CAPTURE_LAYER_IDS=0,2,3 \
CSAHCA_DSV4_CAPTURE_MAX_Q=4 \
CSAHCA_DSV4_CAPTURE_MAX_CALLS=12 \
bash /mnt/Data/yangpd/CSAHCA/integrations/sglang_dsv4/scripts/run_shadow_compare.sh
```

Summarize live-compare logs:

```bash
python3 /mnt/Data/yangpd/CSAHCA/integrations/sglang_dsv4/scripts/summarize_live_compare.py \
  /mnt/Data/yangpd/logs/dsv4_flash/csahca_shadow_compare_*.log
```

Replay captured tensors offline:

```bash
/mnt/Data/yangpd/envs/dsv4_flash/bin/python \
  /mnt/Data/yangpd/CSAHCA/integrations/sglang_dsv4/scripts/replay_capture.py \
  /mnt/Data/yangpd/CSAHCA/results/dsv4_captures/*.pt
```
