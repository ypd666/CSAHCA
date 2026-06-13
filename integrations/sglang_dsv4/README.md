# SGLang DeepSeek-V4 Integration

This folder contains an isolated, env-gated integration hook for testing CSAHCA
inside the real SGLang DeepSeek-V4-Flash serving path.

The current CSAHCA CUDA kernel is not ABI-compatible with SGLang's production
DeepSeek-V4 attention path yet. The production path calls
`DeepseekV4AttnBackend.forward()` and then `flash_mla_with_kvcache()` with a
packed FP8 SWA cache, optional C4/C128 compressed pages, page indices, top-k
lengths, attention sinks, and FlashMLA scheduler metadata. The existing CSAHCA
kernel expects regular BF16/FP16 `q, k_cache, v_cache, selected_chunks`.

This integration therefore starts with a safe hook:

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

## Benchmark

Lightweight HTTP smoke benchmark:

```bash
PORT=30000 NUM_PROMPTS=4 CONCURRENCY=1 INPUT_WORDS=64 MAX_TOKENS=16 \
bash /mnt/Data/yangpd/CSAHCA/integrations/sglang_dsv4/scripts/bench_http_chat.sh
```

SGLang's built-in serving benchmark can also be used:

```bash
PORT=30000 NUM_PROMPTS=16 RANDOM_INPUT_LEN=1024 RANDOM_OUTPUT_LEN=64 \
bash /mnt/Data/yangpd/CSAHCA/integrations/sglang_dsv4/scripts/bench_openai_random.sh
```

For A/B testing, run the same command against the baseline port and the patched
port, then compare output throughput, TPOT, TTFT, and request latency.


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
