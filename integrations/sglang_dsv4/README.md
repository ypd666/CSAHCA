# SGLang DeepSeek-V4 Integration

This folder contains an env-gated hook for testing CSAHCA inside SGLang's
DeepSeek-V4/FlashMLA serving path.

The hook patches `DeepseekV4AttnBackend.forward()` at process startup through
`sitecustomize.py`. It can trace the real ABI, compare CSAHCA output against
FlashMLA on live tensors, or replace FlashMLA output for decode-only smoke
tests.

## Modes

| Variable | Meaning |
| --- | --- |
| `CSAHCA_DSV4_MODE=trace` | Log ABI/runtime information and delegate to FlashMLA. |
| `CSAHCA_DSV4_MODE=csahca` | Run the CSAHCA DSV4 bridge when supported, otherwise delegate. |
| `CSAHCA_DSV4_MODE=require-kernel` | Fail fast if the CSAHCA bridge cannot run. |
| `CSAHCA_DSV4_LIVE_COMPARE=1` | Compare CSAHCA output with FlashMLA and log tolerance results. |
| `CSAHCA_DSV4_REPLACE_OUTPUT=1` | Return CSAHCA output instead of FlashMLA output. Use only for controlled smoke tests. |

Supported prototype cache modes:

- `compress_ratio=0`: SWA only.
- `compress_ratio=4`: SWA + C4 sparse pages.
- `compress_ratio=128`: SWA + C128 sparse pages.

## Smoke Test

Set the SGLang source and Python environment if they are not in the defaults:

```bash
export SGLANG_SRC="${HOME}/src/sglang-main"
export PY="${HOME}/envs/dsv4_flash/bin/python"
bash integrations/sglang_dsv4/scripts/smoke_import_hook.sh
```

Expected output includes `"patched": true`.

## Launch Patched Service

Do not run this on the same GPUs while another 4-GPU service is already using
them.

```bash
PORT=30001 \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
MODEL_PATH="${HOME}/checkpoints/DeepSeek-V4-Flash-HF" \
CSAHCA_DSV4_MODE=trace \
bash integrations/sglang_dsv4/scripts/launch_patched_dsv4.sh
```

To restart a service on an existing port:

```bash
CONFIRM_RESTART=1 PORT=30000 CSAHCA_DSV4_MODE=trace \
bash integrations/sglang_dsv4/scripts/restart_patched_on_port.sh
```

Detached background variant:

```bash
CONFIRM_RESTART=1 PORT=30000 CSAHCA_DSV4_MODE=trace \
bash integrations/sglang_dsv4/scripts/restart_patched_detached.sh
```

## Live Compare

Use this before any replacement run:

```bash
CSAHCA_DSV4_MODE=csahca \
CSAHCA_DSV4_LIVE_COMPARE=1 \
CSAHCA_DSV4_REPLACE_OUTPUT=0 \
CSAHCA_DSV4_COMPARE_FORWARD_MODES=DECODE \
bash integrations/sglang_dsv4/scripts/launch_patched_dsv4.sh
```

Then send a small request and inspect log lines beginning with
`[CSAHCA][DSV4] live_compare`.

## Replacement Smoke

Replacement is intentionally explicit:

```bash
CSAHCA_DSV4_MODE=csahca \
CSAHCA_DSV4_REPLACE_OUTPUT=1 \
CSAHCA_DSV4_REPLACE_FORWARD_MODES=DECODE \
SGLANG_EXTRA_ARGS="--disable-decode-cuda-graph" \
bash integrations/sglang_dsv4/scripts/launch_patched_dsv4.sh
```

This proves functional wiring only. It is not a production speedup benchmark.

## Benchmark Helpers

Lightweight HTTP smoke benchmark:

```bash
PORT=30000 NUM_PROMPTS=4 CONCURRENCY=1 INPUT_WORDS=64 MAX_TOKENS=16 \
bash integrations/sglang_dsv4/scripts/bench_http_chat.sh
```

SGLang's built-in serving benchmark:

```bash
PORT=30000 NUM_PROMPTS=16 RANDOM_INPUT_LEN=1024 RANDOM_OUTPUT_LEN=64 \
bash integrations/sglang_dsv4/scripts/bench_openai_random.sh
```

For A/B testing, run the same workload against baseline and patched services
and compare output throughput, TPOT, TTFT, request latency, and live-compare
correctness logs.
