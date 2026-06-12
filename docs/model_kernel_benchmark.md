# Model-Level Kernel Benchmark

Date: 2026-06-12

Host: one H100 machine

GPU scope: one GPU, `CUDA_VISIBLE_DEVICES=0`

This benchmark places the CSAHCA CUDA kernel inside a small decode block:

```text
hidden -> q_proj -> attention backend -> out_proj -> optional MLP -> residual
```

It is model-level inference code, not a naked kernel microbenchmark. It is not
yet a DeepSeek-V4/SGLang production replacement, because SGLang DeepSeek-V4 uses
packed FP8 paged KV caches and FlashMLA metadata, while the current CSAHCA
kernel expects regular BF16/FP16 `[batch, heads, seq, head_dim]` KV tensors and
explicit selected chunk ids.

## Command

```bash
cd /path/to/CSAHCA
CUDA_VISIBLE_DEVICES=0 \
SEQ_LEN=32768 \
STEPS=200 \
WARMUP=20 \
MLP_RATIO=2.0 \
bash scripts/run_model_kernel_ab.sh
```

Backends:

- `torch-full`: dense full-attention baseline, no CSAHCA kernel.
- `torch-csa`: same sparse CSA algorithm in PyTorch, no CSAHCA kernel.
- `cuda-csa-tiled`: same sparse CSA algorithm using the CSAHCA tiled CUDA kernel.

Common shape:

```text
batch=1, heads=8, head_dim=128, seq_len=32768,
chunk_size=64, top_k=8, dtype=bfloat16
```

## Results

### Decode Block, MLP Ratio 2.0

Generated CSV:

```text
results/model_kernel_ab_h100_gpu0_20260612_135911.csv
```

| Backend | Custom kernel | Latency / token | Tokens/s | Speedup vs cuda-csa-tiled |
| --- | --- | ---: | ---: | ---: |
| torch-full | no | 0.3304 ms | 3026.30 | 2.706x slower |
| torch-csa | no | 0.3311 ms | 3019.94 | 2.712x slower |
| cuda-csa-tiled | yes | 0.1221 ms | 8188.10 | 1.000x |

### Attention-Dominated, MLP Ratio 0.0

Generated CSV:

```text
results/model_kernel_ab_h100_gpu0_attention_only_20260612_135929.csv
```

| Backend | Custom kernel | Latency / token | Tokens/s | Speedup vs cuda-csa-tiled |
| --- | --- | ---: | ---: | ---: |
| torch-full | no | 0.3077 ms | 3249.84 | 4.147x slower |
| torch-csa | no | 0.2807 ms | 3563.07 | 3.783x slower |
| cuda-csa-tiled | yes | 0.0742 ms | 13482.03 | 1.000x |

### Dynamic Chunk Selection, MLP Ratio 0.0

Generated CSV:

```text
results/model_kernel_ab_h100_gpu0_dynamic_20260612_135948.csv
```

| Backend | Custom kernel | Latency / token | Tokens/s | Speedup vs cuda-csa-tiled |
| --- | --- | ---: | ---: | ---: |
| torch-full | no | 0.3120 ms | 3204.94 | 1.040x slower |
| torch-csa | no | 0.3858 ms | 2592.19 | 1.286x slower |
| cuda-csa-tiled | yes | 0.3001 ms | 3332.54 | 1.000x |

## Interpretation

With precomputed selected chunks, the CUDA kernel improves the decode block by
about `2.7x` when projection/MLP work is included, and about `3.8x-4.1x` in the
attention-dominated setting.

When chunk selection is recomputed in PyTorch every step, the end-to-end benefit
drops to about `1.3x` over PyTorch CSA. That means the next practical target is
not only the attention kernel: the selector/indexer path must be optimized or
reused from a serving stack such as SGLang's DSV4 indexer.
