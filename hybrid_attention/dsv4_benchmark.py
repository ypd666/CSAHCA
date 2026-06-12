from __future__ import annotations

import argparse
import math

import torch

from .dsv4_correctness import HEAD_DIM, pack_dsv4_cache
from .extension import dsv4_sparse_decode_forward, dsv4_swa_decode_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark prototype DSV4 SWA paged FP8 decode ABI.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--num-queries", type=int, default=136)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--compress-ratio", type=int, choices=[0, 4, 128], default=0)
    parser.add_argument("--extra-tokens", type=int, default=4096)
    parser.add_argument("--extra-top-k", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    q = torch.randn(args.num_queries, args.heads, HEAD_DIM, dtype=dtype, device=device, generator=generator)
    k_tokens = torch.randn(args.num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device, generator=generator)
    cache = pack_dsv4_cache(k_tokens, page_size=args.page_size)
    token_indices = torch.randint(
        0,
        args.num_tokens,
        (args.num_queries, args.top_k),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    topk_lengths = torch.full((args.num_queries,), args.top_k, dtype=torch.int32, device=device)
    attn_sink = torch.randn(args.heads, dtype=torch.float32, device=device, generator=generator)
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    extra_cache = None
    extra_token_indices = None
    extra_topk_lengths = None
    extra_page_size = None
    if args.compress_ratio:
        extra_page_size = args.page_size // args.compress_ratio
        extra_top_k = args.extra_top_k or (512 if args.compress_ratio == 4 else 128)
        extra_tokens = torch.randn(
            args.extra_tokens,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        extra_cache = pack_dsv4_cache(extra_tokens, page_size=extra_page_size)
        extra_token_indices = torch.randint(
            0,
            args.extra_tokens,
            (args.num_queries, extra_top_k),
            dtype=torch.int32,
            device=device,
            generator=generator,
        )
        extra_topk_lengths = torch.full((args.num_queries,), extra_top_k, dtype=torch.int32, device=device)

    def run_kernel() -> torch.Tensor:
        if args.compress_ratio:
            assert extra_cache is not None
            assert extra_token_indices is not None
            assert extra_topk_lengths is not None
            assert extra_page_size is not None
            return dsv4_sparse_decode_forward(
                q,
                cache,
                token_indices,
                topk_lengths,
                extra_cache,
                extra_token_indices,
                extra_topk_lengths,
                attn_sink,
                args.page_size,
                extra_page_size,
                softmax_scale,
            )
        return dsv4_swa_decode_forward(
            q,
            cache,
            token_indices,
            topk_lengths,
            attn_sink,
            args.page_size,
            softmax_scale,
        )

    with torch.no_grad():
        for _ in range(args.warmup):
            run_kernel()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            out = run_kernel()
        end.record()
        torch.cuda.synchronize()

    latency_ms = start.elapsed_time(end) / args.iters
    query_heads = args.num_queries * args.heads
    print(
        {
            "mode": "dsv4-swa-prototype",
            "dtype": args.dtype,
            "compress_ratio": args.compress_ratio,
            "num_queries": args.num_queries,
            "heads": args.heads,
            "head_dim": HEAD_DIM,
            "top_k": args.top_k,
            "extra_top_k": args.extra_top_k or (512 if args.compress_ratio == 4 else 128 if args.compress_ratio else 0),
            "page_size": args.page_size,
            "extra_page_size": extra_page_size,
            "latency_ms": f"{latency_ms:.4f}",
            "query_heads_per_s": f"{query_heads / (latency_ms / 1000.0):.2f}",
            "checksum": f"{out.float().mean().item():.6f}",
        }
    )


if __name__ == "__main__":
    main()
