from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Callable

import torch

from .extension import csa_decode_forward
from .reference import (
    csa_reference,
    full_decode_attention,
    hca_reference,
    hybrid_reference,
    make_synthetic_case,
    select_topk_chunks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CSA/HCA decode attention variants.")
    parser.add_argument("--mode", choices=["torch-full", "torch-csa", "torch-hca", "torch-hybrid", "cuda-csa"], default="torch-csa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--hca-chunk-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--nvtx", action="store_true")
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_timed(fn: Callable[[], torch.Tensor], *, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def wall_timed(fn: Callable[[], torch.Tensor], *, iters: int) -> float:
    begin = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - begin) * 1000.0 / iters


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    case = make_synthetic_case(
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        dtype=dtype,
        device=args.device,
        seed=args.seed,
    )
    selected = select_topk_chunks(case.q, case.k_cache, case.chunk_size, case.top_k)

    with torch.no_grad():
        if args.mode == "torch-full":
            fn = lambda: full_decode_attention(case.q, case.k_cache, case.v_cache)
            selected_tokens = args.seq_len
        elif args.mode == "torch-csa":
            fn = lambda: csa_reference(
                case.q,
                case.k_cache,
                case.v_cache,
                chunk_size=case.chunk_size,
                top_k=case.top_k,
                selected_chunks=selected,
            )
            selected_tokens = args.top_k * args.chunk_size
        elif args.mode == "torch-hca":
            fn = lambda: hca_reference(case.q, case.k_cache, case.v_cache, chunk_size=args.hca_chunk_size)
            selected_tokens = args.seq_len // args.hca_chunk_size
        elif args.mode == "torch-hybrid":
            fn = lambda: hybrid_reference(
                case.q,
                case.k_cache,
                case.v_cache,
                csa_chunk_size=case.chunk_size,
                csa_top_k=case.top_k,
                hca_chunk_size=args.hca_chunk_size,
            )
            selected_tokens = args.top_k * args.chunk_size + args.seq_len // args.hca_chunk_size
        else:
            fn = lambda: csa_decode_forward(case.q, case.k_cache, case.v_cache, selected, case.chunk_size)
            selected_tokens = args.top_k * args.chunk_size

        for _ in range(args.warmup):
            fn()
        synchronize(args.device)

        if args.nvtx and args.device == "cuda":
            torch.cuda.nvtx.range_push(args.mode)
        if args.device == "cuda":
            latency_ms = cuda_timed(fn, iters=args.iters)
        else:
            latency_ms = wall_timed(fn, iters=args.iters)
        if args.nvtx and args.device == "cuda":
            torch.cuda.nvtx.range_pop()

    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    kv_bytes = args.batch * args.heads * selected_tokens * args.head_dim * dtype_bytes * 2
    effective_gbps = kv_bytes / (latency_ms / 1000.0) / 1e9
    row = {
        "mode": args.mode,
        "device": args.device,
        "dtype": args.dtype,
        "batch": args.batch,
        "heads": args.heads,
        "seq_len": args.seq_len,
        "head_dim": args.head_dim,
        "chunk_size": args.chunk_size,
        "top_k": args.top_k,
        "latency_ms": f"{latency_ms:.4f}",
        "effective_gbps": f"{effective_gbps:.2f}",
    }
    print(row)
    if args.out is not None:
        append_csv(args.out, row)


if __name__ == "__main__":
    main()

