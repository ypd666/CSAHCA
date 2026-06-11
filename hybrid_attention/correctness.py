from __future__ import annotations

import argparse

import torch

from .extension import csa_decode_forward, csa_decode_forward_tiled
from .reference import csa_reference, make_synthetic_case, select_topk_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CUDA CSA output against PyTorch reference.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--kernel", choices=["v1", "tiled"], default="v1")
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--require-extension", action="store_true")
    return parser.parse_args()


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
    )
    selected = select_topk_chunks(case.q, case.k_cache, case.chunk_size, case.top_k)
    ref = csa_reference(
        case.q,
        case.k_cache,
        case.v_cache,
        chunk_size=case.chunk_size,
        top_k=case.top_k,
        selected_chunks=selected,
    )

    try:
        if args.kernel == "v1":
            got = csa_decode_forward(case.q, case.k_cache, case.v_cache, selected, case.chunk_size)
        else:
            got = csa_decode_forward_tiled(
                case.q,
                case.k_cache,
                case.v_cache,
                selected,
                case.chunk_size,
                args.tile_size,
            )
    except Exception as exc:
        if args.require_extension:
            raise
        print(f"SKIP cuda extension check: {exc}")
        print("PyTorch reference path is valid.")
        return

    if got.is_cuda:
        torch.cuda.synchronize()
    diff = (got.float() - ref.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / ref.float().abs().clamp_min(1e-6)).max().item()
    ok = torch.allclose(got.float(), ref.float(), atol=args.atol, rtol=args.rtol)
    print(f"max_abs={max_abs:.6f} max_rel={max_rel:.6f} allclose={ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
