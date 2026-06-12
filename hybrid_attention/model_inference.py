from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn

from .extension import csa_decode_forward, csa_decode_forward_tiled
from .reference import csa_reference, full_decode_attention, select_topk_chunks


class MiniCSADecodeBlock(nn.Module):
    """Tiny decode block that routes its attention through a CSA backend."""

    def __init__(
        self,
        *,
        heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.hidden_dim = heads * head_dim
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False, device=device, dtype=dtype)
        mlp_dim = int(self.hidden_dim * mlp_ratio)
        self.use_mlp = mlp_dim > 0
        if self.use_mlp:
            self.up_proj = nn.Linear(self.hidden_dim, mlp_dim, bias=False, device=device, dtype=dtype)
            self.down_proj = nn.Linear(mlp_dim, self.hidden_dim, bias=False, device=device, dtype=dtype)

    def forward(
        self,
        hidden: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        selected_chunks: torch.Tensor | None,
        *,
        backend: str,
        chunk_size: int,
        top_k: int,
        tile_size: int,
    ) -> torch.Tensor:
        batch = hidden.shape[0]
        q = self.q_proj(hidden).view(batch, self.heads, self.head_dim)
        if backend != "torch-full" and selected_chunks is None:
            selected_chunks = select_topk_chunks(q, k_cache, chunk_size, top_k)
        if backend == "torch-full":
            attn = full_decode_attention(q, k_cache, v_cache)
        elif backend == "torch-csa":
            selected_top_k = selected_chunks.shape[-1]
            attn = csa_reference(
                q,
                k_cache,
                v_cache,
                chunk_size=chunk_size,
                top_k=selected_top_k,
                selected_chunks=selected_chunks,
            )
        elif backend == "cuda-csa":
            attn = csa_decode_forward(q, k_cache, v_cache, selected_chunks, chunk_size)
        elif backend == "cuda-csa-tiled":
            attn = csa_decode_forward_tiled(q, k_cache, v_cache, selected_chunks, chunk_size, tile_size)
        else:
            raise ValueError(f"unknown backend: {backend}")

        out = self.out_proj(attn.reshape(batch, self.hidden_dim))
        if self.use_mlp:
            out = out + self.down_proj(torch.nn.functional.gelu(self.up_proj(hidden)))
        return hidden + 0.01 * out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny CSA decode block inference benchmark.")
    parser.add_argument(
        "--backend",
        choices=["torch-full", "torch-csa", "cuda-csa", "cuda-csa-tiled"],
        default="cuda-csa-tiled",
    )
    parser.add_argument("--selection", choices=["precomputed", "dynamic"], default="precomputed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--mlp-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def make_inputs(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    hidden_dim = args.heads * args.head_dim
    hidden = torch.randn(args.batch, hidden_dim, device=device, dtype=dtype, generator=generator)
    k_cache = torch.randn(
        args.batch,
        args.heads,
        args.seq_len,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    v_cache = torch.randn(
        args.batch,
        args.heads,
        args.seq_len,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return hidden, k_cache, v_cache


def run_decode(
    block: MiniCSADecodeBlock,
    hidden: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_chunks: torch.Tensor,
    args: argparse.Namespace,
    *,
    steps: int,
) -> torch.Tensor:
    for _ in range(steps):
        step_chunks = None if args.selection == "dynamic" else selected_chunks
        hidden = block(
            hidden,
            k_cache,
            v_cache,
            step_chunks,
            backend=args.backend,
            chunk_size=args.chunk_size,
            top_k=args.top_k,
            tile_size=args.tile_size,
        )
    return hidden


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    torch.manual_seed(args.seed)
    block = MiniCSADecodeBlock(
        heads=args.heads,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
        mlp_ratio=args.mlp_ratio,
    ).eval()
    hidden, k_cache, v_cache = make_inputs(args, dtype, device)
    with torch.no_grad():
        q0 = block.q_proj(hidden).view(args.batch, args.heads, args.head_dim)
        selected_chunks = select_topk_chunks(q0, k_cache, args.chunk_size, args.top_k)
        run_decode(block, hidden.clone(), k_cache, v_cache, selected_chunks, args, steps=args.warmup)
        if device.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            final_hidden = run_decode(block, hidden.clone(), k_cache, v_cache, selected_chunks, args, steps=args.steps)
            end.record()
            torch.cuda.synchronize()
            total_ms = start.elapsed_time(end)
        else:
            import time

            begin = time.perf_counter()
            final_hidden = run_decode(block, hidden.clone(), k_cache, v_cache, selected_chunks, args, steps=args.steps)
            total_ms = (time.perf_counter() - begin) * 1000.0

    latency_ms = total_ms / args.steps
    tokens_per_s = args.batch * args.steps / (total_ms / 1000.0)
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    selected_tokens = args.seq_len if args.backend == "torch-full" else args.top_k * args.chunk_size
    kv_bytes = args.batch * args.heads * selected_tokens * args.head_dim * dtype_bytes * 2
    row = {
        "backend": args.backend,
        "uses_custom_kernel": args.backend.startswith("cuda-"),
        "selection": args.selection,
        "dtype": args.dtype,
        "batch": args.batch,
        "heads": args.heads,
        "seq_len": args.seq_len,
        "head_dim": args.head_dim,
        "chunk_size": args.chunk_size,
        "top_k": args.top_k,
        "tile_size": args.tile_size,
        "steps": args.steps,
        "mlp_ratio": args.mlp_ratio,
        "attention_tokens_per_step": selected_tokens,
        "latency_ms_per_token": f"{latency_ms:.4f}",
        "tokens_per_s": f"{tokens_per_s:.2f}",
        "output_checksum": f"{final_hidden.float().mean().item():.6f}",
        "attention_kv_gbps": f"{kv_bytes / (latency_ms / 1000.0) / 1e9:.2f}",
    }
    print(row)
    if args.out is not None:
        append_csv(args.out, row)


if __name__ == "__main__":
    main()
