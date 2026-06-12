from __future__ import annotations

import argparse
import math

import torch

from .extension import dsv4_sparse_decode_forward, dsv4_swa_decode_forward

DIM_NOPE = 448
DIM_ROPE = 64
HEAD_DIM = DIM_NOPE + DIM_ROPE
TILE_SIZE = 64
NUM_SCALE_TILES = DIM_NOPE // TILE_SIZE
NOPE_ROPE_BYTES = DIM_NOPE + DIM_ROPE * 2
SCALE_BYTES_PER_TOKEN = NUM_SCALE_TILES + 1


def bytes_per_page(page_size: int) -> int:
    raw = page_size * (NOPE_ROPE_BYTES + SCALE_BYTES_PER_TOKEN)
    return ((raw + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES) * NOPE_ROPE_BYTES


def pack_dsv4_cache(k_tokens: torch.Tensor, *, page_size: int) -> torch.Tensor:
    """Pack `[tokens, 512]` BF16 keys into the DSV4 SWA byte layout."""

    if k_tokens.dtype != torch.bfloat16:
        raise ValueError("k_tokens must be bfloat16")
    num_tokens = k_tokens.shape[0]
    num_pages = (num_tokens + page_size - 1) // page_size
    cache = torch.zeros(
        (num_pages, bytes_per_page(page_size)),
        dtype=torch.uint8,
        device=k_tokens.device,
    )
    fp8_info = torch.finfo(torch.float8_e4m3fn)

    for loc in range(num_tokens):
        page = loc // page_size
        in_page = loc % page_size
        data_base = in_page * NOPE_ROPE_BYTES
        scale_base = page_size * NOPE_ROPE_BYTES + in_page * SCALE_BYTES_PER_TOKEN

        nope = k_tokens[loc, :DIM_NOPE].float()
        for tile in range(NUM_SCALE_TILES):
            tile_start = tile * TILE_SIZE
            vals = nope[tile_start : tile_start + TILE_SIZE]
            max_abs = vals.abs().max().clamp_min(1e-8)
            scale = 2.0 ** math.ceil(math.log2(float(max_abs / fp8_info.max)))
            exponent = int(round(math.log2(scale)))
            scale_u8 = max(0, min(255, exponent + 127))
            quant = (vals / scale).clamp(float(fp8_info.min), float(fp8_info.max)).to(torch.float8_e4m3fn)
            cache[page, data_base + tile_start : data_base + tile_start + TILE_SIZE] = quant.view(torch.uint8)
            cache[page, scale_base + tile] = scale_u8

        rope_bytes = k_tokens[loc, DIM_NOPE:].contiguous().view(torch.uint8)
        cache[page, data_base + DIM_NOPE : data_base + NOPE_ROPE_BYTES] = rope_bytes

    return cache


def dequant_dsv4_cache(cache: torch.Tensor, token_indices: torch.Tensor, *, page_size: int) -> torch.Tensor:
    flat_u8 = cache.contiguous().view(torch.uint8).reshape(-1)
    flat_fp8 = cache.contiguous().view(torch.float8_e4m3fn).reshape(-1)
    flat_bf16 = cache.contiguous().view(torch.bfloat16).reshape(-1)
    bpp = cache.shape[1]
    loc = token_indices.to(torch.int64).reshape(-1)
    page = loc // page_size
    in_page = loc % page_size
    page_base = page * bpp
    data_base = page_base + in_page * NOPE_ROPE_BYTES
    scale_base = page_base + page_size * NOPE_ROPE_BYTES + in_page * SCALE_BYTES_PER_TOKEN

    device = cache.device
    nope_idx = data_base[:, None] + torch.arange(DIM_NOPE, device=device)[None, :]
    nope = flat_fp8[nope_idx].float()
    scale_idx = scale_base[:, None] + torch.arange(NUM_SCALE_TILES, device=device)[None, :]
    scale = torch.exp2((flat_u8[scale_idx].to(torch.int32) - 127).float())
    nope = nope * scale.repeat_interleave(TILE_SIZE, dim=1)

    rope_base = (data_base + DIM_NOPE) // 2
    rope_idx = rope_base[:, None] + torch.arange(DIM_ROPE, device=device)[None, :]
    rope = flat_bf16[rope_idx]

    out = torch.empty((loc.numel(), HEAD_DIM), dtype=torch.bfloat16, device=device)
    out[:, :DIM_NOPE] = nope.to(torch.bfloat16)
    out[:, DIM_NOPE:] = rope
    return out.reshape(*token_indices.shape, HEAD_DIM)


def reference_attention(
    q: torch.Tensor,
    cache: torch.Tensor,
    token_indices: torch.Tensor,
    topk_lengths: torch.Tensor,
    attn_sink: torch.Tensor | None,
    *,
    page_size: int,
    softmax_scale: float,
    extra_cache: torch.Tensor | None = None,
    extra_token_indices: torch.Tensor | None = None,
    extra_topk_lengths: torch.Tensor | None = None,
    extra_page_size: int | None = None,
) -> torch.Tensor:
    num_queries, heads, _ = q.shape
    out = torch.empty_like(q)

    def gather_valid(
        source_cache: torch.Tensor,
        indices: torch.Tensor,
        length: int,
        source_page_size: int,
    ) -> torch.Tensor:
        slots = indices[:length]
        slots = slots[slots >= 0]
        if slots.numel() == 0:
            return torch.empty((0, HEAD_DIM), dtype=torch.bfloat16, device=q.device)
        return dequant_dsv4_cache(source_cache, slots.reshape(1, -1), page_size=source_page_size).reshape(-1, HEAD_DIM)

    for qi in range(num_queries):
        valid = max(0, min(int(topk_lengths[qi].item()), token_indices.shape[1]))
        pieces = [gather_valid(cache, token_indices[qi], valid, page_size)]
        if extra_cache is not None and extra_token_indices is not None:
            if extra_page_size is None:
                raise ValueError("extra_page_size is required with extra_cache")
            if extra_topk_lengths is None:
                extra_valid = extra_token_indices.shape[1]
            else:
                extra_valid = max(0, min(int(extra_topk_lengths[qi].item()), extra_token_indices.shape[1]))
            pieces.append(gather_valid(extra_cache, extra_token_indices[qi], extra_valid, extra_page_size))
        valid_k = torch.cat(pieces, dim=0).float()
        for h in range(heads):
            scores = torch.mv(valid_k, q[qi, h].float()) * softmax_scale
            if attn_sink is not None:
                scores = torch.cat([attn_sink[h].reshape(1), scores])
                values = torch.cat([torch.zeros(1, HEAD_DIM, device=q.device), valid_k], dim=0)
            else:
                values = valid_k
            probs = torch.softmax(scores, dim=0)
            out[qi, h] = torch.mv(values.t(), probs).to(q.dtype)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check DSV4 SWA FP8 paged-cache CUDA ABI.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--num-queries", type=int, default=7)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--compress-ratio", type=int, choices=[0, 4, 128], default=0)
    parser.add_argument("--extra-tokens", type=int, default=1024)
    parser.add_argument("--extra-top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=4e-2)
    parser.add_argument("--rtol", type=float, default=4e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
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
    topk_lengths = torch.randint(
        max(1, args.top_k // 2),
        args.top_k + 1,
        (args.num_queries,),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    attn_sink = torch.randn(args.heads, dtype=torch.float32, device=device, generator=generator)
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)

    extra_cache = None
    extra_token_indices = None
    extra_topk_lengths = None
    extra_page_size = None
    if args.compress_ratio:
        extra_page_size = args.page_size // args.compress_ratio
        if extra_page_size <= 0:
            raise ValueError("extra_page_size must be positive")
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
        if extra_top_k >= 4:
            extra_token_indices[:, -2:] = -1
        extra_topk_lengths = torch.randint(
            max(1, extra_top_k // 2),
            extra_top_k + 1,
            (args.num_queries,),
            dtype=torch.int32,
            device=device,
            generator=generator,
        )
        got = dsv4_sparse_decode_forward(
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
    else:
        got = dsv4_swa_decode_forward(
            q,
            cache,
            token_indices,
            topk_lengths,
            attn_sink,
            args.page_size,
            softmax_scale,
        )
    expected = reference_attention(
        q,
        cache,
        token_indices,
        topk_lengths,
        attn_sink,
        page_size=args.page_size,
        softmax_scale=softmax_scale,
        extra_cache=extra_cache,
        extra_token_indices=extra_token_indices,
        extra_topk_lengths=extra_topk_lengths,
        extra_page_size=extra_page_size,
    )

    max_abs = (got.float() - expected.float()).abs().max().item()
    max_rel = ((got.float() - expected.float()).abs() / expected.float().abs().clamp_min(1e-6)).max().item()
    ok = torch.allclose(got, expected, atol=args.atol, rtol=args.rtol)
    print(f"max_abs={max_abs:.6f} max_rel={max_rel:.6f} allclose={ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
