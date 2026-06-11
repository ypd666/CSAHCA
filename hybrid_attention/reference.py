from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SyntheticCase:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    chunk_size: int
    top_k: int


def _check_decode_shapes(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
    if q.ndim != 3:
        raise ValueError(f"q must be [batch, heads, head_dim], got {tuple(q.shape)}")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("k_cache and v_cache must be [batch, heads, seq, head_dim]")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have identical shapes")
    if q.shape[0] != k_cache.shape[0] or q.shape[1] != k_cache.shape[1]:
        raise ValueError("q and KV cache batch/head dimensions must match")
    if q.shape[2] != k_cache.shape[3]:
        raise ValueError("q head_dim must match KV head_dim")


def full_decode_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
    """Reference full single-token decode attention."""
    _check_decode_shapes(q, k_cache, v_cache)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bhd,bhsd->bhs", q.float(), k_cache.float()) * scale
    probs = torch.softmax(scores, dim=-1).to(v_cache.dtype)
    return torch.einsum("bhs,bhsd->bhd", probs, v_cache)


def chunk_mean(cache: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if cache.shape[2] % chunk_size != 0:
        raise ValueError("seq length must be divisible by chunk_size")
    batch, heads, seq_len, head_dim = cache.shape
    num_chunks = seq_len // chunk_size
    return cache.view(batch, heads, num_chunks, chunk_size, head_dim).float().mean(dim=3)


def select_topk_chunks(q: torch.Tensor, k_cache: torch.Tensor, chunk_size: int, top_k: int) -> torch.Tensor:
    """Select top-k chunks using dot product against mean K per chunk."""
    _check_decode_shapes(q, k_cache, k_cache)
    chunk_k = chunk_mean(k_cache, chunk_size)
    num_chunks = chunk_k.shape[2]
    top_k = min(top_k, num_chunks)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bhd,bhcd->bhc", q.float(), chunk_k) * scale
    return torch.topk(scores, k=top_k, dim=-1).indices.to(torch.int32)


def gather_chunk_tokens(cache: torch.Tensor, selected_chunks: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Gather selected chunk tokens from a `[B, H, S, D]` cache."""
    batch, heads, _seq_len, head_dim = cache.shape
    offsets = torch.arange(chunk_size, device=cache.device, dtype=selected_chunks.dtype)
    token_idx = selected_chunks[..., None] * chunk_size + offsets
    token_idx = token_idx.reshape(batch, heads, -1).long()
    gather_idx = token_idx[..., None].expand(batch, heads, token_idx.shape[-1], head_dim)
    return torch.gather(cache, dim=2, index=gather_idx)


def csa_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    chunk_size: int,
    top_k: int,
    selected_chunks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compressed sparse attention reference over selected chunks."""
    _check_decode_shapes(q, k_cache, v_cache)
    if selected_chunks is None:
        selected_chunks = select_topk_chunks(q, k_cache, chunk_size, top_k)
    k_selected = gather_chunk_tokens(k_cache, selected_chunks, chunk_size)
    v_selected = gather_chunk_tokens(v_cache, selected_chunks, chunk_size)
    return full_decode_attention(q, k_selected, v_selected)


def hca_reference(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, *, chunk_size: int) -> torch.Tensor:
    """Heavily compressed attention reference over dense chunk summaries."""
    _check_decode_shapes(q, k_cache, v_cache)
    k_summary = chunk_mean(k_cache, chunk_size).to(k_cache.dtype)
    v_summary = chunk_mean(v_cache, chunk_size).to(v_cache.dtype)
    return full_decode_attention(q, k_summary, v_summary)


def hybrid_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    csa_chunk_size: int,
    csa_top_k: int,
    hca_chunk_size: int,
    csa_weight: float = 0.7,
) -> torch.Tensor:
    """Simple CSA/HCA output blend for a V4-inspired mini block."""
    csa_out = csa_reference(q, k_cache, v_cache, chunk_size=csa_chunk_size, top_k=csa_top_k)
    hca_out = hca_reference(q, k_cache, v_cache, chunk_size=hca_chunk_size)
    return csa_weight * csa_out + (1.0 - csa_weight) * hca_out


def make_synthetic_case(
    *,
    batch: int = 1,
    heads: int = 8,
    seq_len: int = 4096,
    head_dim: int = 128,
    chunk_size: int = 64,
    top_k: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> SyntheticCase:
    """Create deterministic synthetic decode-attention tensors."""
    if seq_len % chunk_size != 0:
        raise ValueError("seq_len must be divisible by chunk_size")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn(batch, heads, head_dim, device=device, dtype=dtype, generator=generator)
    k_cache = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype, generator=generator)
    v_cache = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype, generator=generator)
    return SyntheticCase(q=q, k_cache=k_cache, v_cache=v_cache, chunk_size=chunk_size, top_k=top_k)

